论文标题： GPT Understands, Too  

论文地址：[https://arxiv.org/pdf/2103.10385.pdf](https://arxiv.org/pdf/2103.10385.pdf)

代码地址：[https://github.com/THUDM/P-tuning](https://github.com/THUDM/P-tuning)
![image.png](https://cdn.nlark.com/yuque/0/2023/png/21475317/1683707008185-c2d4c25e-2d30-4be5-8388-81e498649ed4.png#averageHue=%23f9f5f2&clientId=u9c0f55dd-a856-4&from=paste&height=361&id=u85a0d7c1&originHeight=361&originWidth=1062&originalType=binary&ratio=1&rotation=0&showTitle=false&size=121637&status=done&style=none&taskId=u398ca158-c01e-44ac-9f6c-66641c1c910&title=&width=1062)











## 建模

在prompt-based场景下，则通常将下游任务转化为Mask Language Model任务，因此此时不需要引入额外的参数，但需要明确一个prompt模板。作者认为一个模板 T 就可以表示为一个token序列：

![image.png](https://cdn.nlark.com/yuque/0/2023/png/27092258/1683945139861-6b60fe8b-3a23-48f1-86d3-31ee3f4e5cf8.png#averageHue=%23f8f8f8&clientId=u53cf9565-7b8c-4&from=paste&height=60&id=ua5f951f5&originHeight=120&originWidth=498&originalType=binary&ratio=2&rotation=0&showTitle=false&size=12339&status=done&style=none&taskId=u706c707f-de0f-4b7c-8afb-92133c70290&title=&width=249)





在P-tuning中，则将模板中的 映射为一个可训练的参数（如上图所示），此时这部分的token则称为pseudo token（有的工作也叫做soft-prompt、virtual token等)。在优化过程中，认为这部分pseudo token也存在序列关系，因此使用双向LSTM对模板T中的pseudo token序列进行表征，则可以使用梯度下降法更新连续的参数。

整个pipeline如下：
![image.png](https://cdn.nlark.com/yuque/0/2023/png/27092258/1683945388374-fc861e18-30c7-4a23-9059-da7b3328ba0e.png#averageHue=%23f3f3f3&clientId=u53cf9565-7b8c-4&from=paste&height=230&id=u37bcded6&originHeight=460&originWidth=1916&originalType=binary&ratio=2&rotation=0&showTitle=false&size=172209&status=done&style=none&taskId=uaeca8218-7fb0-46af-b505-cd0547b5566&title=&width=958)







我个人理解，模型训练的其实是LSTM，也就是用于表征pseudo token的特征提取器。
文章并没有提及pseudo token的选择策略。

## 代码部分
prompt的构建
```python
import torch
import torch.nn as nn


class PromptEncoder(torch.nn.Module):
    def __init__(self, template, hidden_size, tokenizer, device, args):
        super().__init__()
        self.device = device
        self.spell_length = sum(template)
        self.hidden_size = hidden_size
        self.tokenizer = tokenizer
        self.args = args
        # ent embedding
        self.cloze_length = template
        self.cloze_mask = [
            [1] * self.cloze_length[0]  # first cloze
            + [1] * self.cloze_length[1]  # second cloze
            + [1] * self.cloze_length[2]  # third cloze
        ]
        self.cloze_mask = torch.LongTensor(self.cloze_mask).bool().to(self.device)

        self.seq_indices = torch.LongTensor(list(range(len(self.cloze_mask[0])))).to(self.device)
        # embedding
        self.embedding = torch.nn.Embedding(len(self.cloze_mask[0]), self.hidden_size).to(self.device)
        # LSTM
        self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,
                                       dropout=self.args.lstm_dropout,
                                       bidirectional=True,
                                       batch_first=True)
        self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size))
        print("init prompt encoder...")

    def forward(self):
        input_embeds = self.embedding(self.seq_indices).unsqueeze(0)
        output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0]).squeeze()
        return output_embeds
```


