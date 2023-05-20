论文标题： P-Tuning: Prompt Tuning Can Be Comparable to Fine-tuning Across Scales and Tasks  
![image.png](https://cdn.nlark.com/yuque/0/2023/png/21475317/1683707023533-9a568512-dcd4-464d-af6c-194cd6ea6cee.png#averageHue=%23f6f3f0&clientId=u2071a295-cfe8-4&from=paste&height=300&id=uec5676eb&originHeight=300&originWidth=973&originalType=binary&ratio=1&rotation=0&showTitle=false&size=109312&status=done&style=none&taskId=uf8d6e92a-b9a6-4d99-8516-92e136490a8&title=&width=973)
论文地址：[https://arxiv.org/abs/2110.07602](https://arxiv.org/abs/2110.07602)
代码地址：[https://github.com/THUDM/P-tuning-v2](https://github.com/THUDM/P-tuning-v2)





## Prompt构造代码

```python
def get_prompt(self, batch_size):
    prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
    past_key_values = self.prefix_encoder(prefix_tokens)
    bsz, seqlen, _ = past_key_values.shape
    past_key_values = past_key_values.view(
        bsz,
        seqlen,
        self.n_layer * 2, 
        self.n_head,
        self.n_embd
    )
    past_key_values = self.dropout(past_key_values)
    past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
    return past_key_values

prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
```
p-tuning-v2的主要贡献是在原本的输入前添加自定义长度的layer prompts，在后续针对下游任务的训练中冻结BERT模型的所有参数而只训练这些prompts。
直觉上的实现方法是生成N个自定义长度的sequence，然后进行embedding，再和原先的模型拼接。在huggingface中这样的操作应该是比较难做到的。
而P-tuning-v2的源码提供了我认为非常巧妙的实现。简化后（为了自用而写的）的实现方法如下：
```python
class PrefixEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(seq_len, dim_ebd)
        self.trans = torch.nn.Sequential(
            torch.nn.Linear(dim_ebd, dim_ebd),
            torch.nn.Tanh(),
            torch.nn.Linear(dim_ebd, num_layer * 2 * dim_ebd)
        ).to(device)
    def forward(self, prefix):
        prefix_tokens = self.embedding(prefix)
        past_key_values = self.trans(prefix_tokens)
        return past_key_values
```
这一步是获得图（b）中的黄色部分，维度是num_layer * 2 * dim_ebd，乘以2是因为本质上传入的是key和value。
## 模型主体
```python
config = BertConfig.from_pretrained("path")
class BertPrefixForQuestionAnswering(BertPreTrainedModel):
    def __init__(self):
        super().__init__(config)
        self.num_labels = 2
        self.bert = BertModel.from_pretrained("path", add_pooling_layer=False)
        self.path = "path"
        self.bert.load_state_dict(torch.load(self.path))
        self.qa_outputs = torch.nn.Linear(1024, self.num_labels)
        self.dropout = torch.nn.Dropout(0.3)
        self.prefix_encoder = PrefixEncoder()
        self.pre_seq_len = 15
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()

        for param in self.bert.parameters():
            param.requires_grad=False


    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz,
            seqlen,
            num_layer * 2,
            num_head,
            dim_ebd//num_head
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(self, input_ids, token_type_ids, attention_mask):

        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        outputs = self.bert(input_ids = input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values)

        logits = self.qa_outputs(outputs[0])
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        return start_logits, end_logits
```
**get_prompt**函数没什么特别的，就是把**past_key_values**调整到了需要的格式。官方文件是这样描述的：
> **past_key_values**(tuple(tuple(torch.FloatTensor))of lengthconfig.n_layerswith each tuple having 4 tensors of shape(batch_size, num_heads, sequence_length - 1, embed_size_per_head)) — Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

**forward**函数则是把**attention_mask**和**prefix_attention_mask**拼接后，将一系列获取到的参数直接输入到模型中。在此变完成了对p-tuning v2的实现。

源码层面的解答，可以参考：[https://zhuanlan.zhihu.com/p/459305102](https://zhuanlan.zhihu.com/p/459305102)
## 参数和初始化细节
文中关于prompt length、是否对input embedding表示之后继续使用MLP改变表示做了探讨。得出的结论是：根据不同的任务选择不同的策略，分别能达到最好的效果，设置并不通用。
文中**没有**提及具体的初始化信息，但是提到了可以根据Ppt: Pre-trained prompt tuning for few-shot learning所提出的方法，通过优化初始化来提升效果。
一篇阅读笔记如下:
[论文笔记：PPT: Pre-trained Prompt Tuning for Few-shot Learning_北在哪的博客-CSDN博客](https://blog.csdn.net/qq_43183860/article/details/120796864)

