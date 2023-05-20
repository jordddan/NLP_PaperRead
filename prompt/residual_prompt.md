# Residual Prompt Tuning: Improving Prompt Tuning with Residual Reparameterization  
论文地址：[https://arxiv.org/pdf/2305.03937.pdf](https://arxiv.org/pdf/2305.03937.pdf)

代码地址：[https://github.com/arazd/ResidualPrompts](https://github.com/arazd/ResidualPrompts)

![image.png](https://cdn.nlark.com/yuque/0/2023/png/21475317/1683707179197-48ecfe5f-4074-4244-9fdb-0258b5f6c587.png#averageHue=%23f0efed&clientId=u5fd6f948-5028-4&from=paste&height=480&id=u8eb6bd6a&originHeight=480&originWidth=486&originalType=binary&ratio=1&rotation=0&showTitle=false&size=80482&status=done&style=none&taskId=u4e14927a-c060-48c9-9af6-1856dfc953a&title=&width=486)
### Prompt构造方法：
-----------------------

prompt训练代码
```python
    def train_step_lester(self,
                          batch,
                          task=None,
                          get_pred=False):
        prefix_len = self.prefix_len
        model = self.model
        embed_prompt = self.prefix_MLP!=None
        if embed_prompt:
            mlp = self.prefix_MLP
        tokenizer = self.tokenizer

        batch = {k: batch[k].to(self.device) for k in batch}

        inputs_embeds = model.encoder.embed_tokens(batch["source_ids"])

        k = inputs_embeds.shape[0]
        if embed_prompt:
            if self.separate_mlps==False:
                prompt = mlp(model.prompt)
            else:
                prompt = self.pass_separate_mlps()
        else:
            prompt = model.prompt
        inputs_embeds = torch.concat([prompt.repeat(k, 1, 1),
                                      inputs_embeds], axis=1)[:,:self.seq_len]
        full_prefix_len = prompt.shape[0]
        source_mask_updated = torch.concat( (batch["source_mask"][0][0].repeat(k,full_prefix_len),
                                             batch["source_mask"]), axis=1)[:,:self.seq_len]
        encoder_outputs = model.encoder(
                                attention_mask=source_mask_updated,
                                inputs_embeds=inputs_embeds,
                                head_mask=None,
                                output_attentions=None,
                                output_hidden_states=None,
                                return_dict=None,
                            )
```
Residual MLP代码
```python
class ResMLP(torch.nn.Module):
    def __init__(self,
                 bottleneck_size,
                 module_type='MLP1',
                 emb_dimension=512,
                 nonlinearity='relu', # activation function
                 layer_norm=True,
                 dropout=0.0,
                 residual=True,
                 ):
        """MLP class for soft prompt re-parameterization. MLP can have a Residual connection.
        Args:
            bottleneck_size (int): Dimension of the MLP bottlenack.
            module_type (str, optional): Type of MLP to be used.
                Currently supports 1-layer and 2-layer MLPs, and simple transformer layer ('MLP1'/'MLP2'/'transformer').
                Defaults to 'MLP1'.
            emb_dimension (int, optional): Dimension of T5 model embeddings. Defaults to 512 (T5-small embedding dimension).
            residual (bool, optional): Whether to use residual connection in MLP. Defaults to True.
        """
        super().__init__()
        assert module_type in ['MLP1', 'MLP2', 'transformer', 'LSTM', 'LSTM1', 'LSTM2']
        assert nonlinearity in ['relu', 'tanh', 'sigm']

        self.module_type = module_type

        if module_type not in ['LSTM', 'LSTM1', 'LSTM2', 'transformer']:
            layers = [nn.Linear(emb_dimension, bottleneck_size)]

            if nonlinearity=='relu':
                layers.append(nn.ReLU())
            elif nonlinearity=='tanh':
                layers.append(nn.Tanh())
            elif nonlinearity=='sigm':
                layers.append(nn.Sigmoid())

            layers.append(nn.Linear(bottleneck_size, emb_dimension))

            if dropout>0:
                layers.append(nn.Dropout(p=dropout))
            if layer_norm:
                layers.append(nn.LayerNorm(emb_dimension))

            if module_type=='MLP2':
                layers = layers + layers # repeat twice
            self.module = torch.nn.Sequential(*layers)

        elif module_type in ['LSTM1', 'LSTM2', 'LSTM']:
            self.lstm_head = torch.nn.LSTM(input_size=emb_dimension,
                                           hidden_size=emb_dimension // 2,
                                           num_layers=1 if module_type=='LSTM1' else 2,
                                           dropout=0.05,
                                           bidirectional=True,
                                           batch_first=True)
            self.mlp_head = nn.Sequential(nn.Linear(emb_dimension, emb_dimension),
                                          nn.ReLU(),
                                          nn.Linear(emb_dimension, emb_dimension))


        elif module_type=='transformer':
            device = 'cuda'
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dimension, nhead=2, dropout=0.05).to(device)
            self.module = nn.TransformerEncoder(self.encoder_layer, num_layers=2).to(device)

        self.residual = residual
        if self.residual:
            print('Using skip connection in MLP')

    def forward(self, inputs):
        if self.module_type=='LSTM':
            output_embeds = self.mlp_head(self.lstm_head(inputs)[0]).squeeze()
        elif self.module_type in ['LSTM1', 'LSTM2']:
            output_embeds = self.lstm_head(inputs)[0].squeeze()
            if self.residual:
                output_embeds += inputs
            return output_embeds

        if self.residual:
            return self.module(inputs) + inputs
        else:
            return self.module(inputs)

```
与最初版本的prompt tuning 方法相似，只是将prompt过了含有residual connect的MLP

其中有两种MLP的方法
- 对于每个prompt token embedding，用share的MLP映射
- 对于每个prompt token embedding，初始化一个对应的MLP

### Prompt初始化方法：
--------------

使用sampled most common vocab embedding

