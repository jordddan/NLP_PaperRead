# Prefix-Tuning: Optimizing Continuous Prompts for Generation 

论文链接：[https://arxiv.org/pdf/2101.00190.pdf](https://arxiv.org/pdf/2101.00190.pdf) 

代码链接：[https://github.com/XiangLi1999/PrefixTuning/issues](https://github.com/XiangLi1999/PrefixTuning/issues)

![image.png](https://cdn.nlark.com/yuque/0/2023/png/21475317/1683707079576-ea296dc7-b579-4af9-bbc1-034ea7a1b439.png#averageHue=%23f0f0ef&clientId=u6ae0a41b-d894-4&from=paste&height=661&id=ufa265dc5&originHeight=661&originWidth=1444&originalType=binary&ratio=1&rotation=0&showTitle=false&size=355742&status=done&style=none&taskId=u91dbca9a-e45c-4f04-a146-ca2d82bcc7a&title=&width=1444)

![image.png](https://cdn.nlark.com/yuque/0/2023/png/21475317/1683707093221-1c1f52a1-4d7e-456d-8942-61cf4c3801f5.png#averageHue=%235bac5b&clientId=u6ae0a41b-d894-4&from=paste&height=264&id=u8536f93d&originHeight=528&originWidth=679&originalType=binary&ratio=1&rotation=0&showTitle=false&size=77361&status=done&style=none&taskId=u2c7a4f51-9093-42c0-abde-b01515ab0b7&title=&width=340)

### Prompt构造方法：
-----------------

对于GPT2模型，不是直接传入embedding，而是传入embedding经过MLP后得到的key，value
并且，对于每一层，prompt token 的 embedding相同，但是key，value是不相同的
示例代码
初始化prompt token的embedding
```python
self.input_tokens = torch.arange(self.preseqlen).long()
self.wte = nn.Embedding(self.preseqlen, config.n_embd)
self.control_trans = nn.Sequential(
    nn.Linear(config.n_embd, self.mid_dim),
    nn.Tanh(),
    nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd))
self.get_prompt = self.get_prompt_p5

# 用一个大的MLP把每个 embedding 映射到2*n_layer个，其中n_layer个作为key，n_layer个作为value

```
获取key value
```python
  def get_prompt_p5(self, control_code=None, gpt2=None, bsz=None):
    input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
    temp_control = self.wte(input_tokens)
    past_key_values = self.control_trans(temp_control) #bsz, seqlen, layer*2*emb
    bsz, seqlen, _ = past_key_values.shape
    past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                           self.match_n_embd)
    past_key_values = self.dropout(past_key_values)
    past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
    return past_key_values
```
如何forward
```python
past_key_values_prompt = self.get_prompt(src, None, gpt2=gpt2_model, bsz=bsz)
attention_mask = torch.cat([src_attn, src_attn, tgt_attn], dim=1) # bsz, seqlen
# transformer类中，forward函数有个参数是past_key_value，直接把上面的结果传进这个参数即可
'''
以下是huggingface gpt2源码中的forward函数，支持past_key_value作为参数输入
'''
def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
```

### Prompt初始化方法：
-------------------------

随机初始化效果一般，结果方法很大。

作者选择用LM编码的真实单词的embedding来初始化模型，实验中也发现，用一些task相关词的embedding来初始化效果会更好。

