# The Power of Scale for Parameter-Efficient Prompt Tuning  

论文地址： [https://arxiv.org/pdf/2104.08691.pdf](https://arxiv.org/pdf/2104.08691.pdf)

代码地址1：[https://arxiv.org/pdf/2104.08691.pdf](https://arxiv.org/pdf/2104.08691.pdf) （官方）

代码地址1：[https://github.com/mkshing/Prompt-Tuning](https://github.com/mkshing/Prompt-Tuning)（非官方）

![image.png](https://cdn.nlark.com/yuque/0/2023/png/21475317/1683705094677-472ab7d9-4de5-45ad-9c72-e7ad17e4f34d.png#averageHue=%23f3ece5&clientId=u9710d0b1-f778-4&from=paste&height=251&id=ube89a577&originHeight=251&originWidth=508&originalType=binary&ratio=1&rotation=0&showTitle=false&size=46167&status=done&style=none&taskId=u5872f407-a8f8-4479-adfd-7afa676bca5&title=&width=508)

### Prompt构造方法：
---------------------------

示例代码
```python
 def _cat_learned_embedding_to_input(self, input_ids) -> torch.Tensor:
    inputs_embeds = self.transformer.wte(input_ids)

    if len(list(inputs_embeds.shape)) == 2:
        inputs_embeds = inputs_embeds.unsqueeze(0)

    # [batch_size, n_tokens, n_embd]
    learned_embeds = self.soft_prompt.weight.repeat(inputs_embeds.size(0), 1, 1)

    inputs_embeds = torch.cat([learned_embeds, inputs_embeds], dim=1)

    return inputs_embeds
```

把prompt直接当作可训练的embedding

先将ids转化成embedding，将prompt的embedding直接cat在input之前即可

### Prompt初始化方法：

--------------------------------------------------------
 
- random uniform： 简单的随机初始化

- sampled vocab：选择语料库中最常见的5k token 从中采样，选择采样出的token在预训练模型中的embedding

- class label：下游任务的每个类别的标签的embedding作为每个prompt token的embedding


![image.png](https://cdn.nlark.com/yuque/0/2023/png/21475317/1683706352076-ed7fbc32-6a28-4d4c-9142-b5439438429d.png#averageHue=%23edeceb&clientId=u9710d0b1-f778-4&from=paste&height=278&id=u1f7a7eeb&originHeight=278&originWidth=314&originalType=binary&ratio=1&rotation=0&showTitle=false&size=31403&status=done&style=none&taskId=ud40a966f-7304-4537-a3f8-9722b7f8457&title=&width=314)

