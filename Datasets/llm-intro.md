# Large Language Modll Introdution

## LLAMA
- 论文地址： https://arxiv.org/pdf/2302.13971.pdf
- 基座LLM，仅预训练，没有alignment

## Alpaca

- 网页地址：https://crfm.stanford.edu/2023/03/13/alpaca.html
- 在LLAMA-7B的基础上，用52K instruction-following demonstrations上全参数finetune的
- instruction-following 对话模型
- Evaluation：没有evaluation


## Vicuna 
- 网页地址：https://lmsys.org/blog/2023-03-30-vicuna/
- 依照Alpaca的训练过程，在LLAMA7B上用shareGPT的70k对话数据，增强alpaca的训练脚本，全参数finetune。
- instruction-following 对话模型
- Evaluation: 创建80个不同问题，用GPT-4来判断模型输出，模型之间两两对比判断谁的回答更好。
  
  