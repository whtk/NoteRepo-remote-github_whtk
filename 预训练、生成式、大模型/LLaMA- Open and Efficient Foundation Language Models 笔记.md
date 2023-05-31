> 2023 年
1. 提出 LLaMA，是一种基础语言模型，参数量从 7B 到 65B，并且可以使用公开可用的数据集进行训练
2. 13B 参数的模型在大多数的测试中性能优于 175B 的 GPT-3，和 65B 的模型和 Chinchilla-70B、PaLM-540B 可以相提并论

## Introduction

1. 现有的很多模型都认为，参数越多，性能越好
2. 但是有研究表明，性能最好的模型不是参数最多的那个，而是由更大的数据集下训练的较小模型
3. 作者发现，在 1T 的token 上训练 7B 的模型，性能都可以持续提升
4. 这项工作重点是训练语言模型 LLaMA，使用更多的 token 但是能够在合理的推理速度下实现最好的性能，尽管 LLaMA-13B 比 GPT-3 小了十倍，但是在很多测试中性能优于 GPT-3，并且可以在单个 GPU 上运行
5. 而对于 LLaMA-65B 的模型，其性能可以和  Chinchilla-70B、PaLM-540B 等大模型相提并论
6. 还有一个优点就是，训练的时候只使用公开可用的数据。

## 方法
总结起来就一句话：使用标准的优化器，在大量数据集中训练大型 Transformer 模型。

### 预训练数据
所用的数据集是各种领域的混合公开数据集：![[Pasted image 20230303132151.png]]
采用 BPE 算法对数据进行 tokenize。
总体数据包含 1.4T 个 token，且除了 Wikipedia 和 Books，每个 token 只使用一次。

### 架构
主要是基于 Transformer，但是也有一些不同：
1. pre-normalization：在 transformer 的每个 sub layer 之前进行 normalize 而不是之后，同时采用 RMS normalization 方法
2. SwiGLU 激活函数：把 ReLU 几激活函数替换成 SwiGLU 来提高性能
3. Rotary Embeddings：移除位置编码，而是使用 rotary positional embeddings
不同大小模型的超参数的对比：![[Pasted image 20230303134010.png]]

### 优化
使用 AdamW 优化器，采用余弦 lr scheduler，weight decay 为 0.1，梯度裁剪为 1，warmup 数为 2000。

### 高效实现
一些优化训练速度的方法：
+ 使用 causal multi-head attention 减少内存使用和运行时间，不存储注意力权重，且不计算 mask 部分的 score 
+ 保存计算过程中的激活值，通过手动实现 backprop 的过程而不是使用 pytorch 的自动求导 来实现的

训练 65B 的模型时，在 2048 台 80G RAM 的A100 GPU 上训练 21天，总计1.4T 个 token，平均每台 GPU 每秒 380 个 token。

## 结果
对标 GPT-3，在 zero shot 和 few shot 情况下测试了 20 个任务：
+ zero shot：提供 任务的文本描述
+ few shot：提供1-64个任务样例

### Common Sense Reasoning（常识推理） 任务
![[Pasted image 20230303140803.png]]

### closed book QA 任务
![[Pasted image 20230303141048.png]]

### 阅读理解
![[Pasted image 20230303141151.png]]
比 GPT-3 高出很多。

### 数学推理

### 代码生成

### 性能评估
![[Pasted image 20230303144849.png]]
大多数的实验中，性能都可以稳步提高。


### 大规模多任务语言理解（MMLU）
MMLU 包含各领域知识组成的多项选择题，包含人文、社科、科技、工程和教育等多个领域。在 five-shot 条件下进行模型评估，结果如下：![[Pasted image 20230303143613.png]]
LLaMA-65B 在大多数领域平均结果 落后于 Chinchilla-70B 和 PaLM-540B。

一种可能的解释是，训练的数据中使用的书籍和学术论文数量有限。

## 指令 fine tune

对指令的 简单 fine tune 可以快速导致 MMLU 的性能提升：![[Pasted image 20230303143951.png]]


