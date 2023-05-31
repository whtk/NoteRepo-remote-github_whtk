> 论文 - Prefix-Tuning: Optimizing Continuous Prompts for Generation，斯坦福，2021，[IJCNLP](https://aclanthology.org/venues/ijcnlp/)

1. 提出 prefix-tuning，frozen 预训练模型的参数，优化一个连续的 task-specific 的向量（称为 prefix）
2. prefix-tuning 用于 GPT-2 和 BART ，只学习 0.1% 的数据效果优于 fine tune，且泛化性更好

## Introduction

1. fine tune 很贵
2. 一种方法是 lightweight fine tune，如 Adapter
3. 像 GPT-3 这种模型不需要 task-specific 的部署，而是需要 prompting（或者说是 in-context learning）
4. 由此启发，提出 prefix tuning：![](image/Pasted%20image%2020230506153115.png)
5. 上面是 Transformer fine tune 的情况，红色的部分表示全部 fine tune，下面是提出 prefix tuning，prefix 是红色部分，把 prefix （或者说是 “virtual tokens”）放在输入的 token 前面作为新的输入，在 fine tune 的时候，只会 fine tune prefix 的这些向量
6. 还有一个优点，就是可以在一个 batch 中同时测试多个不同的任务（相当于每个 instance 接入不同的 prefix）

## 相关工作（略）

## 问题描述

考虑一个条件生成任务，输入是上下文 $x$，输出 $y$ 是 token 序列，如图：![](image/Pasted%20image%2020230506153935.png)
两个任务：
+ table to text，$x$ 是数据表，$y$ 是文本描述
+ 摘要：$x$ 是人中，$y$  是摘要

### 自回归语言模型

模型 $p_\phi(y\mid x)$ ，参数为 $\phi$，$z=[x;y]$ 表示两者的拼接，$X_{idx},Y_{idx}$ 表示索引。time step $i$ 的输出记为 $h_{i}\in\mathbb{R}^{d}$，$h_i=\left[h_i^{(1)} ; \cdots ; h_i^{(n)}\right]$ 表示这个 time step 所有层的输出，上标表示 Transformer 的第 $j$ 层。

则自回归LM模型描述为：$$h_i=\mathbf{L M}_\phi\left(z_i, h_{<i}\right)$$
最后一层 Transformer 层的 $h_i$ 用于计算下一个 token 的概率 $p_\phi\left(z_{i+1} \mid h_{\leq i}\right)=\operatorname{softmax}\left(W_\phi h_i^{(n)}\right)$，$W_\phi$ 是预训练的权重矩阵。

## Fine Tune

提出 Prefix tuning 的直觉来自于，合适的上下文（context）可以在不改变 LM 参数的情况下引导 LM，context 可以引导模型从 $x$ 中提取的内容来影响 $x$ 的编码。在离散 token 下的优化可能有帮助，但是计算上有挑战。

于是提出，把指令优化为连续的 word embedding，向上可以传播到所有的 Transformer 层，向后可以传播到接入的 后续 token。

具体方法很简单，在自回归 LM 模型输入的头部追加一部分 $z=[\operatorname{PREFIX} ; x ; y]$，图中的 $P_{idx}$ 代表 prefix 的索引。

具体来说，通过初始化一个可训练的矩阵 $P_{\theta} \in \mathbb{R}^{|P_{idx}|\times \operatorname{dim}(h_i)}$  来存储参数：$$h_i= \begin{cases}P_\theta[i,:], & \text { if } i \in \mathrm{P}_{\mathrm{idx}} \\ \operatorname{LM}_\phi\left(z_i, h_{<i}\right), & \text { otherwise }\end{cases}$$
这个过程中，固定参数 $\phi$，只训练 $\theta$。
> 这里的每一层都有prefix，且每层的 prefix 都不共享，和 prompt tuning 只在输入有 soft prompt 不一样！

直接优化 $P_{\theta}$ 不稳定，且会导致性能下降（对学习率很敏感），所以通过一个小的矩阵 $P_\theta^{\prime}$ 来重参数化：$P_\theta[i,:]=\operatorname{MLP}_\theta\left(P_\theta^{\prime}[i,:]\right)$（两个的特征维度不一样，但是长度一样）。

## 实验和结果

table to text：![](image/Pasted%20image%2020230506161527.png)
在三个数据集上效果都好于其他的方法。

摘要：![](image/Pasted%20image%2020230506161803.png)
效果不如 fine tune，原因可能是：
+ XSUM包含的样本更多，文本也更长
+ 摘要任务更困难更复杂