> 论文 - The Power of Scale for Parameter-Efficient Prompt Tuning，Google，2021 EMNLP

1. 提出 prompt tuning，基于 frozen 的语言模型来学习 soft prompt 以用于下游任务
2. soft prompt 是通过反向传播学习得到的，而不是人为输入的
3. 发现，模型越大，和 full fine tune 的差距越小
4. 可以看成是 prefix tuning 的一种简化
5. soft prompt 的鲁棒性很强

## Introduction

1. 研究表明，prompt design 非常有效，但是有一些缺点，如需要人为设计、受输入条件的限制等
2. 提出 prompt tuning，freeze 预训练模型，对每个任务，允许把 k 个额外的 token 放在输入文本之前，这些 soft prompt 是可以端到端训练的，可以缩小和 fine tune 之间的差距：![](image/Pasted%20image%2020230506174616.png)
3. 具体如图：![](image/Pasted%20image%2020230506174730.png)
4. 本文的主要贡献包括：
	1. 提出 prompt tuning，在 LM 中可以和 fine tune 相竞争
	2. 模型越大，性能和鲁棒性的提升越明显
	3. 在域迁移问题中，prompt tuning 效果好于 fine tune
	4. 提出 prompt ensembling，证明其有效性

## Prompt Tuning

将所有的任务都看成是  “Text to Text” 任务，T5 模型为 $\operatorname{Pr}_\theta(Y \mid X)$ ，那么 prompt 是指，在生成 $Y$ 的过程中，额外添加的信息（条件）。通过添加一系列的 tokens 序列 $P$，使得概率 $\operatorname{Pr}_\theta(Y \mid[P ; X])$ 尽可能地大，这个过程中并不会修改模型的参数。而且这个 prompt 通常是手动给的，并且是模型 embedding 的一部分，同时受参数 $\theta$ 的控制。

prompt tuning 移除这个限制，也就是说，prompt 有其自己的参数 $\theta_P$ ，且可以被更新，此时的模型生成变成 $\operatorname{Pr}_{\theta ; \theta_P}(Y \mid[P ; X])$ 能够通过最大 $Y$ 的似然进行反向传播，并且只在 $\theta_P$ 上计算梯度。

给定 $n$ 个 tokens $\{x_1,x_2,\cdots,x_n\}$ ，T5 模型首先获得这些 tokens 的 embedding，从而形成一个矩阵 $X_e\in\mathbb{R}^{n\times e}$ ，$e$ 为 embedding 的维度。提出的 soft prompt 则表示为 $P_e\in\mathbb{R}^{p\times e}$，$p$ 为 prompt 的长度，然后模型的输入为 $[P_e;X_e]\in\mathbb{R}^{(p+n)\times e}$，这个过程中只更新 $P_e$ 的参数。

需要考虑两个点：
+ 参数的初始化：可以随机，也可以从 一些特定任务的 embedding 中选
+ prompt 的长度

主要在 T5 模型中测试的。

> 其实 prompt tuning 和 prefix tuning 的思路几乎一样，区别在于，prompt tuning 不会在 Transformer 的每一层都有一个 prefix，而只是在输入层有一个 prompt，而 prefix tuning 对于每一层的 Transformer layer 都不一样。