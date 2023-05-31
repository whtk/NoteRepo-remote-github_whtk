> 论文 - GPT Understands, Too，2021，清华

1. 传统 fine tune + GPT 在**自然语言理解NLU**方面效果不太行（prefix tuning 主要针对于 NLG 任务）
2. 通过提出的 P-tuning 方法，可以使 GPT 的性能优于或类似于 BERT
3. 在 few shot 的情况下，P-tuning 优于一些 sota 方法

## Introduction

1. 根据目标，预训练模型可以分为三类：
	1. 用于自然语言生成，单向模型，如 GPT
	2. 用于自然语言理解，双向模型，如 BERT
	3. 混合模型，如 UniLM
2. 研究发现，GPT 范式在 fine tune 自然语言理解任务中效果不好
3. 而 GPT-3 的出现表明，单向模型+prompt 可能有助于自然语言理解，但是手动选取 prompt 非常困难，有一些工作关注自动搜索离散 prompt，但是神经网络的本质是连续的，离散 prompt 可能不是最优的
4. **提出 P-tuning，在连续空间中自动搜索参数化的 prompt，然后通过梯度下降法来优化连续的 prompt**
5. 把 P-tuning 用于 GPT 中，在两个 NLU 任务上实验发现，与具有相同规模的BERT模型相比，GPT表现出有竞争力的性能甚至优于 BERT。
6. ALBERT+P-tuning 可以在 SuperGLUE benchmark 上获得最优性能

## Motivation

大模型非常强，但是迁移能力较差，而对下游任务进行 fine tune 又不现实。

于是这些模型采用手工制作的 prompt 来引导下游任务，但是这种 prompt 的性能不可靠，可能一个单词的变化会导致巨大的性能差异：![](image/Pasted%20image%2020230507155443.png)

## P-tuning

给定预训练模型 $\mathcal{M}$ ，离散的 tokens 序列 $\mathbf{x}_{1: n}=\left\{x_0, x_1, \ldots, x_n\right\}$ 被 embedding 层映射为 embedding $\left\{\mathbf{e}\left(x_0\right), \mathbf{e}\left(x_1\right), \ldots, \mathbf{e}\left(x_n\right)\right\}$。在特定情况中，基于上下文 $\mathbf{x}$ ，使用一组目标 tokens $\mathbf{y}$ 的输出 embedding 来进行下游任务的处理。

prompt $\mathbf{p}$ 把上下文 $\mathbf{x}$，目标 $\mathbf{y}$ 和自身合成一个模板 $T$，如，在预测一个国家的首都的任务中，模板可能是 The capital of Britain is “MASK”，其中 The capital of ... is ... . 是prompt，Britain 是 $\mathbf{x}$ ，“mask” 是目标 $\mathbf{y}$ ：![](image/Pasted%20image%2020230507160413.png)

设 $\mathcal{V}$ 代表语言模型的词表，$[P_i]$ 表示模板中的第 $i$ 个prompt。给定模板 $T=\left\{\left[\mathrm{P}_{0: i}\right], \mathbf{x},\left[\mathrm{P}_{i+1: m}\right], \mathbf{y}\right\}$，相比于传统的离散 prompt 要满足 $[P_i]\in\mathcal{V}$ 然后把模板映射为 $\left\{\mathbf{e}\left(\left[\mathrm{P}_{0: i}\right]\right), \mathbf{e}(\mathbf{x}), \mathbf{e}\left(\left[\mathrm{P}_{i+1: m}\right]\right), \mathbf{e}(\mathbf{y})\right\}$，P-tuning 将 $[P_i]$ 视为一个伪 token，此时模板被映射为：$$\left\{h_0, \ldots, h_i, \mathbf{e}(\mathbf{x}), h_{i+1}, \ldots, h_m, \mathbf{e}(\mathbf{y})\right\}$$
其中，$h_i(0\le i < m)$ 是可以训练的 tensor，从而可以通过下游任务的目标函数进行训练，找出最优的连续的 prompt：$$\hat{h}_{0: m}=\underset{h}{\arg \min } \mathcal{L}(\mathcal{M}(\mathbf{x}, \mathbf{y}))$$
想法很直接，但是有两个挑战：
+ 如果 $h$ 随机初始化，很容易陷入局部最小
+ $h_i$ 之间应该是相关而非独立的

于是通过一个小型的神经网络作为 encoder 来建模 $h_i$（也就是不直接学习$h_i$），即可解决这两个问题，实际中，这个网络选择的是 LSTM+ReLU+两层MLP，即：$$\begin{aligned}
h_i & =\operatorname{MLP}\left(\left[\overrightarrow{h_i}: \overleftarrow{h_i}\right]\right) \\
& =\operatorname{MLP}\left(\left[\operatorname{LSTM}\left(h_{0: i}\right): \operatorname{LSTM}\left(h_{i: m}\right)\right]\right)
\end{aligned}$$
在推理的时候，可以丢弃 LSTM，只要最终的 $h$。

## 结果
![](image/Pasted%20image%2020230507161832.png)

