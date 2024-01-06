> 论文 - LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS，微软，2022

1. fine tune 大模型非常困难，提出 LoRA，freeze 预训练模型的权重，在每个 Transformer 结构加入 可训练的 rank decomposition matrices，减少下游任务训练所需的参数量
2. 拿 GPT-175B 模型做对比，参数可以减少 10000 倍，并且相比于 adapter 没有额外的推理延迟

## Introduction

1. 背景：fine tune 预训练模型非常困难
2. 有些研究通过 fine tune 部分参数或者学习新的模块来缓解，但是现有的方法都会引入 推理延迟（inference latency），而且效果可能不如 fine tune 的 baseline，导致存在效率和性能的权衡
3. 模型在学习过程中，权重的调整是“低秩”的，于是提出 Low-Rank Adaptation（LoRA）方法，通过优化自适应过程中 dense layer 变化的秩分解矩阵来间接训练神经网络中的一些 dense layer，同时保持预训练模型的权重冻结：![](image/Pasted%20image%2020230426221503.png)以GPT-3 为例，即使图中的 $d=12288$，$r$ 等于 1 或者 2 都足够了。
4. LoRA 的优点有：
	1. 预训练模型可以共享，可以为不同的任务构建不同的 LoRA 模块
	2. 实现高效训练
	3. 采用线性设计，在部署的时候可以合并训练的矩阵和原始的冻结权重，不引入推理延迟
	4. 能够和其他的方法组合

记 Transformer 层的输入输出维度为 $d_{model}$，用 $W_q,W_k,W_v,W_o$ 分别表示注意力机制中的 query、key、value 和 输出投影矩阵，用 $W$ 或者 $W_0$ 表示预训练的权重矩阵，$\Delta W$ 表示累积的梯度，$r$ 表示 LoRA 模块的秩，采用 Adam 优化器，$d_{ffn}=4\times d_{model}$。

## 语言模型存在问题

设有一个自回归的预训练语言模型 $P_\Phi(y\mid x)$，参数为 $\Phi$，如 GPT 模型。

如果要把这个模型自适应到下游任务中，每个任务所用的训练集为 $\mathcal{Z}=\left\{\left(x_i, y_i\right)\right\}_{i=1, \ldots, N}$，其中 $x_i,y_i$ 都是 tokens 序列， 在 full fine tune 的情况下，模型初始化为预训练的权重 $\Phi_0$，更新为 $\Phi_0+\Delta \phi$，通过最大化以下目标函数：$$\max _{\Phi} \sum_{(x, y) \in \mathcal{Z}} \sum_{t=1}^{|y|} \log \left(P_{\Phi}\left(y_t \mid x, y_{<t}\right)\right)
$$

这种 full fine tune 的缺点是，对于每个下游任务，都要重新学习一个不同的参数集，其参数 $\Delta\Phi$ 的维度和 $\Phi_0$ 相同，如果预训练模型很大，那这个参数就会很大。

本文采用了一个参数高效的方法，将参数增量 $\Delta\Phi$ 通过一个更小的参数集进行编码 $\Delta\Phi=\Delta\Phi(\Theta)$，且 $\Theta$ 的维度远小于 $\Phi_0$。此时优化问题变成：$$\max _{\Theta} \sum_{(x, y) \in \mathcal{Z}} \sum_{t=1}^{|y|} \log \left(p_{\Phi_0+\Delta \Phi(\Theta)}\left(y_t \mid x, y_{<t}\right)\right)$$
当模型采用 GPT-3 175B 时，参数量能够变为原来的 $0.01\%$。

## 方法

现有的方法缺点：
+ Adapter 会引入延迟
+ 直接优化 prompt 很难做

### 低秩参数更新

研究表明，预训练的语言模型具有较低的 "instrisic dimension"，于是假设更新的权重也具有较低的秩，对于预训练的权重矩阵 $W\in\mathbb{R}^{d\times k}$ ，当更新参数的时候，通过添加一个低秩的分解来替代 $\Delta W$：$W_0+\Delta W=W_0+B A$，其中 $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}, r \ll \min (d, k)$， 训练的时候，$W_0$ 保持冻结，$A,B$  包含可训练的参数，在前向传播的时候，两个分量都和输入相乘：$$h=W_0 x+\Delta W x=W_0 x+B A x$$
$A$ 采用随机高斯初始化，$B$ 初始化为 0。

**在部署的时候，直接用 $W=W_0+BA$ 替换原来的权重，此时计算的时候维度不变因此不会引入推理延迟。**

**而且通过增加秩 $r$ ，模型性能会收敛到 full fine tune 的性能。**

