> NAACL 2022

1. Adapter 还是需要相对大量的参数
2. 提出 AdapterBias，在 Transformer 层的 hidden output 中添加 token 无关的 shift，也就是只加了一个 vector 和一个 linear 层
3. 进行了大量的实验来表明其有效性

## Introduction

1. 在低资源的数据上 fine tune 预训练的模型 会不稳定
2. 于是提出 [[../经典模型和算法/PEFT/Adapter]]，但是不知道 adapter 是不是能够进一步实现 parameter-efficient
3. Diff pruning 通过学习 task specific 的 "diff" 向量，拓展原始的预训练参数，同时通过 L0 正则化提高稀疏性
4. BitFit 表明，对于中小型数据，只在 一些 bias term 上进行 fine tune 可以和 fine tune 整个模型相竞争
5. 这些工作的核心都是在PLM 输出层添加  task-specific shifts 用于适应不同的任务
6. 基于此理念，本文提出了由一个 vector 和一个线性层 $L_\alpha$ 组成的 AdapterBias，在 shifts 中添加 token-dependent biases：
	1. vector 表示特定任务的 shift
	2. $L_\alpha$ 为输入 token 产生权重
	3. 通过 vector 和 weight 来添加 token-dependent shift
	4. 和 BitFit 有点类似，比较如下：![](./image/Pasted%20image%2020230416105422.png)
	5. 最终可以以更少的参数获得和 Adapter 相当的性能，而且通过一些正则化措施可以进一步降低参数量

## 相关工作（略）

## 方法

为了在不同的任务中进行更好的自适应，adapter 需要 token-specific。AdapterBias 则基于输入的 token 为 bias 产生合适的权重。

在 fine tune 预训练模型是，记训练数据为 $D=\left(x_i, y_i\right)_{n=1}^N$，假设模型的参数为 $\theta$，AdapterBias 的参数为 $\theta^\prime$，训练的时候，冻结 $\theta$ ，只 fine tune $\theta^\prime$。

架构如图：![](./image/Pasted%20image%2020230416110023.png)
包含两个部分：
+ vector $v$
+ linear layer $L_\alpha$

$v$ 是 task-specific shift，$L_\alpha$ 输出一个 token-dependent 加权权重向量 $\alpha=\left[\alpha_1, \alpha_2 \ldots \alpha_m\right]^T$，其中 $\alpha_i$ 为第 $i$ 个 token’s representation shift 的权重。通过加权计算，AdapterBias 能够专注于对任务更重要的权重，并且可以有效地适用于不同的下游任务。

> 这里 第二个 sub layer 的 layer norm 层也做了 fine tune。

定义 bias 为 $B$：$$B=v \otimes \alpha^T=\left(\begin{array}{llll}
\alpha_1 v & \alpha_2 v & \ldots & \alpha_m v
\end{array}\right)$$
其中，$v \in \mathbb{R}^r, \alpha \in \mathbb{R}^{m},B \in \mathbb{R}^{r \times m}$，表示 $m$ 个 token，每个 token 的表征为 $r$。具体来说，计算过程如下：![](./image/Pasted%20image%2020230416111114.png)
假设序列的长度为 $m=3$，第一层 layer norm 输出的表征为 $(r_1,r_2,r_3)$，其维度就是 Transformer 中的 $d_{model}$（BERT 中就是 768）。通过 FFN 之后的 token 作为 $L_\alpha$ 的输入，而其输出 $\alpha \in \mathbb{R}^3$，然后乘起来得到 $B$，比如其中的 $b_1$ 就可以看成是 第一个 token 的 bias。

### 进一步提高 AdapterBias 的参数效率

方法1：跨层共享
根据 Adapter 中的实验，低层的 adapter其实是存在一些冗余的，于是可以在不同的层中共享 adapter 的权重来减少参数。

方法2：$L_0$ 正则化
对 $L_\alpha$ 添加 dropout，进一步提高参数效率。此时的优化问题可以看成：$$\min _{\theta^{\prime}} L\left(D ; \theta, \theta^{\prime}\right)+\lambda\left\|\theta_{L_\alpha}^{\prime}\right\|_0$$
其中，$L(D;\cdot)$ 表示原来的损失，$\lambda$ 为超参数。

## 实验

在 HuggingFace PyTorch 中的 BERT 和 RoBERTa 上进行实验。

在 GLUE 上的结果：![](./image/Pasted%20image%2020230416133039.png)
可以用最少的参数达到和其他模型相当的性能。

不同模型的泛化性对比：![](./image/Pasted%20image%2020230416133534.png)
在多个模型上效果都不错。

消融实验：![](./image/Pasted%20image%2020230416134415.png)
共享线形层 $L_\alpha$ 的参数效果最好。
对于正则化，可以在 base 模型中提高性能，但是在 large 模型中不能。