
1. 提出了一种新的损失，generalized end-to-end (GE2E) loss，使得 ASV 模型的训练比以前 tuple-based 的端到端损失函数（TE2E）更有效
2. GE2E loss function updates the network in a way that emphasizes examples that are difficult to verify at each step of the training process，且 GE2E loss does not require an initial stage of example selection
3. 引入了MultiReader技术，可以进行域自适应——支持多关键词及多方言

## Introduction

tuple-based E2E 损失为：一个 evaluation utterance $\mathbf{x}_{j \sim}$ 和 $M$ 个 enrollment utterances $\mathbf{x}_{k m},m=1, \ldots, m=1, \ldots, M$ 组成一个 tuple $\left\{\mathbf{x}_{j \sim},\left(\mathbf{x}_{k 1}, \ldots, \mathbf{x}_{k M}\right)\right\}$ 送到 LSTM 网络中，这里的 $j,k$ 代表所属的说话人，两个可能相同也可能不相同。

如果是相同的，称 tuple 为 positive的，反之为 negative，对于每个 tuple，计算 LSTM 输出再进行 L2归一化后的结果：$\left\{\mathbf{e}_{j \sim},\left(\mathbf{e}_{k 1}, \ldots, \mathbf{e}_{k M}\right)\right\}$，这里的 $\mathbf{e}$ 代表 embedding。tuple $\left(\mathbf{e}_{k 1}, \ldots, \mathbf{e}_{k M}\right)$ 的中心代表了说话人的声纹：$$\mathbf{c}_k=\mathbb{E}_m\left[\mathbf{e}_{k m}\right]=\frac{1}{M} \sum_{m=1}^M \mathbf{e}_{k m}$$
然后定义余弦相似度函数 $s=w \cdot \cos \left(\mathbf{e}_{j \sim}, \mathbf{c}_k\right)+b$，最后 TE2E 损失计算为：$$L_{\mathrm{T}}\left(\mathbf{e}_{j \sim}, \mathbf{c}_k\right)=\delta(j, k)(1-\sigma(s))+(1-\delta(j, k)) \sigma(s)$$
其中，$\sigma(x)=1 /\left(1+e^{-x}\right)$ 为 sigmoid 函数，当  $i=j$ 时，$\delta(j, k)=1$，其他情况都是 0。

考虑到 positive 和 negative tuple，这个损失很像 triplet loss。

本文则提出 GE2E 损失，以更有效的方式从不同长度的输入序列构建 tuple，从而提高ASV系统性能和训练速度。

## Generalize E2E 模型

整个系统架构：![](./image/Pasted%20image%2020221227093043.png)

### 训练方法

每个 batch 都有 $N \times M$ 个 utterance，来自 $N$ 个不同说话人，每个人 $M$ 个 utterance，特征向量 $\mathbf{x}_{j i}$ 表示说话人 $j$ 的第 $i$ 句话特征。

把 LSTM 网络表示为 $f\left(\mathbf{x}_{j i} ; \mathbf{w}\right)$ ，输出的特征（d-vector）定义为输出经过 L2 归一化的结果：$$\mathbf{e}_{j i}=\frac{f\left(\mathbf{x}_{j i} ; \mathbf{w}\right)}{\left\|f\left(\mathbf{x}_{j i} ; \mathbf{w}\right)\right\|_2}$$
由此定义相似度矩阵：$$\mathbf{S}_{j i, k}=w \cdot \cos \left(\mathbf{e}_{j i}, \mathbf{c}_k\right)+b$$
这里的 $w$ 约束为正数。

TE2E 和 GE2E 的区别在于：
+ TE2E 计算的是一个标量，即在 tuple 中 $\mathbf{e}_{j \sim}$ 和单个 $\mathbf{c}_k$ 的相似度
+ GE2E 计算相似度矩阵，定义的是 $\mathbf{e}_{j i}$ 和 所有的中心 $\mathbf{c}_k$ 之间的相似度

如上图不同颜色表示不同说话人。训练时，想要有颜色的值变大，灰色的值变小。有两种方法可以实现：
+ Softmax：每个 embedding $\mathbf{e}_{ji}$ 的损失定义为：$$L\left(\mathbf{e}_{j i}\right)=-\mathbf{S}_{j i, j}+\log \sum_{k=1}^N \exp \left(\mathbf{S}_{j i, k}\right)$$优化这个损失可以使得每个 embedding 接近其对应的中心远离其他中心。
+ Contrast：在正对和最近的几个负对之间定义对比损失为：$$L\left(\mathbf{e}_{j i}\right)=1-\sigma\left(\mathbf{S}_{j i, j}\right)+\max _{\substack{1 \leq k \leq N \\ k \neq j}} \sigma\left(\mathbf{S}_{j i, k}\right)$$
两个都有效，contrast loss 在 TD-SV 表现更好，softmax loss 在 TI-SV 表现更好。

并且发现，在计算中心时不考虑 $\mathbf{e}_{ji}$ 效果更好，所以实际计算相似度矩阵时自己那一派的有点不同：$$\begin{aligned}
\mathbf{c}_j^{(-i)} & =\frac{1}{M-1} \sum_{\substack{m=1 \\
m \neq i}}^M \mathbf{e}_{j m}, \\
\mathbf{S}_{j i, k} & = \begin{cases}w \cdot \cos \left(\mathbf{e}_{j i}, \mathbf{c}_j^{(-i)}\right)+b & \text { if } \quad k=j ; \\
w \cdot \cos \left(\mathbf{e}_{j i}, \mathbf{c}_k\right)+b & \text { otherwise. }\end{cases}
\end{aligned}$$
最后总的损失是所有相似度矩阵的求和：$$L(\mathbf{x} ; \mathbf{w})=L(\mathbf{S})=\sum_{j, i} L\left(\mathbf{e}_{j i}\right)$$
### 和 TE2E 对比（略）

### MultiReader 技术进行训练

在一个数据集 $D_1$ 训练可能导致过拟合，所以引入另一个数据集 $D_2$ 让模型也在这个数据集中表现良好，$D_2$ 相当于在训练过程中的 regularization（有点类似于 normal regularization），最终将损失定义为：$$L\left(D_1, D_2 ; \mathbf{w}\right)=\mathbb{E}_{x \in D_1}[L(\mathbf{x} ; \mathbf{w})]+\alpha \mathbb{E}_{x \in D_2}[L(\mathbf{x} ; \mathbf{w})]$$
这里第二项就相当于正则项。

## 实验（略）