> NIPS 2020，Seoul National University

1. Normalizing ﬂows（NF）在参数复杂度上很低效
2. 提出 NanoFlow，采用单个 neural density estimator 来建模多个 stage 下的 变换
3. 提出了 一种有效的减参方法 和 ﬂow indication embedding 的概念

## Introduction

1. flow 模型存在一个 question：到底是 NF 需要很大的网络来实现 expressive bijection，还是神经网络的表征能力被低效利用？
	1. 作者认为应该考虑模型参数复杂度，模型的 expressiveness 不应该随着参数线性增长
2. 提出 NanoFlow，在多个 flow 中采用 shared density estimator

## 背景

NF 学习 数据 和 已知先验（通常为高斯分布） 之间的 bijective mapping。具体来说，学习的是映射 $f(\boldsymbol{x})=\boldsymbol{z}$，将复杂数据分布 $P_X$ 转为已知先验分布 $P_Z$。从而可以采用 change of variables 公式来计算概率密度分布：
$$\log P_X(\boldsymbol{x})=\log P_Z(\boldsymbol{z})+\log|\det(\frac{\partial f(\boldsymbol{x})}{\partial\boldsymbol{x}})|$$
其中 $\det(\frac{\partial f(\boldsymbol{x})}{\partial\boldsymbol{x}})$ 为雅各比行列式。为了增强表达力，通常会分为多步的 flow：
$$f=f^K\circ f^{K-1}\circ...\circ f^1(\boldsymbol{x}),$$
其中 $K$ 为数量。记 $\boldsymbol{x}=\boldsymbol{z}^0,\boldsymbol{z}=\boldsymbol{z}^K$，则 $f^k(\boldsymbol{z}^{k-1})=\boldsymbol{z}^k$。此时 $\log P_X(\boldsymbol{x})$ 可以表示为：
$$\log P_X(\boldsymbol{x})=\log P_Z(\boldsymbol{z})+\sum_{k=1}^K\log|\det(\frac{\partial f^k(\boldsymbol{z}^{k-1})}{\partial\boldsymbol{z}^{k-1}})|$$
由于计算行列式需要 $O(n^3)$ 复杂度，NF 模型设计时通常确保雅各布矩阵是三角的来简化计算。

给定训练数据 $\boldsymbol{x}$，假设将 $\boldsymbol{x}$ 分为 $G$ 组，记为 $\{X_1,...,X_G\}$，则模型训练来学习相同维度下 $X$ 和 $Z$ 之间的 bijective mapping。此过程通过所谓的 afﬁne transformation $f:\boldsymbol{X}\rightarrow\boldsymbol{Z}$ 来实现，即将分组后的数据建模如下：
$$Z_i=\sigma_i(\boldsymbol{X}_{<i};\theta)\cdot\boldsymbol{X}_i+\mu_i(\boldsymbol{X}_{<i};\theta),\quad i=1,...,G$$
其中 $\boldsymbol{X}_{<i}$ 表示第 $i$ 组之间的所有数据。而对于采样后的噪声 $Z$，反向生成 $X$ 如下：
$$X_i=\frac{Z_i-\mu_i(X_{<i};\theta)}{\sigma_i(X_{<i};\theta)},\quad i=1,2,...,G$$
当 $G=dim(x)$ 时上面就是一个纯自回归的 flow。当 $G=2$ 时，就是 bipartite ﬂows。考虑 $K$ 个 step，此时每次转换写为 $f^k:Z^{k-1}\rightarrow Z^k$，其中 $X=Z^0,Z=Z^K$。第 $k$ 个 flow 对应的参数为 $\theta^k$，此时可以将 $f^k$ 重写为：
$$\boldsymbol{Z}_i=\sigma_i(\boldsymbol{X}_{<i};\theta^k)\cdot\boldsymbol{X}_i+\mu_i(\boldsymbol{X}_{<i};\theta^k)$$

## NanoFlow

NanofLow 的目的是，将 expressiveness of the bijections
和 parameter efﬁciency of density estimation from neural networks 进行解耦。

![](image/Pasted%20image%2020240108213710.png)

### 参数共享和分解

将 $f_{\mu,\sigma}^{k}$ 用单个神经网络 $f_{\mu,\sigma}$ 来表示，其参数为 $\theta$，此时所有的 $\mu^k,\sigma^k$ 的估计过程都可以通过共享的 $f_{\mu,\sigma}$ 实现。从而将 flow 的参数减少为 $\frac 1 K$，将这个模型称为 NanoFlow-naive，但是实验表明这种简单的参数复用并不适合建模复杂数据分布，会导致性能的下降。

于是提出，将这部分的模型分解为两部分：
+ 用于计算 hidden representation 的网络
+ 用于估计密度的 projection layer

从而原始公式上也可以分为两部分 $f_{\mu,\sigma}^{k}\circ g$，其中 $g(\cdot;\hat{\theta})$ 的参数为 $\hat{\theta}$，相当于共享表征提取网络参数，不共享 projection layer，从而 $f_{\mu,\sigma}^{k}$ 有独立的参数。

假设 $g$ 有足够的密度估计能力，则 projection layer 可以是浅层的 1x1 卷积。

从而可以构造任意长度的 flow，此时 $f^k$ 可以写为：
$$\boldsymbol{Z}_i=\sigma_i(\boldsymbol{X}_{<i};\hat{\theta},\epsilon^k)\cdot\boldsymbol{X}_i+\mu_i(\boldsymbol{X}_{<i};\hat{\theta},\epsilon^k)$$
其中 $\epsilon^k$ 为每个 flow 的独立的 projection layer 参数。

### Flow indication embedding

即使实现了上述的参数分解，$g$ 也需要在没有 context 的条件下学习每个变换的中间表征，于是引入 ﬂow indication embedding，使得共享模型可以同时学习到多个 bijective transformations。

对于每个 $f^k$，定义 embedding vector $e^k\in\mathbb{R}^D$，然后将 embedding 送入到 $g(\cdot;\hat{\theta})$ 中作为一个额外的 context，来引导其学习不同的中间密度。
> 在推理的时候，以逆序送入到 flow 模型中。

至于这么引入，有以下几种：
+ 拼接到输入
+ additive bias，$e^k$ 投影为 channel-wise 的 bias
+ multiplicative gating，计算一个 scale 来控制 feature 的传入

具体哪个方法更好和网络结构有关。

## 实验（略）

