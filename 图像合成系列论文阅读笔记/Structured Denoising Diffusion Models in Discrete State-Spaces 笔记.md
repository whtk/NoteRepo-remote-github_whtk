> Google，NIPS 2021

1. 提出 Discrete Denoising Diffusion Probabilistic Models（D3PMs），是对 multinomial diffusion 的 generalization，通过采用均匀分布来实现加噪过程
2. 包含以下三种传输矩阵进行加噪：
	1. 模仿连续空间中的高斯核的矩阵
	2. 基于 embedding 空间最邻近思想的矩阵
	3. 引入 absorbing states 的矩阵
3. 表明了，传输矩阵的选择非常重要
4. 同时引入了一个新的损失函数，将变分下界和一个额外的交叉熵损失进行组合
5. 在文本和图像中进行了实验

## Introduction

1. [Argmax Flows and Multinomial Diffusion- Learning Categorical Distributions 笔记](../经典模型和算法/Argmax%20Flows%20and%20Multinomial%20Diffusion-%20Learning%20Categorical%20Distributions%20笔记.md) 中已经提出使用带有离散状态空间的  diffusion 模型，但是没有实验用于大规模的文本或图像合成的模型
2. 本文是离散 diffusion 的拓展，采用更 structured categorical corruption process 来塑造数据分布，如下图，
![](image/Pasted%20image%2020231009211639.png)


3. 同时还引入了一个新的辅助损失来稳定训练和一系列的 noise schedules 来提高性能

## 背景：diffusion

在 diffusion 中，加噪过程的分布 $q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})$ 理论上可以上任意的，而当这个分布满足以下条件是，我们才可以有效地训练其近似 $p_\theta$：
+ 可以从分布 $q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})$ 中进行高效的采样得到 $\boldsymbol{x}_t$，从而可以独立的优化 $L_{t-1}$ 这一项
+ 分布 $q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t,\boldsymbol{x}_0)$ 是可求的，从而可以计算损失函数中的 KL 散度

最近的工作都是定义在连续空间的，即 $q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})~=~\mathcal{N}(\boldsymbol{x}_t|\sqrt{1-\beta_t}\boldsymbol{x}_{t-1},\beta_t\boldsymbol{I})$，从而 $p_\theta(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)=\mathcal{N}\left(\boldsymbol{x}_{t-1}|\boldsymbol{\mu}_\theta(\boldsymbol{x}_t,t),\boldsymbol{\Sigma}_\theta(\boldsymbol{x}_t,t)\right)$。

## 离散状态空间下的 diffusion 模型

其实 15 年最原始的那篇论文考虑了 binary random variables 下的 diffusion，后来 [Argmax Flows and Multinomial Diffusion- Learning Categorical Distributions 笔记](../经典模型和算法/Argmax%20Flows%20and%20Multinomial%20Diffusion-%20Learning%20Categorical%20Distributions%20笔记.md) 拓展到 categorial random variable，其传输矩阵为  uniform
transition probabilities。本文给出一个更通用的框架。

对于标量、离散随机变量，有 $K$ 类，即 $x_t,x_{t-1}\in1,\dots,K$，其 forward transition probabilities（前向转移概率）可以通过矩阵 $[\boldsymbol{Q}_{t}]_{ij}=q(x_{t}=j|x_{t-1}=i)$ 来表示，这里把 $x$ 拓展为 one-hot 的格式 $\boldsymbol{x}$，从而：
$$q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})=\mathrm{Cat}(\boldsymbol{x}_t;\boldsymbol{p}=\boldsymbol{x}_{t-1}\boldsymbol{Q}_t)$$
这里的 $\boldsymbol{p}$ 是一个长为 $K$ 的概率向量。
> 这里假设每个 token 之间是独立的。

从而整个过程中的边缘分布和后验分布为：
$$\begin{aligned}q(\boldsymbol{x}_t|\boldsymbol{x}_0)&=\mathrm{Cat}\left(\boldsymbol{x}_t;\boldsymbol{p}=\boldsymbol{x}_0\overline{\boldsymbol{Q}}_t\right),\quad\mathrm{with}\quad\overline{\boldsymbol{Q}}_t=\boldsymbol{Q}_1\boldsymbol{Q}_2\ldots\boldsymbol{Q}_t\\q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t,\boldsymbol{x}_0)&=\frac{q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1},\boldsymbol{x}_0)q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_0)}{q(\boldsymbol{x}_t|\boldsymbol{x}_0)}=\mathrm{Cat}\left(\boldsymbol{x}_{t-1};\boldsymbol{p}=\frac{\boldsymbol{x}_t\boldsymbol{Q}_t^\top\odot\boldsymbol{x}_0\overline{\boldsymbol{Q}}_{t-1}}{\boldsymbol{x}_0\overline{\boldsymbol{Q}}_t\boldsymbol{x}_t^\top}\right)\end{aligned}$$
那么假设 revere 过程也是每个独立的 token 之间的乘积，从而 KL 散度可以简单计算为，每个随机变量的某种和，也就是满足上面说的两个条件。

下面就主要看看 $\boldsymbol{Q}_t$ 的选择和最终分布 $q(\boldsymbol{x}_t|\boldsymbol{x}_0)$ （当 $T$ 趋于无穷，所谓的 stationary distributions） 。

### Markov transition matrices 的选择

D3PMs 的好处是，我们可以控制 $\boldsymbol{Q}_t$，其除了每一行加起来为 1 之外，唯一的约束就是，累乘之后的结果 $\overline{\boldsymbol{Q}}_{t}=\boldsymbol{Q}_{1}\boldsymbol{Q}_{2}\ldots\boldsymbol{Q}_{t}$ 必须收敛到一个一直的分布（当 $t$ 很大时）。

作者认为，大部分的离散数据，都可以在 $\boldsymbol{Q}_t$ 加上 domain-dependent structure 用于控制 forward corruption 过程，下面讨论三种矩阵。

Uniform：其实就是 Multinomial Diffusion 里面提出的：
$$\begin{aligned}\boldsymbol{Q}_t&=(1-\beta_t)\boldsymbol{I}+\beta_t/K\mathbb{1}\mathbb{1}^T,\quad \beta_t\in[0,1]\end{aligned}$$
最终的 stationary distribution 也是 uniform 的。

Absorbing state：考虑带有 absorbing state（称为 $[MASK]$）的传输矩阵，也就是每个 token 要么保持不变，要么以概率 $\beta_t$ 转移到 $[MASK]$ 这个 token。最终的 stationary distribution 不是 uniform 但是 has all the mass on the $[MASK]$ token。

Discretized Gaussian：模仿连续空间下的 diffusion，采用 离散的、截断的高斯分布。这样会使得相似状态之间的转移概率更高，非常适合图像这种 ordinal data。

Token embedding distance：文本数据没有 ordinal structure，但是也存在一些语义相关性。采用 embedding 空间中的相似度引导 forward 过程，构造  doubly-stochastic transition matrix，使得相似的 embedding 之间的转移概率更高，同时保证 stationary distribution 也是 uniform 的。

### Noise schedules

对于 discretized Gaussian diffusion，在离散化之前线性增加高斯分布的方差。

对于 uniform diffusion ，使用 cosine schedule，从而 cumulative probability  为 cosine 函数。

对于一个通用的 $\boldsymbol{Q}_t$，考虑从 $\boldsymbol{x}_t$ 到 $\boldsymbol{x}_0$ 到 $0$ 之间的互信息的插值，即 $I(\boldsymbol{x}_t;\boldsymbol{x}_0)\approx(1-\frac tT)H(\boldsymbol{x}_0)$。对于 absorbing-state D3PM，退化为 $(T-t+1)^{-1}$，此时正好是原始 DDPM 论文中提出的 Bernoulli diffusion process。

### 反向过程的参数化

虽然可以用神经网络直接预测反向过程 $p_\theta(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)$，但是还是采用原始 DDPM 中的想法，也就是用神经网络 $\operatorname{nn}_\theta(\boldsymbol{x}_t)$ 预测 $\widetilde{p}_\theta(\widetilde{\boldsymbol{x}}_0|\boldsymbol{x}_t)$，从而参数化为：
$$p_\theta(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)\propto\sum_{\widetilde{\boldsymbol{x}}_0}q(\boldsymbol{x}_{t-1},\boldsymbol{x}_t|\widetilde{\boldsymbol{x}}_0)\widetilde{p}_\theta(\widetilde{\boldsymbol{x}}_0|\boldsymbol{x}_t)$$

如果 $\widetilde{p}_\theta(\widetilde{\boldsymbol{x}}_0|\boldsymbol{x}_t)$ 只在 $\boldsymbol{x}_0$ 处有值（其他位置为 0）时，KL 散度 $D_\text{KL}[q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t,\boldsymbol{x}_0)||p_\theta(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)]$ 为 0 。

### 损失函数

原始的 diffusion 优化负的变分下界 $L_{\mathrm{vb}}$，这里还引入了一个额外的去噪目标函数用于 $\boldsymbol{x}_0$ 的参数化，总的损失函数为：
$$L_\lambda=L_\mathrm{vb}+\lambda\operatorname{E}_{q(\boldsymbol{x}_0)}\mathbb{E}_{q(\boldsymbol{x}_t|\boldsymbol{x}_0)}[-\log\widetilde{p}_\theta(\boldsymbol{x}_0|\boldsymbol{x}_t)]$$
> 这里的第二项是不是可以看成是 VAE 中的重构损失。

### 和现有的文本概率模型的联系

D3PMs 其实和现有的文本概率语言模型存在一些联系。

BERT 其实是一个 one-step 的 diffusion：此时对应的 transition matrix 是 uniform transition matrix 和 absorbing state（$[MASK]$）的组合（即 $\boldsymbol{Q}=\alpha\mathbb{1}e_m^T+\beta\mathbb{1}\mathbb{1}^T/K+(1-\alpha-\beta)I$，其中 $e_m$ 是只在 $[MASK]$ 位置为 1 的 one-hot token。然后进行一步的 diffusion 过程，也就是这一步 $q(\boldsymbol{x}_1|\boldsymbol{x}_0)$ 会以 $10\%$ 的概率替换成 $[MASK]$，以 $5\%$ 的概率均匀采样为其他的 token，这其实就是 BERT 的目标函数：
$$L_{vb}-L_{T}=-\mathbb{E}_{q(\boldsymbol{x}_{1}|\boldsymbol{x}_{0})}[\operatorname{log}p_{\theta}(\boldsymbol{x}_{0}|\boldsymbol{x}_{1})]=L_{BERT}$$

自回归模型也是离散 diffusion 模型：考虑这样一种情况，$[MASK]$ token 是一个一个添加的，此时的 diffusion 的 time step 就是序列的长度 $N=T$，且 forward 过程可以写为：
$$q([\boldsymbol{x}_t]_i\mid{\boldsymbol{x}_0})=[\boldsymbol{x}_0]_i\mathrm{~if~}i<N-t \mathrm{~else~} [MASK]$$
此时 $q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t,\boldsymbol{x}_0)$ 为 delta 函数 $q(\left[\boldsymbol{x}_{t-1}\right]_i\left|\boldsymbol{x}_t,\boldsymbol{x}_0\right)={\delta}_{[\boldsymbol{x}_t]_i}\mathrm{~if~}i\neq T-t\mathrm{~else~}\delta_{[\boldsymbol{x}_0]_i}$。

此时的 KL 散度退化为 $D_{KL}(q([\boldsymbol{x}_{t-1}]_i|\boldsymbol{x}_t,\boldsymbol{x}_0)~||~p_\theta([\boldsymbol{x}_{t-1}]_i|\boldsymbol{x}_t))=-\log p_\theta([\boldsymbol{x}_0]_i|\boldsymbol{x}_t)$，其实就是自回归模型标准的交叉熵损失函数。

(Generative) Masked Language-Models (MLMs) 也是 diffusion 模型：MLM 通常是采样 $\boldsymbol{x}_0$ 然后，然后 mask $k$ 个 token，然后学习预测被 mask 的 token。可以证明， D3PM absorbing model trained on the usual ELBO objective with the x0-parameterization from 3.3 reduces to a reweighted version of this MLM objective。

### 文本生成（略） 

### 图像生成（略）



