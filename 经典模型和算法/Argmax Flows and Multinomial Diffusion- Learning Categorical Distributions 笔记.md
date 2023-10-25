> NIPS 2021，UvA-Bosch Delta Lab, University of Amsterdam

1. 提出两种 flows 和 diffusion 的拓展，Argmax Flows 和 Multinomial Diffusion，可用于 categorical data 如语言或图像分割
2. Argmax Flows 由一些列的连续分布（如 normalizing flow）和一个 argmax 函数组成
3. Multinomial Diffusion 在 diffusion 过程中逐步引入 categorical noise
4. 提出的方法超过了现有的 dequantization 方法

> Multinomial：多项式、多项

## Introduction

1. 许多高维数据都是 categorical 的
2. normalizing flows 通常用于建模连续分布
3. diffusion 主要用于学习 ordinal data（定序数据），如图像
4. 本文为 categorical variables 引入两种方法的拓展：
	1. Argmax Flows
	2.  Multinomial Diffusion

## 背景

### Normalizing Flows

给定数据 $\mathcal{V}=\mathbb{R}^d,\mathcal{Z}=\mathbb{R}^d$，其 PDF 分别为 $p_V,p_Z$，normalizing flows 学习一个双射且可微的变换 $g:\mathcal{Z}\to\mathcal{V}$，使得在任意点 $v\in\mathcal{V}$ 都可以通过  change-of-variables formula 得到，即：
$$p_V(\boldsymbol{v})=p_Z(\boldsymbol{z})\cdot\left|\det\frac{\mathrm{d}\boldsymbol{z}}{\mathrm{d}\boldsymbol{v}}\right|,\quad\boldsymbol{v}=g(\boldsymbol{z})$$
这里的 $p_Z$ 可以是任何分布（通常是标准高斯分布），从而，normalizing flows 可以准确地学习分布密度，但是上式只限于连续的密度。

为了学习 ordinal discrete data 的密度，通常会加入 dequantization noise，然后将其看成是从 $v\to x$ 的映射，从一个方向是 deterministic 的（$x=\operatorname{round}(v)$），另一个方向看是 stochastic 的（$v=x+u,u\sim q(u|x)$），从而 dequantization  可以看成是  probabilistic right-inverse for the rounding operation：
$$P(\boldsymbol{x})=\int P(\boldsymbol{x}|\boldsymbol{v})p(\boldsymbol{v})\operatorname{d\boldsymbol{v}},\quad P(\boldsymbol{x}|\boldsymbol{v})=\delta(\boldsymbol{x}=\operatorname{round}(\boldsymbol{v})),$$
> 这里的 $v$ 的连续的，$x$ 是离散的。

此时，使用 normalizing flows 来建模 $p(\boldsymbol{v})$ ，损失为：
$$\log P(\boldsymbol{x})\geq\mathbb{E}_{\boldsymbol{v}\sim q(\boldsymbol{v}|\boldsymbol{x})}\left[\log P(\boldsymbol{x}|\boldsymbol{v})+\log p(\boldsymbol{v})-\log q(\boldsymbol{v}|\boldsymbol{x})\right]=\mathbb{E}_{\boldsymbol{v}\sim q(\boldsymbol{v}|\boldsymbol{x})}\left[\log p(\boldsymbol{v})-\log q(\boldsymbol{v}|\boldsymbol{x})\right]$$
最后一项，$$\begin{aligned}
&\text{The last equality holds under the constraint that the support of }q(\boldsymbol{v}|\boldsymbol{x})\text{ is enforced to be only over the} \\
&\operatorname{region}\mathcal{S}=\{\boldsymbol{v}\in\mathbb{R}^d:\boldsymbol{x}=\operatorname{round}(\boldsymbol{v})\}\text{ which ensures that }P(\boldsymbol{x}|\boldsymbol{v})=1.
\end{aligned}$$

### diffusion

略

## Argmax Flows（略）

## Multinomial Diffusion

![](image/Pasted%20image%2020231008111747.png)

用于  categorical data。和之前的不一样，这里的 $\boldsymbol{x}_t$ 是 one-hot 格式的，即 $\boldsymbol{x}_t\in\{0,1\}^K$，对于类别 $k$，$x_k=1$，其他都为 $0$。
> 忽略了维度轴以简化，例如维度可能是 $h\times w\times K$，忽略了 feature map 维度 $h\times w$。

采用 categorical distribution  来定义 multinomial diffusion 的过程，其有 $\beta_t$ 的概率均匀采样一个类别：
$$q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})=\mathcal{C}(\boldsymbol{x}_t|(1-\beta_t)\boldsymbol{x}_{t-1}+\beta_t/K)$$
这里的 $\mathcal{C}$ 就代表  categorical distribution。
> 简单来说，有 $\beta_t$ 的概率从 $K$ 个类别中随机均匀采样，有 $1-\beta_t$ 的概率保持不变。注意这里并不是转移概率。

又因为满足马尔可夫链，从而：
$$q(\boldsymbol{x}_t|\boldsymbol{x}_0)=\mathcal{C}(\boldsymbol{x}_t|\bar{\alpha}_t\boldsymbol{x}_0+(1-\bar{\alpha}_t)/K)$$
其中，$\begin{aligned}\alpha_t=1-\beta_t\text{ and }\bar{\alpha}_t=\prod_{\tau=1}^t\alpha_\tau\end{aligned}$。

根据连续的 diffusion 的理论，我们通常计算后验 $q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t,\boldsymbol{x}_0)$，这里 categorical posterior 可以写为：
$$\begin{gathered}
q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t,\boldsymbol{x}_0) =\mathcal{C}(x_{t-1}|\boldsymbol{\theta}_{\mathrm{post}}(\boldsymbol{x}_t,\boldsymbol{x}_0)),\quad\mathrm{where}\quad\boldsymbol{\theta}_{\mathrm{post}}(\boldsymbol{x}_t,\boldsymbol{x}_0)=\tilde{\boldsymbol{\theta}}/\sum_{k=1}^K\tilde{\theta}_k \\
\tilde{\theta} =[\alpha_t\boldsymbol{x}_t+(1-\alpha_t)/K]\odot[\bar{\alpha}_{t-1}\boldsymbol{x}_0+(1-\bar{\alpha}_{t-1})/K]. 
\end{gathered}$$
> 

在原始的 DDPM 中，通常预测噪声。但是对于离散数据，预测噪声很困难，所以还是从 $\boldsymbol{x}_t$ 中为 $\hat{\boldsymbol{x}}_0$ 预测一个概率向量：$\hat{\boldsymbol{x}}_0=\mu{(\boldsymbol{x}_t,t)}$，总的来说：
$$p(\boldsymbol{x}_0|\boldsymbol{x}_1)=\mathcal{C}(\boldsymbol{x}_0|\hat{\boldsymbol{x}}_0)\mathrm{~and~}p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)=\mathcal{C}(\boldsymbol{x}_{t-1}|\boldsymbol{\theta}_{\mathrm{post}}(\boldsymbol{x}_t,\hat{\boldsymbol{x}}_0))\mathrm{~where~}\hat{\boldsymbol{x}}_0=\mu(\boldsymbol{x}_t,t)$$
> 简单来说，就是用神经网络预测不带噪声的样本 $\hat{\boldsymbol{x}}_0$，然后就和普通的 diffusion 一样，带入到 $q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t,\boldsymbol{x}_0)$ 中进行降噪。

此时损失函数中的 KL 散度计算为：
$$\mathrm{KL}\left(q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t,\boldsymbol{x}_0)|p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)\right)=\mathrm{KL}\left(\mathcal{C}(\boldsymbol{\theta}_{\mathrm{post}}(\boldsymbol{x}_t,\boldsymbol{x}_0))|\mathcal{C}(\boldsymbol{\theta}_{\mathrm{post}}(\boldsymbol{x}_t,\hat{\boldsymbol{x}}_0))\right),$$
而这两个 categorical 分布之间的散度进一步写为：
$$\sum_k\boldsymbol{\theta}_\mathrm{post}(\boldsymbol{x}_t,\boldsymbol{x}_0))_k\cdot\left.\log\frac{\boldsymbol{\theta}_\mathrm{post}(\boldsymbol{x}_t,\boldsymbol{x}_0))_k}{\boldsymbol{\theta}_\mathrm{post}(\boldsymbol{x}_t,\hat{\boldsymbol{x}}_0))_k}\right. $$
