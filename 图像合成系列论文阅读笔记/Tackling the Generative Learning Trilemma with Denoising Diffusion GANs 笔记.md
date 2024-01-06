> ICLR，2022，NVIDIA

1. 生成模型存在三个关键的挑战（the generative learning trilemma）：
	1. 高采样质量
	2. mode coverage
	3. 快速采样
2. 现有模型都是在这三个点之间做 trade-off，diffusion 可以实现很高的质量和多样性，但是采样速度很慢，作者认为，原因在 Gaussian assumption
3. 为了实现大的 denoising step size 从而减少采样次数，提出通过使用一个complex multimodal distribution 来建模 denoising distribution
4. 引入 denoising diffusion generative adversarial networks (denoising diffusion GANs)，采用  multimodal conditional GAN 来建模每个 denoising step，可以实现快 2000× 的采样速度

> 什么是 multimodal distribution ：简单来说，就是分布函数有多个峰。而高斯分布通常是单峰的（只考虑协方差矩阵为对角阵）。

> 感觉这篇文章很有意思，在传统的 DDPM 中，reverse process 是通过公式计算的（所谓的 deterministic）以用于近似 denoising distribution，而这里则引入一个 GAN generator 来建模任意分布。从模型上看，DDPM 中的 $\mathbf{x}_t$ 是通过一个公式计算的，而这里则是先用 generator 来预测 $\mathbf{x}_{0}$，然后再基于一个转换公式来计算得到 $\mathbf{x}_{t-1}$，同时 discriminator 也对应改变。

## Introduction

1. diffusion models 通常假设 denoising distribution 为近似的高斯分布，但是当 denoising step 无穷小时，这个假设才成立，当使用稍微大一点的 step size 时，就需要  non-Gaussian multimodal distribution（在图像合成中，当多个 clean image 可以对应于相同的 noisy image 时，即满足）
2. 于是提出将 denoising distribution 参数化为 expressive multimodal distribution ，从而使得大的 denoising steps。提出 denoising diffusion GAN，其中 denoising distributions 通过 conditional GAN 来建模

## 背景

数据分布 $\mathbf{x}_0\sim q(\mathbf{x}_0)$ ，然后 forward process 逐渐加噪：
$$q(\mathbf{x}_{1:T}|\mathbf{x}_0)=\prod_{t\geq1}q(\mathbf{x}_t|\mathbf{x}_{t-1}),\quad q(\mathbf{x}_t|\mathbf{x}_{t-1})=\mathcal{N}(\mathbf{x}_t;\sqrt{1-\beta_t}\mathbf{x}_{t-1},\beta_t\mathbf{I}),$$
reverse denoising process 定义为：
$$p_\theta(\mathbf{x}_{0:T})=p(\mathbf{x}_T)\prod_{t\geq1}p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t),\quad p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)=\mathcal{N}(\mathbf{x}_{t-1};\mu_\theta(\mathbf{x}_t,t),\sigma_t^2\text{I}),$$
其中的 $\boldsymbol{\mu}_\theta(\mathbf{x}_t,t)$ 和 $\sigma_t^2$ 为 denoising model 的均值和方差，目标是最大化似然 $p_\theta(\mathbf{x}_0)=\int p_\theta(\mathbf{x}_{0:T})d\mathbf{x}_{1:T}$，实现则是通过最大化 ELBO，可以写为：
$$\mathcal{L}=-\sum_{t\geq1}\mathbb{E}_{q(\mathbf{x}_t)}\left[D_{\mathrm{KL}}\left(q(\mathbf{x}_{t-1}|\mathbf{x}_t)\|p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t))\right]+C\right. $$
其中的 $C$ 为常数，和 $\theta$ 无关。上式就是 denoising distribution $q(\mathbf{x}_{t-1}|\mathbf{x}_{t})$ 和 模型 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ 之间的 KL 散度，但是根据 DDPM 的论文，上式并不可求，于是 DDPM 把它写成了另一种形式（也就是引入 $\mathbf{x}_0$ 作为条件）。

在 diffusion 模型中，分布 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ 通常采用高斯分布建模。

## DENOISING DIFFUSION GANS

### 为什么需要 multimodal denoising distribution

在 DDPM 中，通常假设 $q(\mathbf{x}_{t-1}|\mathbf{x}_{t})$ 为高斯分布（但是是未知且不可求的），但是这不一定是正确的。

分布 $q(\mathbf{x}_{t-1}|\mathbf{x}_{t})$ 可以用贝叶斯公式写为 $q(\mathbf{x}_{t-1}|\mathbf{x}_t)\propto q(\mathbf{x}_t|\mathbf{x}_{t-1})q(\mathbf{x}_{t-1})$，其中 $q(\mathbf{x}_t|\mathbf{x}_{t-1})$ 为已知的高斯分布，$q(\mathbf{x}_{t-1})$ 为 time step $t$ 的边缘分布，在两种情况下，真实的 denoising distribution 才是高斯分布：
+ step size $\beta_t$ 无穷小，此时两种的乘积主要由 $q(\mathbf{x}_t|\mathbf{x}_{t-1})$ 主导，从而 $q(\mathbf{x}_{t-1}|\mathbf{x}_{t})$ 也是高斯（具体证明 1949 年，on the theory of stochastic processes, with particular reference to applications）
+ 边缘分布 $q(\mathbf{x}_{t-1})$ 也是高斯，此时 $q(\mathbf{x}_{t-1}|\mathbf{x}_{t})$ 也是高斯

本文认为，这两种情况都不成立，从而 $q(\mathbf{x}_{t-1}|\mathbf{x}_{t})$ 并不是高斯分布，下图给出了一个一维情况下的示例：
![](image/Pasted%20image%2020230930223114.png)
可以看到，随着 denoising step 增加， true denoising distribution 更复杂且多峰。

### 采用 CONITIONAL GANS 来建模 denoising distribution

目标是减少 denoising diffusion steps $T$，于是提出采用一个 expressive multimodal distribution 来建模  denoising distribution，而由于条件 GAN 可以建模复杂的分布，因此用来近似 $q(\mathbf{x}_{t-1}|\mathbf{x}_{t})$。

此时的 $T$ 可以很小（ $T \le 8$ ），此时训练是为了，匹配 conditional GAN generator 建模的分布 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ 和 $q(\mathbf{x}_{t-1}|\mathbf{x}_{t})$，这个过程采用对抗损失，通过最小化每个 step 的散度 $D_\mathrm{adv}$：
$$\min_\theta\sum_{t\geq1}\mathbb{E}_{q(\mathbf{x}_t)}\left[D_{\mathrm{adv}}(q(\mathbf{x}_{t-1}|\mathbf{x}_t)\|p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t))\right],$$
取决于不同的对抗训练，这里的 $D_\mathrm{adv}$ 可以是 Wasserstein distance, Jenson-Shannon divergence 或者 f-divergence，本文用的是 non-saturating GAN，此时的 $D_\mathrm{adv}$ 为 softened reverse KL 。

同时为了实现对抗训练，有一个 时间相关的 time-dependent discriminator $D_\phi(\mathbf{x}_{t-1},\mathbf{x}_t,t):\mathbb{R}^N\times\mathbb{R}^N\times\mathbb{R}\to[0,1]$，其输入为 $\mathbf{x}_{t-1},\mathbf{x}_t$，而判断的是 $\mathbf{x}_{t-1}$ 是否是 $\mathbf{x}_{t}$ 降噪后的结果。
> 注意：这里和一般的判别器判别的东西不一样！！

通过下式来训练 discriminator：
$$\min_\phi\sum_{t\geq1}\mathbb{E}_{q(\mathbf{x}_t)}[\mathbb{E}_{q(\mathbf{x}_{t-1}|\mathbf{x}_t)}[-\log(D_\phi(\mathbf{x}_{t-1},\mathbf{x}_t,t)]+\mathbb{E}_{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}[-\log(1-D_\phi(\mathbf{x}_{t-1},\mathbf{x}_t,t))]],$$
其中，fake 样本来自分布 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$，而 real 样本则是来自 $q(\mathbf{x}_{t-1}|\mathbf{x}_{t})$，同时由于第一项中需要从未知分布 $q(\mathbf{x}_{t-1}|\mathbf{x}_{t})$ 中采样，我们使用等式：
$$q(\mathbf{x}_t,\mathbf{x}_{t-1})=\int d\mathbf{x}_0q(\mathbf{x}_0)q(\mathbf{x}_t,\mathbf{x}_{t-1}|\mathbf{x}_0)=\int d\mathbf{x}_0q(\mathbf{x}_0)q(\mathbf{x}_{t-1}|\mathbf{x}_0)q(\mathbf{x}_t|\mathbf{x}_{t-1})$$
可以将第一项从写为：
$$\mathbb{E}_{q(\mathbf{x}_t)q(\mathbf{x}_{t-1}|\mathbf{x}_t)}[-\log(D_\phi(\mathbf{x}_{t-1},\mathbf{x}_t,t))]=\mathbb{E}_{q(\mathbf{x}_0)q(\mathbf{x}_{t-1}|\mathbf{x}_0)q(\mathbf{x}_t|\mathbf{x}_{t-1})}[-\log(D_\phi(\mathbf{x}_{t-1},\mathbf{x}_t,t))].$$

在之前的 DDPM 中，模型参数化为：$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t):=q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0=f_\theta(\mathbf{x}_t,t))$，然后采样时，用这个模型先预测 $\mathbf{x}_0$（不考虑预测噪声的那种方法），然后从分布 $q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)$ 中采样得到 $\mathbf{x}_{t-1}$（对应论文中 $\mathbf{x}_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)\right)+\sigma_t\mathbf{z}$ 这一步了），而且这个分布 $q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)$ 不管 step size 是多少都是高斯分布。类似地，本文定义 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ 为：
$$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t):=\int p_\theta(\mathbf{x}_0|\mathbf{x}_t)q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)d\mathbf{x}_0=\int p(\mathbf{z})q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0=G_\theta(\mathbf{x}_t,\mathbf{z},t))d\mathbf{z},$$
其中 $p_\theta(\mathbf{x}_{0}|\mathbf{x}_t)$ 为  GAN generator $G_\theta(\mathbf{x}_t,\mathbf{z},t):\mathbb{R}^N\times\mathbb{R}^L\times \mathbb{R}\to{\mathbb{R}}^N$ 强加的隐式分布。其输入为高斯噪声 $\mathbf{z}$  和 $\mathbf{x}_0$，输出为 $\mathbf{x}_t$。
> 这里的意思就是，由于 $q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)$ 是已知的，那么我只要定义一个 generator 来建模分布 $p_\theta(\mathbf{x}_{0}|\mathbf{x}_t)$，那么最终就可以得到 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ 的分布。

一些优点：
+ 由于 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ 的建模方式和原始的 DDPM 很相似，因此可以用原始 DDPM 的网络架构，区别在于，DDPM 中，$\mathbf{x}_0$ 被建模为 $\mathbf{x}_t$ 的一个确定的映射，而本文则是通过 generator 来从随机 latent variable $\mathbf{z}$ 来生成的。This is the key difference that allows our denoising distribution to become ultimodal and complex in contrast to the unimodal denoising model in DDPM
+ 采用一个网络来预测不同 time step $t$ 下的 $\mathbf{x}_{t-1}$ 是很困难的，但是本文则只需要从 $\mathbf{x}_{t}$ 中预测一个 $\mathbf{x}_{0}$

整个训练流程如图：
![](image/Pasted%20image%2020230930231706.png)

## 相关工作（略）

## 实验（略）
