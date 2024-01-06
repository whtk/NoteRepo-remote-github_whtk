> CVPR 2022，Nanyang Technological University

1. 把 VQ-VAE 和自回归模型结合已经可以生成高质量的合成图像
2. 但是自回归模型是有一个扫描顺序的，使得现有的自回归模型缺乏 global information
3. 本文表明，使用 VQ-VAE 得到 content-rich discrete visual codebook 时，diffusion 模型可以生成带有 global context 的高质量样本
4. 提出的 VQ-DDM 可以实现 comparable performance，但是复杂度更低，同时可以超过其他的 VQ+自回归的生成方法

## Introduction

1. VQ+自回归包括 PixelCNN 系列、transformer 系列，但是这些方法都只是根据已有的预测未知的，从而引入 inductive bias
2. DDPM 可以很好的缓解 global information 缺失的问题，DDPM 的问题在于生成和计算很耗时
3. 提出 VQ-DDM，包含离散 VAE 和离散 diffusion 模型，有两个 statge：
	1. 学习图像 abundant and efficient 的离散表征
	2. 通过离散的 diffusion 拟合这些 codes 的先验分布
4. 由于  codebook 的 bias 会限制生成质量，提出 re-build and fine-tune(ReFiT) strategy 来构造 codebook，也可以减少参数的数量

## 预备知识

### 连续状态空间下的 diffusion 模型（略）

### 图像的离散表征

给定 codebook $\mathbb{Z}\in\mathbb{R}^{K\times d}$，$K$ 为 latent variables 的数量，$d$ 为每个 latent variables 的维度，然后通过 encoder $E$ 将高维输入 $\mathbf{x}\in\mathbb{R}^{c\times H\times W}$ 压缩到 latent tensor $\mathbf{h}\in\mathbb{R}^{h\times w\times d}$，而 $\mathbf{z}$ 为量化后的 $\mathbf{h}$。decoder 用于从量化后的表征 $\mathbf{z}_q$ 中重构输入：
$$\begin{aligned}\mathbf{z}&=\mathrm{Quantize}(\mathbf{h}):=\arg\min_k||h_{i,j}-z_k||,\\\hat{\mathbf{x}}&=D(\mathbf{z})=D(\mathrm{Quantize}(E(\mathbf{x}))).\end{aligned}$$
由于量化操作不可为微，于是采用 straight-through gradient estimator 将来自 decoder 的重构误差反向传播到 encoder。模型以端到端的方式训练如下：
$$L=||\mathbf{x}-\hat{\mathbf{x}}||^2+||sg[E(\mathbf{x})]-\mathbf{z}||+\beta||sg[\mathbf{z}]-E(\mathbf{x})||$$
此后，VQ-GAN 将 VQ-VAE 进行拓展，将 L1 和 L2 损失替换为感知损失，而且加了一个额外的 discriminator。

codebook 训练的过程类似于聚类，聚类中心就是离散 latent codes。

## 方法

![](image/Pasted%20image%2020231011164455.png)

如图，首先通过 VQ-VAE 将图像压缩到 discrete variables，然后采用 diffusion 模型来拟合这些 codes 的联合分布。

图中黑色部分表示通过 uniform resampling 引入的噪声，采样时，latent codes 先从均匀的 categorical distribution 中采样，然后执行 $T$ 次的 reverse process 来得到 target latent codes，最终 latent codes 通过 decoder 生成图像。

### 离散 diffusion 模型

假设一共有 $K$ 个类，$z_{t}\in\{1,\ldots,K\}$，其对应的 one-hot 表征为 $\mathbf{z}_t\in\{0,1\}^K$，对应的分布记为 $\mathbf{z}_t^{\mathrm{logits}}$，此时离散的 diffusion 过程可以写为：
$$q(\mathbf{z}_t|\mathbf{z}_{t-1})=\mathrm{Cat}(\mathbf{z}_t;\mathbf{z}_{t-1}^\mathrm{logits}\mathbf{Q}_t)$$
这里的 $\mathbf{Q}_t$ 为传输矩阵，本文用的是 $\mathbf{Q}_{t}=(1-\beta_{t})\mathbf{I}+\beta_{t}/K$，即 $\mathbf{z}_t$ 有 $1-\beta_t$ 的概率保持上一个 times step 的状态，有 $\beta_t$ 的概率从均匀分布中采样，也就是可以进一步写为：
$$q(\mathbf{z}_t|\mathbf{z}_{t-1})=\mathrm{Cat}(\mathbf{z}_t;(1-\beta_t)\mathbf{z}_{t-1}^\mathrm{logits}+\beta_t/K)$$
而且可以直接从 $\mathbf{z}_0$ 中计算 $\mathbf{z}_t$：
$$\begin{gathered}q(\mathbf{z}_t|\mathbf{z}_0)=\mathrm{Cat}(\mathbf{z}_t;\bar{\alpha}_t\mathbf{z}_0+(1-\bar{\alpha}_t)/K)\\\\or\quad q(\mathbf{z}_t|\mathbf{z}_0)=\mathrm{Cat}(\mathbf{z}_t;\mathbf{z}_0\bar{\mathbf{Q}}_t);\bar{\mathbf{Q}}_t=\prod_{s=0}^t\mathbf{Q}_s.\end{gathered}$$
然后采用  cosine noise schedule，即：
$$\bar{\alpha}=\frac{f(t)}{f(0)},\quad f(t)=\cos\left(\frac{t/T+s}{1+s}\times\frac\pi2\right)^2$$
然后采用贝叶斯规则，后验 $q(\mathbf{z}_{t-1}|\mathbf{z}_t,\mathbf{z}_0)$ 计算为：
$$\begin{gathered}
q(\mathbf{z}_{t-1}|\mathbf{z}_t,\mathbf{z}_0)
=\operatorname{Cat}(\mathbf{z}_t;\boldsymbol{\theta}(\mathbf{z}_t,\mathbf{z}_0)/\sum_{k=1}^K\theta_k(z_{t,k},z_{0,k})), \\
\boldsymbol{\theta}(\mathbf{z}_t,\mathbf{z}_0)=[\alpha_t\mathbf{z}_t^\text{logits}+(1-\alpha_t)/K] \\
\odot[\bar{\alpha}_{t-1}\mathbf{z}_0+(1-\bar{\alpha}_{t-1})/K]. 
\end{gathered}$$
之前的方法是，用神经网络 $\mu(\mathbf{z}_t,t)$ 从 $\mathbf{z}_t$ 中预测 $\hat{\mathbf{z}}_0$ ，此时 reverse 过程表示为：
$$\begin{aligned}
p_\theta(\mathbf{z}_0|\mathbf{z}_1)& =\mathrm{Cat}(\mathbf{z}_0|\hat{\mathbf{z}}_0),  \\
p_\theta(\mathbf{z}_{t-1}|\mathbf{z}_t)& =\operatorname{Cat}(\mathbf{z}_t|\operatorname{N}[\boldsymbol{\theta}(\mathbf{z}_t,\hat{\mathbf{z}}_0)]). 
\end{aligned}$$
这里的 $\operatorname{N}[\boldsymbol{\theta}(\mathbf{z}_t,\hat{\mathbf{z}}_0)]$ 表示 $\boldsymbol{\theta}(\mathbf{z}_t,\mathbf{z}_0)/\sum_{k=1}^K\theta_k(z_{t,k},z_{0,k})$。

本文则采用一个神经网络 $\mu(\mathbb{Z}_t,t)$ 来学习预测噪声 $n_t$，然后得到 $\hat{\mathbf{z}}_0$ 的 logits：
$$\hat{\mathbf{z}}_0=\mu(\mathbf{Z}_t,t)+\mathbf{Z}_t$$

注意，这里的神经网络是基于 $\mathbf{Z}_t\in\mathbb{N}^{h\times w}$。
>也就是图像的所有的表征。

此时的损失函数为：
$$\begin{gathered}\mathrm{KL}(q(\mathbf{z}_{t-1}|\mathbf{z}_t,\mathbf{z}_0)||p_\theta(\mathbf{z}_{t-1}|\mathbf{z}_t))=\sum_k\mathrm{N}[\boldsymbol{\theta}(\mathbf{z}_t,\mathbf{z}_0)]\times\log\frac{\mathrm{N}[\boldsymbol{\theta}(\mathbf{z}_t,\mathbf{z}_0)]}{\mathrm{N}[\boldsymbol{\theta}(\mathbf{z}_t,\hat{\mathbf{z}}_0)]}\end{gathered}$$
### Re-build and Fine-tune Strategy

离散的表征是来自 VQ-VAE 的 codebook 的，但是 codebook 可能会很大（如 $K=16384$），而转移矩阵的大小和 $K$ 成二次方的关系，即 $O(K^2T)$。

从而提出 Re-build and Fine-tune (ReFit) strategy 来减少 $K$ 的大小，提升重构性能。

这是 VQ-VAE 的损失函数：
$$L=||\mathbf{x}-\hat{\mathbf{x}}||^2+||sg[E(\mathbf{x})]-\mathbf{z}||+\beta||sg[\mathbf{z}]-E(\mathbf{x})||$$
可以发现，这里只有第二项才涉及到更新 codebook，且每次迭代只有被用上的那几个 code 才会被更新，大部分 code 都不会被更新。

提出，在已有训练好的 encoder 前提下，重构 codebook 使得所有的 code 都有机会被选择。假设 VQ-VAE 的 encoder 为 $E_s$，decoder 为 $D_s$，得到的 codebook 为 $\mathbb{Z}_t$，首先将每个图像 $\mathbf{x}\in\mathbb{R}^{c\times H\times W}$ 编码到 latent feature $\mathbf{h}\in\mathbb{R}^{d\times h\times w}$，下一步采样 $P$ 个 code（这里的 $P$ 远大于 $K_t$），由于训练 codebook 的过程是寻找聚类中心，直接对这 $P$ 个特征采用 k-means 聚类，然后采用聚类中心重构 $\mathbb{Z}_t$，然后把原始的 codebook 替换为重构后的 codebook，然后在训练好的 VQ-VAE 上进行 fine tune。

## 实验和分析（略）