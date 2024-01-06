> Google、MIT，ICLR 2019

1. 提出可以控制隐变量属性（latent attributes）的 seq2seq TTS 模型
2. 模型基于 VAE 框架，是一个条件生成模型，有两个 level 的 hierarchical latent variables
	1. 第一个 level 是 categorical variable，用于表示属性组（attribute group），有可解释性
	2. 第二个 level 基于第一个，为多元高斯分布，实现特定的属性配置，且可以解耦并精细化控制
3. 训练一个可以从带噪音频中推理出 speaker 和 style attributes 的模型，然后可以用来合成 clean speech

## Introduction

1. 在 [Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron 笔记](Towards%20End-to-End%20Prosody%20Transfer%20for%20Expressive%20Speech%20Synthesis%20with%20Tacotron%20笔记.md) 的基础上，拓展 Tacotron 2 来建模两个不同的 latent space：一个用于 labeled 属性，一个用于 unlabeled 属性
2. 每个 latent variable 都以采用 Gaussian mixture priors 的 VAE 框架建模，从而使得 latent space 可以：
	1. 学习 disentangled attribute representations，每个维度都控制不同的因子
	2. discover a set of interpretable clusters, each of which corresponds to a representative mode in the training data (e.g., one cluster for clean speech and another for noisy speech)
	3. provide a systematic sampling mechanism from the learned prior
3. 主要贡献：
	1. 提出  principled probabilistic hierarchical generative model，可以提升采样稳定性，实现 disentangled attribute control、提升 interpretability 和 quality
	2. 将 latent encoding 通过两个混合分布分解成两个模型，分别建模 speaker attributes 和 latent attributes，从而使得模型可以基于 speaker 或 latent encoding 进行语音合成

## 模型

Tacotron-like TTS 以文本序列 $\mathbf{Y}_t$ 和 optional 的 categorical label $\mathbf{y}_o$ 作为输入，然后使用 自回归的 decoder 一帧一帧地预测声学特征 $\mathbf{X}$，然后通过最小化 MSE 重构误差进行训练。其可以看成是，拟合概率模型 $p(\mathbf{X}\mid\mathbf{Y}_t,\mathbf{y}_o)=\prod_np(\mathbf{x}_n\mid \mathbf{x}_1,\mathbf{x}_2,\ldots,\mathbf{x}_{n-1},\mathbf{Y}_t,\mathbf{y}_o)$，最大化生成的数据的似然。其中每帧 $\mathbf{x}_n$ 的条件分布都是固定方差的高斯分布，均值为 step $n$ 时 decoder 的预测输出。

其实可以很方便地引入 unpredictable latent attributes。于是采用  graphical model with hierarchical latent variables，可以捕获这些 attributes。
![](image/Pasted%20image%2020230905153221.png)
### 基于 HIERARCHICAL LATENT VARIABLES 的条件生成模型

引入两个 latent variable $\mathbf{y}_l,\mathbf{z}_l$，其中 $\mathbf{y}_l$ 是一个 K-way categorical discrete variable，称为 latent attribute class，$\mathbf{z}_l$ 是一个 D-dimensional continuous variable，称为 latent attribute representation，为了基于 $\mathbf{Y}_t,\mathbf{y}_o$ 生成语音 $\mathbf{X}$，$\boldsymbol{y}_l$ 首先从先验 $p(\mathbf{y}_l)$ 中采样，然后从条件分布 $p\left(\mathbf{z}_l \mid \boldsymbol{y}_l\right)$ 中采样得到 $\boldsymbol{z}_l$，然后语音帧序列来自 $p(\mathbf{X}\mid\mathbf{Y}_t,\boldsymbol{y}_o,\boldsymbol{z}_l)$，联合分布可以写为：
$$p(\mathbf{X},\mathbf{y}_l,\mathbf{z}_l\mid\mathbf{Y}_t,\mathbf{y}_o)=p(\mathbf{X}\mid\mathbf{Y}_t,\mathbf{y}_o,\mathbf{z}_l)\left.p(\mathbf{z}_l\mid\mathbf{y}_l)\right.p(\mathbf{y}_l)$$
而且假设 先验 $p(\mathbf{y}_l)=\frac1K$（也就是没有任何已知信息），且 $p(\mathbf{z}_{l}\mid\mathbf{y}_{l})=\mathcal{N}\left(\boldsymbol{\mu}_{\mathbf{y}_{l}},\mathrm{diag}(\boldsymbol{\sigma}_{\mathbf{y}_{l}})\right)$ 为方差和均值都是可学习的高斯分布，从而 $\mathbf{z}_l$ 的边缘分布为权重相同的混合高斯分布（GMM）。

### 变分推理和训练

条件输出分布 $p(\mathbf{X}\mid\mathbf{Y}_t,\mathbf{y}_o,\mathbf{z}_l)$ 采用神经网络来建模，用的是 VAE 的框架，采用 variational distribution $q(\mathbf{y}_l\mid\mathbf{X})q(\mathbf{z}_l\mid\mathbf{X})$ 来近似后验分布 $p(\mathbf{y}_{l},\mathbf{z}_{l}\mid\mathbf{X},\mathbf{Y}_{t},\mathbf{y}_{o})$，其中 $q(\mathbf{y}_l\mid\mathbf{X})$ 采用高斯分布建模（对角斜方差矩阵），其均值和方差通过神经网络计算。对于 $q(\mathbf{z}_l\mid\mathbf{X})$，认为它是 $p(\mathbf{y}_{l}\mid\mathbf{X})$ 的近似，从而可以写为：
$$p(\mathbf{y}_l|\mathbf{X})=\int_{\mathbf{z}_l}p(\mathbf{y}_l|\mathbf{z}_l)p(\mathbf{z}_l|\mathbf{X})d\mathbf{z}_l=\mathbb{E}_{p(\mathbf{z}_l|\mathbf{X})}\left[p(\mathbf{y}_l|\mathbf{z}_l)\right]\approx\mathbb{E}_{q(\mathbf{z}_l|\mathbf{X})}\left[p(\mathbf{y}_l|\mathbf{z}_l)\right]:=q(\mathbf{y}_l|\mathbf{X})$$
其实就是 GMM 的解。

然后就和 VAE 差不多，通过最大 ELBO 来训练：
$$\begin{aligned}\mathcal{L}(p,q;\mathbf{X},\mathbf{Y}_t,\mathbf{y}_o)&=\mathbb{E}_{q(\mathbf{z}\mid\mathbf{X})}[\log p(\mathbf{X}\mid\mathbf{Y}_t,\mathbf{y}_o,\mathbf{z}_l)]\\&-\mathbb{E}_{q(\mathbf{y}_l\mid\mathbf{X})}[D_{KL}(q(\mathbf{z}_l\mid\mathbf{X})\mid\mid p(\mathbf{z}_l\mid\mathbf{y}_l))]-D_{KL}(q(\mathbf{y}_l\mid\mathbf{X})\mid\mid p(\mathbf{y}_l))\end{aligned}$$
其中 $q(\mathbf{z}_l\mid\mathbf{X})$ 采用 MC 采样来估计。

### A CONTINUOUS ATTRIBUTE SPACE FOR CATEGORICAL OBSERVED LABELS

Categorical observed labels 如 speaker identity 可以看成是 continuous attribute space 的一种 categorization。
给定观察到的 label，这些 attribute 仍然有一些 variation，，目标是学习这种 continuous attribute space，来建模 within-class variation。

于是在 $\mathbf{y}_o$ 和 $\mathbf{X}$ 之间引入 continuous latent variable $\mathbf{z}_o$，称为 observed attribute representation，每个 observed class（如说话人）都形成 continuous space 下的一个  mixture component，其条件分布为高斯：$p(\mathbf{z}_o\mid\mathbf{y}_o)=\mathcal{N}(\boldsymbol{\mu}_{\mathbf{y}_o},\operatorname{diag}(\boldsymbol{\sigma}_{\mathbf{y}_o}))$，和之前一样，变分分布 $q(\mathbf{z}_o\mid\mathbf{X})$ 也用神经网络来建模，其 ELBO 为：
$$\begin{aligned}\mathcal{L}_o(p,q;\mathbf{X},\mathbf{Y}_t,\mathbf{y}_o)&=\mathbb{E}_{q(\mathbf{z}_o|\mathbf{X})q(\mathbf{z}_l|\mathbf{X})}[\log p(\mathbf{X}\mid\mathbf{Y}_t,\mathbf{z}_o,\mathbf{z}_l)]-D_{KL}(q(\mathbf{z}_o\mid\mathbf{X})\mid\mid p(\mathbf{z}_o\mid\mathbf{y}_o))\\&-\mathbb{E}_{q(\mathbf{y}|\mathbf{X})}[D_{KL}(q(\mathbf{z}_l\mid\mathbf{X})\mid\mid p(\mathbf{z}_l\mid\mathbf{y}_l))]-D_{KL}(q(\mathbf{y}_l\mid\mathbf{X})\mid\mid p(\mathbf{y}_l)).\end{aligned}$$
为了使的 $\mathbf{z}_o$ 能够从 latent attribute 中解耦 observed attribute，$p(\mathbf{z}_o\mid\mathbf{y}_o)$ 的方差初始化小于 $p(\mathbf{z}_l\mid\mathbf{y}_l)$。

当方差固定且接近 0 时，这个公式就是使用 lookup table。

### 神经网络架构

将分布 $p(\mathbf{X}|\mathbf{Y}_t,\mathbf{z}_o,\mathbf{z}_l),q(\mathbf{z}_l\mid\mathbf{X}),q(\mathbf{z}_o\mid\mathbf{X})$ 用神经网络来建模，分别对应图中的  synthesizer, observed encoder, 和 latent encoder，其中 synthesizer 对应 Tacotron 2 架构，包含 text encoder 和 自回归的 speech decoder。$\mathbf{z}_l,\mathbf{z}_o$ 直接拼接到 text encoder 的输出中，然后作为 decoder 的输入。两个后验分布 $q(\mathbf{z}_l\mid\mathbf{X}),q(\mathbf{z}_o\mid\mathbf{X})$ 都通过 recurrent encoder 将可变长度的输入映射到两个固定维度的向量，分别对应均值和方差。

![](image/Pasted%20image%2020230905162407.png)

## 相关工作（略）

## 实验（略）

