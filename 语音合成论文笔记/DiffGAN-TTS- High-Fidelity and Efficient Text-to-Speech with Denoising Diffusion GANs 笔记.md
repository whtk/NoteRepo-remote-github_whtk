> preprint，腾讯 AI Lab，yu dong

1. DDPM 很难用于实时语音处理
2. 提出 DiffGAN-TTS，基于 denoising diffusion generative adversarial networks，采用对抗训练来近似 denoising 的分布
3. multi-speaker TTS 实验表明，DiffGAN-TTS 只要 4 个 denoising steps 就可以生成高质量的样本
4. 提出 active shallow diffusion mechanism  进一步加速训练

> 其实就是把 DiffGAN 用在语音合成中以加速采样，然后模型结构用的还是 FastSpeech 2 的那一套，同时提出 active shallow diffusion mechanism，先训练一个模型冻住参数，然后训练 diffusion decoder 的部分来进一步加速训练。

## Introduction

1. 采用 expressive acoustic generator 来建模 denoising distribution，在推理时允许大的 denoising steps，可以极大减少 denoising steps 和加速采样，同时引入 active shallow diffusion mechanism 来进一步加速采样
2. 设计了一个 two stage 的方案：
	1. stage 1 训练一个基本的 acoustic model
	2. stage 2 训练 diffusion model

## Diffusion 模型（略）

## DiffGAN-TTS

![](image/Pasted%20image%2020231001094946.png)

尽管 DDPM 可以建模复杂的分布，但是其推理速度很慢。[Tackling the Generative Learning Trilemma with Denoising Diffusion GANs 笔记](../图像合成系列论文阅读笔记/Tackling%20the%20Generative%20Learning%20Trilemma%20with%20Denoising%20Diffusion%20GANs%20笔记.md) 通过采用 GAN 来建模非高斯分布，从而使得 diffusion 的 step size 可以变大，本文将其引入 multi-speaker TTS。

### Acoustic generator 和 Discriminator

本文关注于 multi-speaker TTS，模型输入 phoneme sequence $\mathbf{y}$，然后使用 multi-speaker generator 生成中间的 mel 谱 特征 $\mathbf{x}_0$，然后采用 HiFi-GAN vocoder 来产生时域波形。这里的 acoustic generator 就是用 DiffGAN 来建模的。

训练流程如下：
![](image/Pasted%20image%2020231001095120.png)

采用 conditional GAN 来建模 denoising distribution，也就是训练一个 conditional generator $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ 来估计分布 $q(\mathbf{x}_{t-1}|\mathbf{x}_t)$，损失为最小化每个 time step 的 散度 $D_\mathrm{adv}$：
$$\min_\theta\sum_{t\geq1}\mathbb{E}_{q(\mathbf{x}_t)}[D_{\mathrm{adv}}(q(\mathbf{x}_{t-1}|\mathbf{x}_t)||p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t))]$$
采用  least-squares GAN (LS-GAN) 训练公式来最小化 $D_\mathrm{adv}$。

记 speaker-ID 为 $s$，那么 discriminator 是 diffusion-step-dependent and speaker-aware，其结构如图 1 c，而且分为 conditional and unconditional 两个输出（ diffusion step embedding 和 speaker embedding 被视为 condition）。然后同样采用 DiffGAN 的隐式建模，即 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t):=q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0=f_\theta(\mathbf{x}_t,t))$，训练的时候，首先得到预测的 $\mathbf{x}_0^{\prime}=f_\theta(\mathbf{x}_t,t)$，然后从分布 $q(\mathbf{x}_{t-1}^{\prime}|\mathbf{x}_0^{\prime},\mathbf{x}_t)$ 中采样得到 $\mathbf{x}_{t-1}^{\prime}$，然后 $(\mathbf{x}_{t-1}^{\prime},\mathbf{x}_t)$ 输入到 discriminator 计算 $D_\mathrm{adv}$（这个就是负样本），而正样本是 $(\mathbf{x}_{t-1},\mathbf{x}_t)$。
> 和 DiffGAN 中生成 $\mathbf{x}_0^{\prime}$ 需要从 latent variable $z\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ 中采样不同，由于这里的 decoder 输入有 variance-adapted text encodings 和 speaker-ID，所以不需要额外的 latent variable。也就是说，generator $G_\theta(\mathbf{x}_t,\mathbf{y},t,s)$ 基于 phoneme input $\mathbf{y}$、diffusion step index $t$ 和 speaker ID $s$ 来从 $\mathbf{x}_t$ 预测 $\mathbf{x}_0$。

### 训练损失

discriminator 损失为：
$$\begin{gathered}\mathcal{L}_D=\sum_{t\geq1}\mathbb{E}_{q(\mathbf{x}_t)q(\mathbf{x}_{t-1}|\mathbf{x}_t)}[(D_\phi(\mathbf{x}_{t-1},\mathbf{x}_t,t,s)-1)^2]\\+\mathbb{E}_{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}[D_\phi(\mathbf{x}_{t-1},\mathbf{x}_t,t,s)^2]\end{gathered}$$
而为了训练 acoustic generator，还使用了 feature matching loss $\mathcal{L}_{fm}$，使得 discriminator 的  feature space 中 real 和 fake 数据进尽可能地相似：
$$\mathcal{L}_{fm}=\mathbb{E}_{q(\mathbf{x}_t)}[\sum_{i=1}^N\|D_\phi^i(\mathbf{x}_{t-1},\mathbf{x}_t,t,s)-D_\phi^i(\mathbf{x}_{t-1}^{\prime},\mathbf{x}_t,t,s)\|_1]$$
然后还有 Acoustic reconstruction loss（和 FastSpeech 2 ）相似：
$$\begin{aligned}
\mathcal{L}_{recon}& =\mathcal{L}_{mel}(\mathbf{x}_0,\mathbf{x}_0^{\prime})+\lambda_d\mathcal{L}_{duration}(\mathbf{d},\hat{\mathbf{d}})+  \\
&\lambda_p\mathcal{L}_{pitch}\left(\mathbf{p},\hat{\mathbf{p}}\right)+\lambda_e\mathcal{L}_{energy}(\mathbf{e},\hat{\mathbf{e}})
\end{aligned}$$
上面的 $\mathbf{d},\mathbf{b},\mathbf{e}$ 分别为  target duration, pitch 和 energy 。

generator 总的损失为：
$$\mathcal{L}_{G}=\mathcal{L}_{adv}+\mathcal{L}_{recon}+\lambda_{fm}\mathcal{L}_{fm}$$
其中：
$$\mathcal{L}_{adv}=\sum_{t\geq1}\mathbb{E}_{q(\mathbf{x}_t)}\mathbb{E}_{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}[(D_\phi(\mathbf{x}_{t-1},\mathbf{x}_t,t,s)-1)^2]$$

### Active shallow diffusion mechanism

![](image/Pasted%20image%2020231001105524.png)

采用 two-stage 的训练方案：

stage 1，训练一个 basic acoustic model，记为 $G_\psi^\text{base}{ ( \mathbf{y},s)}$，其损失为 $$\min_\psi\sum_{t\geq0}\mathbb{E}_{q(\mathbf{x}_t)}[\mathrm{Div}(q_\mathrm{diff}^t(G_\psi^\mathrm{base}(\mathbf{y},s)),q_\mathrm{diff}^t(\mathbf{x}_0))]$$
其中，$q_{\mathrm{diff}}^t(\cdot)$ 为 diffusion 的采样过程，即 $\mathbf{x}_t=q_{\mathrm{diff}}^t(\mathbf{x}_0)$。这个目标函数使得模型无法区分 GT 中的 acoustic features 和 预测的特征。
> 其实就是让模型在输入为 $\mathbf{y},s$ 时，输出尽可能接近 $\mathbf{x}_{0}$。

stage 2，basic acoustic model 权重保持冻结，此时 diffusion decoder 对应的损失从 $D_{\mathrm{adv}}(q(\mathbf{x}_{t-1}|\mathbf{x}_t)||p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t))$ 变成 $D_\mathrm{adv}(q(\mathbf{x}_{t-1}|\mathbf{x}_t)||p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t,\hat{\mathbf{x}}_0))$。

### 模型架构

acoustic generator 中的 transformer encoder 用的是 [FastSpeech 2- Fast and High-Quality End-to-End Text to Speech 笔记](FastSpeech%202-%20Fast%20and%20High-Quality%20End-to-End%20Text%20to%20Speech%20笔记.md) 中的 FFT 架构，variance adaptor 也是用它的。
> aligner 用的是 hidden-Markov-model (HMM)-based forced aligner。

diffusion decoder 用的是 non-causal WaveNet（[WaveNet- A Generative Model for Raw Audio 笔记](WaveNet-%20A%20Generative%20Model%20for%20Raw%20Audio%20笔记.md)），但是由于输入是 mel 谱，所以 dilation 为 1。

discriminator 采用纯卷积网络。

mel decoder 为 4 层  FFT blocks。

## 实验（略）