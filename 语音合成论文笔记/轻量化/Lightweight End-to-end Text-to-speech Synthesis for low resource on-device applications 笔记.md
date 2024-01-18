> Interspeech 2023 Workshop SSW Program，Alexa AI

1. 当前的 E2E SOTA 模型计算复杂且耗内存，从而不适用于实时的离线设备
2.  提出 Lightweight E2E-TTS 只需很少的计算资源实现高质量的合成
3. 在 LJSpeech 上实验，可以减少 90% 的参数且 10x faster in RTF

> 总结，没有任何创新点。

## Introduction 

1. 提出在单个架构中联合训练轻量化的 acoustic 模型和 vocoder 模型，主要贡献如下：
	1. 提出  Lightweight E2E-TTS (LE2E)，联合训练 LightSpeech 和 Multi-Band MelGAN 
	2. 基于现有的 GAN 的 discriminator 提出 upgraded loss objective
	3. 在 LJSpeech 上实现 3.79 的 MOS

## 相关工作（略）

## 方法

模型结构如图：
![](image/Pasted%20image%2020231224211438.png)

其中 generator 包括 acoustic latent encoder 和 acoustic decoder，discriminator 包括 multi-period (MPD) 和 multi-resolution discriminators (MRD)。

### Generator

acoustic latent model 受 [LightSpeech- Lightweight and Fast Text to Speech with Neural Architecture Search 笔记](LightSpeech-%20Lightweight%20and%20Fast%20Text%20to%20Speech%20with%20Neural%20Architecture%20Search%20笔记.md) 启发，但是训练用于生成 unsupervised acoustic latents（可以直接用于 vocoder 的）。

模型输入 phoneme 和 positional embedding，分为三部分：
+ text encoder $E$，通过 transformer 层生成  positional aware phoneme embeddings
+ variance adaptor $V$，输入为 前面的 embedding，包含 duration predictor 和 pitch predictor
+ acoustic decoder $D$，输入为 variance adaptor 的输出

vocoder 则基于  Multi-Band MelGAN，先生成不同子频带的波形，最后通过 Pseudo Quadrature Mirror Filter bank 生成最终的音频。

### Discriminators

采用 BigVGAN 中提出的一系列 discriminator，包含 multi-period discriminator (MPD) 和 multi-resolution discriminator (MRD)，且每个 discriminator 都包含一些 sub-discriminator，两个分别在时域和频域进行判别。

### 训练目标函数

包含 duration predictor、pitch predictor loss 和 waveform level 的 GAN loss 和 回归损失。

Duration loss：用的是 MSE loss，GT duration 来自 Kaldi 中的外部 aligner

Pitch loss：对 pitch 采用 256-bin 的量化，然后用交叉熵损失，通过 softmax 来预测 logits

对抗训练损失：
假设有 $K$ 个 discriminator，记为 $D_k$ 。

GAN loss 用的就是最小均方误差损失，生成器和判别器的 loss 如下：
$$\begin{gathered}\mathcal{L}_D(x,\hat{x})=\min_{D_k}\mathbb{E}_{\boldsymbol{x}}\left[(D_k(\boldsymbol{x}-1)^2\right]+\mathbb{E}_{\hat{\boldsymbol{x}}}\left[(D_k(\hat{\boldsymbol{x}})^2\right]\\\\\mathcal{L}_G(\hat{\boldsymbol{x}})=\min_G\mathbb{E}_{\hat{\boldsymbol{x}}}\left[(D_k(\hat{\boldsymbol{x}}-1)^2\right]\end{gathered}$$

Feature matching loss 为 L1 loss：
$$\mathcal{L}_{FM}(\boldsymbol{x},\hat{\boldsymbol{x}})=\mathbb{E}_{\boldsymbol{x},\boldsymbol{\hat{x}}}\left[\sum_{i=1}^T\frac1{N_i}\left\|D_k^{(i)}(\boldsymbol{x})-D_k^{(i)}(\hat{\boldsymbol{x}})\right\|_1\right]$$

采用两种重构损失来辅助训练：
Multi-resolution STFT loss 为预测的和真实的 STFT 线性 spectrogram 的幅度的差异和：
$$\begin{gathered}\mathcal{L}_{se}(\boldsymbol{s},\boldsymbol{\hat{s}})=\frac{\|\boldsymbol{s}-\boldsymbol{\hat{s}}\|}{\|\boldsymbol{s}\|_F},\quad\mathcal{L}_{mag}(\boldsymbol{s},\boldsymbol{\hat{s}})=\frac1S\left\|\log(\boldsymbol{s})-\log(\boldsymbol{\hat{s}})\right\|\\\mathcal{L}_{STFT}(\boldsymbol{x},\hat{\boldsymbol{x}})=\frac1M\sum_{m=1}^M\mathbb{E}_{\boldsymbol{x},\hat{\boldsymbol{x}}}\left[\mathcal{L}_{sc}(\boldsymbol{s},\hat{\boldsymbol{s}})+\mathcal{L}_{mag}(\boldsymbol{s},\hat{\boldsymbol{s}})\right]\end{gathered}$$
其中的 $M$ 表示不同的 STFT resolution。且是在  full-band 和 sub-band predictions 上计算的 loss，总的 multi-resolution STFT loss 定义为：
$$\mathcal{L}_{STFT}(\boldsymbol{x},\boldsymbol{\hat{x}})=\frac12\left(\mathcal{L}_{STFT}^{full}(\boldsymbol{x},\boldsymbol{\hat{x}})+\mathcal{L}_{STFT}^{sub}(\boldsymbol{x},\boldsymbol{\hat{x}})\right)$$

Mel-Spectrogram loss 也称为 power loss，在 full-band prediction 下定义为 L1 loss：
$$\mathcal{L}_{mel}(\boldsymbol{x},\hat{\boldsymbol{x}})=\mathbb{E}_{\boldsymbol{x},\hat{\boldsymbol{x}}}\left[\left\|\boldsymbol{m}-\hat{\boldsymbol{m}}\right\|_1\right]$$

最后，总损失为：
$$\begin{aligned}\mathcal{L}&=\mathcal{L}_{dur}+\mathcal{L}_{f0}+\mathcal{L}_{G}+\lambda_{FM}\mathcal{L}_{FM}+\\&+\lambda_{mel}\mathcal{L}_{mel}+\lambda_{STFT}\mathcal{L}_{STFT}\end{aligned}$$
