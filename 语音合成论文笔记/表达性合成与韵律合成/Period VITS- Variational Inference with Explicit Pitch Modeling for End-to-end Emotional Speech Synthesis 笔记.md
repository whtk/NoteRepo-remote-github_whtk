> ICASSP 2023，LINE Corp & NAVER Corp

1. 一些全端到端的 TTS 通常会生成不稳定的 pitch contour（当数据集包含情感属性时）
2. 提出 Period VITS，端到端的 TTS 模型，引入了显式的 periodicity generator
    1. 引入 frame pitch predictor，从输入文本预测韵律特征，如 pitch 和 voicing flags
    2. periodicity generator 从这两个特征中生成 sample-level sinusoidal source，使得 decoder 可以重构 pitch
    3. 整个模型通过变分推断和对抗训练联合优化

> 本质就是在 VITS 中引入一个模块来计算 pitch 和 voicing flag，然后用这个计算周期，引入 HiFi-GAN decoder 中来实现韵律建模。

## Introduction

1. 之前的级连模型通常存在问题，如使用预定义特征、两个独立模型的分开优化
2. 基于 GAN 和 VAE 的全端到端模型，如 VITS，可以生成自然的语音，但是在情感语音合成等更具挑战性的任务上表现有限
3. 提出 Period VITS，可以在生成波形时显式地提供 sample-level 和 pitch-dependent 的 periodicity
    1. 包含两个主要模块：prior encoder 和 waveform decoder
    2. prior encoder 包含 frame prior network 和 frame pitch predictor，可以同时生成先验分布的参数和每帧的 prosodic feature；且主要通过 normalizing flows 来学习复杂的先验分布和 prosodic feature（如 pitch 和 voicing flag），从而得到正弦的 source signal
    3. decoder 端，periodic source 送入到 HiFiGAN vocoder 中的每个上采样表征中，来确保波形的韵律稳定性
    4. 训练的时候以变分的方式端到端优化
4. 之前已经有些方法关注语音信号中的 periodicity，但是这些方法通常使用预定义的 acoustic features，并且只优化 vocoder 部分，而本文的模型可以端到端训练，通过辅助的 pitch 信息来获得最优的 latent acoustic features；且采用非自回归模型，生成速度更快
5. 实验结果表明，提出的模型在多说话人情感 TTS 的表现优于所有 baseline，且在中性和悲伤风格上的 score 与录音相当

## 方法

### 概览

整体框架如图：
![](image/Pasted%20image%2020240406192937.png)

模型基于 [VITS- Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech 笔记](../VITS-%20Conditional%20Variational%20Autoencoder%20with%20Adversarial%20Learning%20for%20End-to-End%20Text-to-Speech%20笔记.md)，包含 posterior encoder 和 decoder，先验分布由 prior encoder 建模。

posterior encoder 将输入的 linear spectrogram 转换为 latent acoustic features，prior encoder 将文本转换为 latent features，decoder 从学习到的 latent features 重构波形。

引入 latent y 表示 prosodic feature，作为 VAE latent variable z 的一个独立的 source，来显式地建模生成语音的 pitch。采用最大化给定文本后波形的 log-likelihood 来训练，简化为优化似然下界：
$$\begin{aligned}
&\log p(x|c)=\log\int\int p(x,z,y|c)dzdy \\
&\geq\int\int q(z,y|x)\log\frac{p(x,z,y|c)}{q(z,y|x)}dzdy \\
&=\int\int q(z|x)q(y|x)\log\frac{p(x|z,y)p(z|c)p(y|c)}{q(z|x)q(y|x)}dzdy \\
&=E_{q(z,y|x)}[\log p(x|z,y)]-D_{KL}(q(z|x)||p(z|c)) \\
&-D_{KL}(q(y|x)||p(y|c)),
\end{aligned}$$
其中 $p$ 为生成模型的分布，$q$ 为近似后验分布，$E$ 为期望计算，$D_{KL}$ 为 KL 散度。假设 z 和 y 在给定 x 的条件下是独立的，因此 $q(z,y|x)$ 可以分解为 $q(z|x)q(y|x)$。

由于从 x 提取 pitch 是确定性操作，可以定义 $q(y|x)=\delta(y-y_{gt})$，将公式的第三项转换为：
$$-\log p(y_{gt}|c)+const.$$

其中 $y_{gt}$ 为 GT pitch 值。可以通过最小化预测值和真实值之间的 L2 norm 来优化，假设 $p(y|c)$ 为固定单位方差的高斯分布。公式中的三项可以解释为 VAE 的波形重构损失 $L_{recon}$、先验/后验分布的 KL 散度损失 $L_{kl}$ 和从文本重构 pitch 的损失 $L_{pitch}$。采用 mel-spectrogram loss 作为 $L_{recon}$。

prior encoder 建模的先验分布需要表示相同音素包含的丰富的发音变化。于是采用 [VISinger- Variational Inference with Adversarial Learning for End-to-End Singing Voice Synthesis](../歌声合成/VISinger-%20Variational%20Inference%20with%20Adversarial%20Learning%20for%20End-to-End%20Singing%20Voice%20Synthesis.md) 中提出的 frame prior network，将 phoneme-level 先验分布扩展到 frame-level。
> 此操作实验中证实可以稳定发音。

然后引入 frame pitch predictor，从 frame prior network 的 hidden layer 预测 frame-level 的韵律特征（基频（F0）和 voicing flag（v）），作为后续周期性生成器的输入。也 通过L2 norm 进行优化：
$$L_{pitch}=\|\log F_0-\log\hat{F}_0\|_2+\|v-\hat{v}\|_2.$$

先验分布通过 normalizing flow $f$ 来增强建模能力：
$$p(z|c)=N(f(z);\mu(c),\sigma(c))\left|\det\frac{\partial f(z)}{\partial z}\right|,$$
其中 $\mu(c)$ 和 $\sigma(c)$ 表示从文本表征中得到的可训练的均值和方差。

### 带有 periodicity generator 的 decoder

已经有研究表明，GAN-based vocoder 模型在从 acoustic features 重构波形时通常会产生 artifacts，因为无法估计 pitch 和 periodicity。作者发现这些 artifacts 在端到端 TTS 模型中也有，尤其在训练集包含大 pitch 方差的情感数据集上。于是用 sine-based source signal 显式地建模语音波形的周期成分，但是不能直接将其整合到 VITS 中的 HiFi-GAN-based decoder（即 vocoder）中，因为 sine-based source signal 应该是 sample-level 特征，而 HiFi-GAN 的输入通常是 frame-level acoustic feature。

受 Period-HiFi-GAN 的启发，如图：
![](image/Pasted%20image%2020240406195450.png)

关键在于，将 下采样层用于 sample-level 的 pitch 相关的输入来匹配上采样的 frame-level 的分辨率。

用 periodicity generator 生成 sample-level 的 periodic source（将 sinusoidal source、voiceing flag 和高斯噪声一起作为输入）。但是并没有直接将 pre-cov 得到的 sample-level 的输出直接加到上采样层中，因为发现会降低性能。


### 训练目标

除了前面的 loss，还采用了 GAN loss $L_{adv}$ 和 feature matching loss $L_{fm}$ 来训练 waveform 的 GAN，还有 L2 duration loss $L_{dur}$ 来训练 duration model。总的 loss 为：
$$L_{total}=L_{recon}+L_{kl}+L_{pitch}+L_{dur}+L_{adv}+L_{fm}.$$

## 实验（略）
