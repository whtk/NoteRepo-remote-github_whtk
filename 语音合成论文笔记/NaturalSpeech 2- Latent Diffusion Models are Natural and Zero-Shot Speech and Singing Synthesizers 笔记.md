> MSRA，2023，ICLR 2024

1. 将大规模、多说话人 TTS 系统拓展到域外数据集，需要捕获语音中的多种特征如 说话人身份、韵律、风格等
2. 现有的大规模的 TTS 通常将语音量化为离散的 token，然后采用语言模型一个接一个生成 token，从而导致：
	1. 韵律不稳定
	2. word skipping/repeating
	3. 合成质量差
3. 提出 NaturalSpeech 2，采用基于 RVQ 的音频 codec 来获得量化后的 latent vector，采用 diffusion 来基于文本生成这些  latent vector
4. 设计了一个语音 prompt 机制，在 diffusion 中利用 in-context learning 
5. 在 zero-shot 时，在 prosody/timbre similarity, robustness,  voice quality 都超过了现有的 TTS 系统

> 用 diffusion 来建模从文本到 latent 的过程，这里的 latent 是通过 codec 生成的离散向量（RVQ），但是 diffusion 生成的是连续的向量。同时还有一个创新就是，在 pitch/duration predictor 和 diffusion 的 attention 中引入 speech prompt 以实现 zero-shot。

## Introduction

1. 提出 NaturalSpeech 2，可以实现 expressive prosody, good robustness，且很强的 zero-shot 合成能力
![](image/Pasted%20image%2020231114170432.png)

首先训练 codec encoder 将语音波形转换为 latent vector 序列，然后用 codec decoder 从 latent vector 中重构波形。

完成训练后，把从训练集中提取的 latent vectors 作为 latent diffusion 模型的 target。而 diffusion 模型基于 phoneme encoder、duration predictor、pitch predictor。

推理时，latent diffusion 模型从 text/phoneme 中生成 latent vector，然后用 decoder 从 vector 中生成语音波形。

本文要点：
+ 使用连续的 vector 而非 离散的 token
+ 使用 diffusion 而非 自回归模型
+ speech prompt 来实现 in-context learning

模型参数为 400M，44K 小时的数据。

## 背景（略）

## NaturalSpeech 2

NaturalSpeech 2 包含：
+ audio codec（encoder + decoder）
+ 带有先验的 diffusion 模型

### 连续向量的 audio codec

连续向量的好处在于：
+ 比离散 token 有着更低的压缩率和更高的比特率，从而确保高质量的重构音频
+ 每个帧都只有一个 vector 而不是多个  token，从而不会增加序列长度

audio codec 包含：
+ encoder，一些卷积层，下采样率为 200
+ RVQ，和 [SoundStream- An End-to-End Neural Audio Codec 笔记](../语音领域其他论文笔记/SoundStream-%20An%20End-to-End%20Neural%20Audio%20Codec%20笔记.md) 一样，输出多个 vector，然后求和作为量化后的 vector
+ decoder：encoder 结构的镜像

![](image/Pasted%20image%2020231126110605.png)

流程如下：
Audio Encoder $:h=f_{\mathrm{enc}}(x)$,

Residual Vector Quantizer $:\{e_j^i\}_{j=1}^R=f_{\mathrm{rvq}}(h^i),\quad z^i=\sum_{j=1}^R e_j^i,\quad z=\{z^i\}_{i=1}^n$,

Audio Decoder $:x=f_{\mathrm{dec}}(z)$

其中，$R$ 为 quantizers 的数量。

实际上，要得到连续的向量，不需要量化器而只需要 VAE 即可。但是为了 regularization and efficiency 考量，采用有很多个 quantizer 的 RVQ 来近似得到连续的向量。好处在于：
+ 训练 LDM 时不需要存储连续的向量，可以节约内存，只需要存 codebook 和 id
+ 预测连续向量时，添加了一个额外的损失（基于这些 id）

### 非自回归的 LDM 模型

采用 diffusion 模型，基于文本序列 $y$ 来预测量化后的 latent vector $z$。

先验模型包含 phoneme encoder、duration predictor 和 pitch predictor，作为 diffusion 模型的 condition $c$。

以 SDE 的方式来看 diffusion 模型，前向 SDE 将 latent vector $z_0$ （即 codec 的输出）转为高斯噪声：
$$\mathrm{d}z_t=-\frac12\beta_tz_t\mathrm{~d}t+\sqrt{\beta_t}\mathrm{~d}w_t,\quad t\in[0,1]$$
其中 $w_t$ 为标准布朗分布。

其解为：
$$z_t=e^{-\frac12\int_0^t\beta_sds}z_0+\int_0^t\sqrt{\beta_s}e^{-\frac12\int_0^t\beta_udu}\mathrm{d}w_s$$
根据伊藤积分的性质，条件分布 $p(z_t|z_0)$ 为高斯分布 $\mathcal{N}(\rho(z_0,t),\Sigma_t)$，其中 $\rho(z_0,t)=e^{-\frac12\int_0^t\beta_sds}z_0,\Sigma_t=I-e^{-\int_0^t\beta_sds}$。

反向 SDE 将高斯噪声转换为数据分布 $z_0$：
$$\mathrm{d}z_t=-(\frac12z_t+\nabla\log p_t(z_t))\beta_t\mathrm{~d}t+\sqrt{\beta_t}\mathrm{~d}\tilde{w}_t,\quad t\in[0,1]$$
考虑 ODE 反向过程：
$$\mathrm{d}z_t=-\frac12(z_t+\nabla\log p_t(z_t))\beta_t\mathrm{~d}t,\quad t\in[0,1]$$
即，训练一个神经网络 $s_{\theta}$ 来估计 score $\nabla\log p_t(z_t)$，从高斯噪声 $z_{1}\sim\mathcal{N}(0,1)$ 中开始，数值求解 SDE 方程或者 ODE 方程。

但是实验发现，预测 $\hat{z}_0$ 效果更好，因此本文的神经网络 $s_\theta(z_t,t,c)$ 基于 WaveNet，预测数据 $\hat{z}_0$ 而非 score。

损失函数为：
$$\mathcal{L}_{\mathrm{diff}}=\mathbb{E}_{z_0,t}[||\hat{z}_0-z_0||_2^2+||\Sigma_t^{-1}(\rho(\hat{z}_0,t)-z_t)-\nabla\log p_t(z_t)||_2^2+\lambda_{ce-rvq}\mathcal{L}_{ce-\mathrm{rvq}}]$$
第一项为 data loss，第二项为 score loss，第三项为一个新的 基于 RVQ的 交叉熵损失。具体来说，对于每个 residual quantizer $j\in[1,R]$，首先计算 residual $\hat{z}_0-\sum_{i=1}^{j-1}e_i$（这里的 $e_i$ 是第 $i$ 个 quantizer 量化后的 GT  embedding），然后计算 residual vector 和 quantizer $j$ 中的 codebook 的每个 embedding 的 L2 距离，然后用 softmax 得到一个概率分布（得到的其实就是一个长为 $K$ 的概率向量，$K$ 为 codebook 的大小）。然后计算他们和 GT embedding ID （其实就是一个 one-hot vector）之间的交叉熵。
> 目的就是训练模型使得对于每个 level 的 quantizer 都能正确预测其 index。

先验模型的 Phoneme Encoder and Duration/Pitch Predictor 都是 Transformer 模块，FFN 层替换为 CNN。

### 用于 in-context learning 的 speech prompting

给定 latent 序列 $z$，随机切一个段 $z^{u{:}\upsilon}$ 作为 prompt（对应下图中的 $z^p$，然后将剩下的两段拼接得到一个新的序列 $z^{\backslash u{:}v}$ 作为 target，如图：
![](image/Pasted%20image%2020231116160031.png)

采用基于 transformer 的 encoder 来处理 $z^p$ 得到 hidden sequence，然后给出两个方案：
+ 对于 duration and pitch predictors，在卷积层中插入一个 QKV attention layer，其中 Q 为卷积层的 hidden sequence，K V 为 prompt encoder 的输出
+ 对于 diffusion 模型，设计两个 attention block：
	+ 第一个 attention 采用 $m$ 个随机初始化的 embedding 作为 query 来 attend prompt 的 hidden sequence，最终得到的结果的长度为 $m$
	+ 第二个采用 WaveNet layer 中的 hidden sequence 作为 query，前面第一个 block 得到的长为 $m$ 的 results 作为 K V
	+ 把第二个 attention block 的输出作为 FiLM 模块的 condition

### 和 NaturalSpeech 的联系

NaturalSpeech 主要关注合成质量，且只关注单说话人的数据集，而 NaturalSpeech 2 关注 speech diversity，用的是大规模的、多说话人的、in-the-wild 数据集。

两者的结构不同，2 主要区别在于：
+ 用了 diffusion 模型来提高建模能力
+ 采用 RVQ 来 regularize latent vectors，以实现重构质量和预测困难程度之间的 trade-off
+ 语音 prompt 机制来实现 zero-shot 的合成

## 实验（略）

用 MLS 44k 小时的数据集（包含文本+语音）。

采用 PyWorld 提取 frame-level 的 pitch。

采用两个 benchmark 数据集来评估：LibriSpeech 和 VCTK。

用 YourTTS 和 VALL-E 作为 baseline。


