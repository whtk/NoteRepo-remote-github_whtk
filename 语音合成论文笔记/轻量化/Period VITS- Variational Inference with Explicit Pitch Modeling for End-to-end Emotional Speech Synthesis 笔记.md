> ICASSP 2023，LINE Corp & NAVER Corp
<!-- 翻译&理解 -->
<!-- Several fully end-to-end text-to-speech (TTS) models have been proposed that have shown better performance compared to cas- cade models (i.e., training acoustic and vocoder models separately). However, they often generate unstable pitch contour with audible artifacts when the dataset contains emotional attributes, i.e., large diversity of pronunciation and prosody. To address this problem, we propose Period VITS, a novel end-to-end TTS model that incor- porates an explicit periodicity generator. In the proposed method, we introduce a frame pitch predictor that predicts prosodic features, such as pitch and voicing flags, from the input text. From these features, the proposed periodicity generator produces a sample-level sinusoidal source that enables the waveform decoder to accurately reproduce the pitch. Finally, the entire model is jointly optimized in an end-to-end manner with variational inference and adversarial objectives. As a result, the decoder becomes capable of generat- ing more stable, expressive, and natural output waveforms. The experimental results showed that the proposed model significantly outperforms baseline models in terms of naturalness, with improved pitch stability in the generated samples. -->
1. 一些全端到端的 TTS 通常会生成不稳定的 pitch contour（当数据集包含情感属性时）
2. 提出 Period VITS，端到端的 TTS 模型，引入了显式的 periodicity generator
    1. 引入 frame pitch predictor，从输入文本预测韵律特征，如 pitch 和 voicing flags
    2. periodicity generator 从这些特征中生成 sample-level sinusoidal source，使得 decoder 可以重构 pitch
    3. 整个模型通过变分推断和对抗训练联合优化

## Introduction
<!-- Text-to-speech (TTS) has recently had a significant impact due to the rapid advancement of deep neural network-based approaches [1]. In most previous studies, TTS models were built as a cascade archi- tecture of two separate models—an acoustic model that generates pre-defined acoustic features (e.g. mel-spectrogram) from text [2, 3] and a vocoder that synthesizes waveform from the acoustic fea- ture [4, 5, 6]. Although these cascade models were able to generate speech reasonably well, they typically suffered from an error deriv- ing from the use of pre-defined features and separated optimization for the two independent models. Sequential training or fine-tuning can mitigate the quality degradation [7], but the training procedure is complicated -->
1. 之前的级连模型通常存在问题，如使用预定义特征、两个独立模型的分开优化
<!-- To address this problem, several works have investigated the use of fully end-to-end architecture that jointly optimizes the acoustic and vocoding models1 [8, 9, 10, 11]. One of the most success- ful works is VITS [10], which adopts a variational autoencoder (VAE) [12] with the augmented prior distribution by normalizing flows [13]. The VAE is used to acquire the trainable latent acoustic features from waveforms, whereas the normalizing flows are used to make the hidden text representation as powerful as the latent features. -->
<!-- However, we found that although VITS generates natural- sounding speech when trained with a reading style dataset, its performance is limited when applied to more challenging tasks, such as emotional speech synthesis, where the dataset has signifi- cant diversity in terms of pronunciation and prosody. Specifically, the model generates less intelligible voices with unstable pitch con- tour. Although the intelligibility problem could be addressed by expanding the phoneme-level parameters of prior distribution to frame-level parameters [14], it is still a challenge to generate ac- curate pitch information due to the architectural limitation of the non-autoregressive vocoders [15]. -->
2. 基于 GAN 和 VAE 的全端到端模型，如 VITS，可以生成自然的语音，但是在情感语音合成等更具挑战性的任务上表现有限
<!-- To tackle this, we propose Period VITS, a novel TTS system that explicitly provides sample-level and pitch-dependent periodic- ity when generating the target waveform. In particular, the proposed model consists of two main modules termed the prior encoder and the waveform decoder (hereinafter simply called “decoder”). On the prior encoder side, we employ a frame prior network with a frame pitch predictor that can simultaneously generate the parameters of the prior distribution and prosodic features in every frame. Note that the parameters are used to learn expressive prior distribution with normalizing flows and the prosodic features such as the pitch and the voicing flags are used to produce the sample-level sinusoidal source signal. On the decoder side, this periodic source is fed to ev- ery up-sampled representation in the HiFi-GAN-based vocoder [6], to guarantee pitch stability in the target waveform. Note that the training process optimizes the entire model in the end-to-end scheme from the variational inference point of view. -->
3. 提出 Period VITS，可以在生成波形时显式地提供 sample-level 和 pitch-dependent 的 periodicity
    1. 包含两个主要模块：prior encoder 和 waveform decoder
    2. prior encoder 包含 frame prior network 和 frame pitch predictor，可以同时生成先验分布的参数和每帧的 prosodic feature；且主要通过 normalizing flows 来学习复杂的先验分布和 prosodic feature（如 pitch 和 voicing flag），从而得到正弦的 source signal
    3. decoder 端，periodic source 送入到 HiFiGAN vocoder 中的每个上采样表征中，来确保波形的韵律稳定性
    4. 训练的时候以变分的方式端到端优化
<!-- Several works have addressed similar problems by focusing on the periodicity of speech signals in the vocoder context [16, 17, 18]. The proposed model differs in that the conventional methods have used pre-defined acoustic features and only optimized the vocoder part separately. In contrast, the proposed architecture has the benefit of end-to-end training and obtains optimal latent acoustic features guided by the auxiliary pitch information. In addition, another prior work has tackled the pitch stability problem by adopting a chunked autoregressive architecture in the vocoder [15]. Unlike that method, our proposal can generate waveforms in much faster speed, thanks to the fully non-autoregressive model architecture. -->
4. 之前已经有些方法关注语音信号中的 periodicity，但是这些方法通常使用预定义的 acoustic features，并且只优化 vocoder 部分，而本文的模型可以端到端训练，通过辅助的 pitch 信息来获得最优的 latent acoustic features；且采用非自回归模型，生成速度更快
<!-- The experimental results show that the proposed model per- formed significantly better than all the baseline models includ- ing end-to-end and cascade models in terms of naturalness in the multi-speaker emotional TTS task. Moreover, the proposed model achieved comparable scores to the recordings for neutral and sad style with no statistically significant difference. -->
5. 实验结果表明，提出的模型在多说话人情感 TTS 的表现优于所有 baseline，且在中性和悲伤风格上的 score 与录音相当

## 方法

### 概览
<!-- The overall model architecture is shown in Fig. 1. Inspired by VITS [10], we adopt a VAE whose prior distribution is conditioned on text. The VAE is composed of a posterior encoder and a decoder, whereas the prior distribution is modeled by a prior encoder. -->
整体框架如图：
![](image/Pasted%20image%2020240406192937.png)

模型基于 [VITS- Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech 笔记](../VITS-%20Conditional%20Variational%20Autoencoder%20with%20Adversarial%20Learning%20for%20End-to-End%20Text-to-Speech%20笔记.md)，包含 posterior encoder 和 decoder，先验分布由 prior encoder 建模。
<!-- Specifically, the posterior encoder converts the input linear spec- trogram to latent acoustic features, while the prior encoder trans- forms the text into the latent features. The decoder is responsible for reconstructing waveforms from the learned latent features. -->
posterior encoder 将输入的 linear spectrogram 转换为 latent acoustic features，prior encoder 将文本转换为 latent features，decoder 从学习到的 latent features 重构波形。
<!-- In addition, we introduce a latent variable y to represent the prosodic feature as a separated source from the VAE latent variable z to explicitly model the pitch information of the generated speech. The proposed model is trained to maximize the log-likelihood of waveform x given text c. However, as it is intractable, we optimize a lower bound on the marginal likelihood, as follows [12]: -->
引入 latent y 表示 prosodic feature，作为 VAE latent variable z 的一个独立的 source，来显式地建模生成语音的 pitch。采用最大化给定文本后波形的 log-likelihood 来训练，简化为优化似然下界：
$$\begin{aligned}
&\log p(x|c)=\log\int\int p(x,z,y|c)dzdy \\
&\geq\int\int q(z,y|x)\log\frac{p(x,z,y|c)}{q(z,y|x)}dzdy \\
&=\int\int q(z|x)q(y|x)\log\frac{p(x|z,y)p(z|c)p(y|c)}{q(z|x)q(y|x)}dzdy \\
&=E_{q(z,y|x)}[\log p(x|z,y)]-D_{KL}(q(z|x)||p(z|c)) \\
&-D_{KL}(q(y|x)||p(y|c)),
\end{aligned}$$
<!-- proximate posterior distribution for z and y, E denotes the expectation operator, and DKL represents the Kullback–Leibler (KL) diver- gence. In addition, we assume z and y are conditionally independent given x2. Therefore, q(z, y|x) can be factorized as q(z|x)q(y|x).
  -->
其中 $p$ 为生成模型的分布，$q$ 为近似后验分布，$E$ 为期望计算，$D_{KL}$ 为 KL 散度。假设 z 和 y 在给定 x 的条件下是独立的，因此 $q(z,y|x)$ 可以分解为 $q(z|x)q(y|x)$。
<!-- Furthermore, as pitch extraction from x is a deterministic oper- ation, we can define q(y|x) = δ(y − ygt) and transform the third term of (1) as follows:
 -->
由于从 x 提取 pitch 是确定性操作，可以定义 $q(y|x)=\delta(y-y_{gt})$，将公式的第三项转换为：
$$-\log p(y_{gt}|c)+const.$$
<!-- where ygt represents the observed ground truth pitch value. We can optimize this part by minimizing the L2 norm between predicted and ground truth by assuming a Gaussian distribution for p(y|c) with fixed unit variance. The three terms in (1) can then be inter- preted as wave reconstruction loss of VAE Lrecon, KL divergence loss between prior/posterior distributions Lkl, and pitch reconstruc- tion loss from text Lpitch, respectively. Following [10], we adopt mel-spectrogram loss for Lrecon.
 -->
其中 $y_{gt}$ 为 GT pitch 值。可以通过最小化预测值和真实值之间的 L2 norm 来优化，假设 $p(y|c)$ 为固定单位方差的高斯分布。公式中的三项可以解释为 VAE 的波形重构损失 $L_{recon}$、先验/后验分布的 KL 散度损失 $L_{kl}$ 和从文本重构 pitch 的损失 $L_{pitch}$。采用 mel-spectrogram loss 作为 $L_{recon}$。
<!-- As the proposed method not only focuses on reading-style TTS, but also TTS with an emotional dataset with significant diversity in terms of pronunciation, the prior distribution modeled by the prior encoder needs to represent the rich acoustic variation of pronunci- ation within the same phoneme. To this end, we adopt the frame prior network proposed in [14]. It expands phoneme-level prior dis- tribution to frame-level fine-grained distribution. We confirmed in preliminary experiments that this was effective to stabilize the pro- nunciation not only for singing voice synthesis but also for multi- speaker emotional TTS. In addition, we introduce a frame pitch pre- dictor from a hidden layer of the frame prior network to predict the frame-level prosodic features, i.e., fundamental frequency (F0) and voicing flag (v), which are subsequently used as inputs for the pe- riodicity generator described in Section 2.3. As discussed in Sec- tion 2.1, these features are optimized using the L2 norm, as follows3:
 -->
prior encoder 建模的先验分布需要表示相同音素包含的丰富的发音变化。于是采用 [VISinger- Variational Inference with Adversarial Learning for End-to-End Singing Voice Synthesis](../歌声合成/VISinger-%20Variational%20Inference%20with%20Adversarial%20Learning%20for%20End-to-End%20Singing%20Voice%20Synthesis.md) 中提出的 frame prior network，将 phoneme-level 先验分布扩展到 frame-level。
> 实验中证实可以稳定发音

然后引入 frame pitch predictor，从 frame prior network 的 hidden layer 预测 frame-level 的韵律特征（基频（F0）和 voicing flag（v）），作为后续周期性生成器的输入。也 通过L2 norm 进行优化：
$$L_{pitch}=\|\log F_0-\log\hat{F}_0\|_2+\|v-\hat{v}\|_2.$$
<!-- The prior distribution is augmented by normalizing flow f to en- hance the modeling capability, as in VITS: -->
先验分布通过 normalizing flow $f$ 来增强建模能力：
$$p(z|c)=N(f(z);\mu(c),\sigma(c))\left|\det\frac{\partial f(z)}{\partial z}\right|,$$
<!-- where μ(c) and σ(c) represent trainable mean and variance parame- ters calculated from text representation, respectively.
 -->
其中 $\mu(c)$ 和 $\sigma(c)$ 表示从文本表征中得到的可训练的均值和方差。

### 带有 periodicity generator 的 decoder
<!-- It has been reported that GAN-based vocoder models typically pro- duce artifacts when reconstructing waveforms from acoustic features due to their inability to estimate pitch and periodicity [15]. We found that these artifacts are also observed in end-to-end TTS mod- els, particularly when trained on a dataset with large pitch variance, such as an emotional one. To address this problem, we use a sine- based source signal to explicitly model the periodic component of speech waveforms, which has proven to be effective in some previ- ous works [16, 17, 19]. However, it is not straightforward to incorpo- rate it into the HiFi-GAN-based decoder (i.e., vocoder) architecture in VITS, as a sine-based source signal is supposed to be a sample- level feature, while the input of HiFi-GAN is typically a frame-level acoustic feature4.-->
已经有研究表明，GAN-based vocoder 模型在从 acoustic features 重构波形时通常会产生 artifacts，因为无法估计 pitch 和 periodicity。作者发现这些 artifacts 在端到端 TTS 模型中也有，尤其在训练集包含大 pitch 方差的情感数据集上。于是用 sine-based source signal 显式地建模语音波形的周期成分，但是不能直接将其整合到 VITS 中的 HiFi-GAN-based decoder（即 vocoder）中，因为 sine-based source signal 应该是 sample-level 特征，而 HiFi-GAN 的输入通常是 frame-level acoustic feature。
<!-- To overcome this mismatch, we devise a model architecture in- spired by a pitch-controllable HiFi-GAN-based vocoder [20]. Fig- ure 2 shows the decoder architecture of the proposed model. The key idea is to successively apply down-sampling layers to the sample- level pitch-related input to match the resolution of the up-sampled frame-level feature. We call the module to generate a sample-level periodic source as periodicity generator. We input the sinusoidal source together with the voicing flags and Gaussian noise, as this setting performed better in a previous work [17]. In addition, unlike the previous work in [20], we avoid directly adding the sample-level output of the pre-conv module in Fig. 2 to the up-sampled features, as we found this degrades the sample quality. -->
受 Period-HiFi-GAN 的启发，如图：
![](image/Pasted%20image%2020240406195450.png)

关键在于，将 下采样层用于 sample-level 的 pitch 相关的输入来匹配上采样的 frame-level 的分辨率。

用 periodicity generator 生成 sample-level 的 periodic source（将 sinusoidal source、voiceing flag 和高斯噪声一起作为输入）。但是并没有直接将 pre-cov 得到的 sample-level 的输出直接加到上采样层中，因为发现会降低性能。
