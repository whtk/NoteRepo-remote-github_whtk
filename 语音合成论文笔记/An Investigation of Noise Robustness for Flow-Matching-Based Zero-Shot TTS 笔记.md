> Interspeech 2024，Microsoft
<!-- 翻译&理解 -->
<!-- Recently, zero-shot text-to-speech (TTS) systems, capable of
synthesizing any speaker’s voice from a short audio prompt,
have made rapid advancements. However, the quality of
the generated speech significantly deteriorates when the audio
prompt contains noise, and limited research has been conducted
to address this issue. In this paper, we explored various strate-
gies to enhance the quality of audio generated from noisy audio
prompts within the context of flow-matching-based zero-shot
TTS. Our investigation includes comprehensive training strate-
gies: unsupervised pre-training with masked speech denoising,
multi-speaker detection and DNSMOS-based data filtering on
the pre-training data, and fine-tuning with random noise mix-
ing. The results of our experiments demonstrate significant im-
provements in intelligibility, speaker similarity, and overall au-
dio quality compared to the approach of applying speech en-
hancement to the audio prompt. -->
1. 现有的 zero shot TTS 当 prompt 包含噪声时，生成的语音质量会显著下降
2. 本文探索在 基于 flow-matching 的 zero-shot TTS 中增强噪声 prompt 生成的语音质量的策略：
    + 使用 mask speech denoising 进行无监督预训练
    + 在预训练数据上进行 multi-speaker detection 和 DNSMOS-based data filtering
    + 使用 random noise mixing 进行微调
3. 实验结果表明，相比于对 prompt 进行 speech enhancement，本文的方法在可懂性、说话人相似度和整体音频质量上都有显著提升

## Introduction
<!--  Existing zero-shot TTS
models tend to generate speech with a style of noise similar to
that contained in the audio prompt. This property is undesirable
for many applications that require clean speech. In this paper,
we aim to develop a zero-shot TTS system that can generate
high-quality clean speech from any speaker, regardless of the
existence of background noise in the audio prompt. We refer to
this property as the noise robustness of zero-shot TTS -->
1. 现有 zero-shot TTS 生成的语音风格与 prompt 中的噪声相似；本文目的是开发噪声鲁棒的 zero-shot TTS 系统
<!-- While there has been a surge of research interest in zero-
shot TTS technology, research on noise robustness is limited.
The most naive approach involves applying speech enhance-
ment (SE) to the audio prompt before feeding it to a zero-shot
TTS model. While this approach is simple, even the latest SE
models inevitably cause processing artifacts (e.g., [14, 15]),
which result in degraded speech quality from the zero-shot
TTS model. Our preliminary experiment revealed that the ap-
plication of SE causes degradation in both intelligibility and
speaker characteristics of the generated audio. Recently, Fu-
jita et al. [16] proposed enhancing the noise robustness of zero-shot TTS by training a noise-robust speaker embedding extrac-
tor using a self-supervised learning model. While the authors
reported promising results, their method is only applicable to a
zero-shot TTS system based on speaker embeddings. Most re-
cent zero-shot TTS models utilize in-context learning, such as
neural-codec-based language modeling [5, 6, 10] or audio in-
filling [9, 12, 13], instead of representing the audio prompt as a
speaker embedding. It is essential to study the noise robustness
of zero-shot TTS in state-of-the-art model architectures. -->
2. 目前对 zero-shot TTS 的噪声鲁棒性研究：
    + 最简单的方法是在将 prompt 输入 zero-shot TTS 模型之前对其进行 speech enhancement（SE）。但是会导致伪影
    + 通过训练 speaker embedding extractor 来提高 zero-shot TTS 的噪声鲁棒性，但是只适用于基于 speaker embeddings 的 zero-shot TTS 系统
<!-- In the spirit of advancing state-of-the-art technology, this
paper presents our efforts to improve the noise robustness of
flow-matching-based zero-shot TTS [9], one of the leading
models in terms of intelligibility and speaker characteristics
preservation. We explored a range of training strategies, includ-
ing generative pre-training [17] with masked speech denoising,
multi-speaker detection and DNSMOS [18]-based data filtering
on the pre-training data, as well as the fine-tuning with ran-
dom noise mixing. Through experiments with both clean and
noisy audio prompt settings, we demonstrate that intelligibility,
speaker similarity, and overall audio quality can be consistently
improved compared to an approach that applies SE to the audio
prompt. In addition, as a byproduct, we demonstrate for the first
time that our zero-shot TTS model achieves better speaker sim-
ilarity compared to the ground-truth audio in the widely used
cross-utterance evaluation setting on LibriSpeech [19]. -->
3. 本文旨在提高基于 flow-matching 的 zero-shot TTS 的噪声鲁棒性
    + 探索了一系列训练策略，包括 mask speech denoising 的生成式预训练、multi-speaker detection 和 DNSMOS-based data filtering 以及 random noise mixing 的微调
    + 实验结果表明，相比于对 prompt 进行 SE，本文的方法在可懂性、说话人相似度和整体音频质量上都有显著提升
    + 本文 zero-shot TTS 模型在 LibriSpeech 上的跨 utterance 评估中，说话人相似度优于 ground-truth audio

<!-- Flow-matching based zero-shot TTS -->
## 基于 flow-matching 的 zero-shot TTS
<!-- Our TTS system closely follows Voicebox [9], which consists of
a flow-matching-based audio model and a regression-based du-
ration model. This section covers the overview of each model. -->
模型基于 [Voicebox- Text-Guided Multilingual Universal Speech Generation at Scale 笔记](Voicebox-%20Text-Guided%20Multilingual%20Universal%20Speech%20Generation%20at%20Scale%20笔记.md)，包括基于 flow-matching 的 audio 模型和基于回归的 duration 模型。

<!-- The objective of the audio model is to generate a log mel
spectrum˜
x ∈ D×T given a frame-wise phoneme index se-
quence a ∈ T
+ under the condition that the value of˜
x is par-
tially known as xctx ∈ D×T. Here, D represents the feature
dimension, and T is the sequence length. xctx is also known as
the audio context, and the known value of˜
x is filled in; other-
wise, the value is set to zero. In the inference,˜
x is generated
based on xctx and a where a part of xctx is filled by the log mel
spectrum of the audio prompt. Based on the in-context learning
capability of the model, the speaker characteristics of the gener-
ated part of˜
x becomes similar to that of the audio prompt. The
estimated˜
x is then converted to the speech signal based on a
vocoder. -->
audio 模型的目标是在给定 frame-wise phoneme index sequence $a \in \mathbb{Z}^T_+$ 的情况下，生成 log mel spectrum $\tilde{x} \in \mathbb{R}^{D \times T}$，其中 $D$ 表示特征维度，$T$ 表示序列长度。$x_{\text{ctx}} \in \mathbb{R}^{D \times T}$ 是已知的音频上下文，未知的部分填充为 0。在推理时，$\tilde{x}$ 基于 $x_{\text{ctx}}$ 和 $a$ 生成，其中 $x_{\text{ctx}}$ 的一部分由音频 prompt 的 log mel spectrum 填充。模型具有 in-context learning 能力，生成的 $\tilde{x}$ 的说话人特征与音频 prompt 相似。最后 $\tilde{x}$ 通过 vocoder 转为语音。
<!-- The audio model needs to be trained to enable sampling
from P (˜ x|a, xctx). It is achieved based on the flow-matching
framework. This technique morphs a simple initial distribution
p0 into a more complex distribution p1 that closely matches
the observed data distribution. The model is trained based on the conditional flow-matching objective [20]. Specifically,
the model is trained to estimate a time-dependent vector field
vt, t ∈ [0, 1], which is used to construct a flow φt that pushes
the initial distribution towards the target distribution. The sam-
pling process of˜
x is achieved by solving the ordinary differen-
tial equation with the estimated vector field vt and initial ran-
dom value sampled from p0. Refer [20] for more details. -->
audio 模型需要训练以从 $P(\tilde{x} | a, x_{\text{ctx}})$ 中采样。基于 flow-matching 框架实现。模型基于条件 flow-matching 目标进行训练。具体来说，估计时间相关的向量场 $v_t, t \in [0, 1]$ 来构建流 flow $\phi_t$，从初始分布转到目标分布。采样过程通过求解向量场 $v_t$ 和从 $p_0$ 中采样的初始随机值的常微分方程实现。
<!-- The duration model follows the regression-based approach
detailed in [9]. This model takes a phoneme sequence p ∈ N
+,
where N represents the number of phonemes. The model is
˜
trained to predict the duration for each phoneme
l ∈ N
+ under
the condition that the value of˜
l is partially known as lctx ∈ N
+.
Similar to the audio model, lctx is filled by the known value of
˜
l, and the unknown part is filled by zero. The model is trained
based on the mean square error loss on the predicted duration.
Refer [9] for more details. -->
duration 模型输入 phoneme sequence $p \in \mathbb{N}^N_+$，其中 $N$ 表示音素数。模型训练以预测每个音素的持续时间 $\tilde{l} \in \mathbb{N}^N_+$。

<!-- Unsupervised pre-training of audio model -->
### 无监督预训练 audio 模型
<!-- Liu et al. [17] proposed to pre-train the flow-matching-based au-
dio model with a large amount of unlabeled training data. They
reported superior audio model quality after fine-tuning. During
pre-training, the phoneme sequence a is dropped, and the model
is trained to predict the distribution of P (˜ x|xctx). For each train-
ing sample, n non-consecutive random segments are selected
with a constraint of the minimum number of frames, MinF, of
each masked segment. In this work, we set MinF = 5 for all
our exploration based on our preliminary experiment. -->
使用大量无标签训练数据预训练 flow-matching-based audio 模型再微调后可以得到更好的 audio 模型质量。在预训练期间，音素序列 $a$ 被丢弃，模型训练以预测 $P(\tilde{x} | x_{\text{ctx}})$ 的分布。对于每个训练样本，选择 n 个非连续的随机片段，每个 mask 片段的最小帧数为 $Min_F$。本文设置 $Min_F = 5$。

## 噪声鲁棒性训练策略

<!-- Data filtering in pre-training -->
### 预训练下的数据过滤
<!-- We want to utilize a large amount of unlabeled data for pre-
training to further improve the performance of the audio mod-
els. However, real-world data is often low-quality and noisy,
and using such data without proper filtering can negatively im-
pact the model performance. Therefore, to ensure the quality
of our models, we explore data filtering techniques that can ef-
fectively identify and prioritize high-quality, noise-free samples
for pre-training. Consequently, we employ the following two
strategies to filter the pre-training data.
Our first strategy involves filtering out the samples with
more than one speaker. To detect the multiple speakers in an
audio sample, we employ an in-house speaker change detec-
tion model, and discard a sample whenever the speaker change
is detected. Our second strategy involves assessing the speech
quality of the samples. We employ the DNSMOS [18], a neural
network-based mean opinion score estimator1, to evaluate the
speech quality. We then discard samples that fall below a cer-
tain DNSMOS value threshold DNSMOST. In the experiments
section, we explore the impact of different threshold values for
our second strategy. -->
采用以下两种策略来过滤大量的无标签的预训练数据：
+ 过滤掉多说话人的样本。使用 speaker change detection model 检测多说话人，当检测到 speaker change 时丢弃样本
+ 评估样本的语音质量。使用 DNSMOS 评估语音质量，丢弃低于阈值的样本

<!-- Masked speech denoising in pre-training -->
### 预训练下的 mask speech denoising
<!-- Masked speech denoising, introduced in WavLM [21], is an
approach to enhance the model’s ability to focus on relevant
speech signals amid noise. It involves estimating clean audio
for the masked part from the noisy audio input. Inspired by the success of WavLM, we investigate a similar approach for flow-
matching-based model pre-training. -->
本文探索了类似于 WavLM 的 mask speech denoising 方法。
<!-- During pre-training, in a probability of P pre
n , we simu-
late noisy speech by mixing training samples with randomly
selected noise, which yields pairs of noisy speech and clean
speech. We use the noisy speech to extract the context input
xctx, and the original training sample as the training target. In
the noise mixing phase, we randomly sample the noise from
the DNS challenge corpus [22], crop it, and mixed it with the
signal-to-noise ratio (SNR) ranging from 0dB to 20 dB. We
ensure that the duration of the noise does not exceed 50% of
that of the training audio. We also explore the mixing of a sec-
ondary speaker into the audio with a probability P pre
s , drawing
parallels to WavLM. The secondary speaker is picked from the
same training batch of the primary speaker. All the mixing set-
tings are the same as the noise mixing one, except that the SNR
ranges between [0, 10] dB. -->
在预训练期间，以概率 $P^{\text{pre}}_n$，通过将训练样本与随机选噪声混合来模拟噪声语音，得到 带噪语音和干净语音 对。使用 带噪语音提取上下文输入 $x_{\text{ctx}}$，原始训练样本作为 target。在噪声混合阶段，从 DNS challenge corpus 随机采样噪声，混合使其 SNR 在 0dB 到 20 dB 之间。

<!-- Fine-tuning with random noise mixing -->
### 使用随机噪声混合进行 fine tune
<!-- We also explore the fine-tuning strategy of the audio model.
Conventionally, the audio model is fine-tuned with clean train-
ing data [17]. On the other hand, Fujita et al. [16] concurrently2
proposed to fine-tune their zero-shot TTS model by including
noise to the audio prompt in a 50% ratio to improve the noise
robustness. In our work, we also explore the similar approach in
the context of flow-matching-based zero-shot TTS. Specifically,
we randomly add noise in a probability P ft
n to the audio to ex-
tract the audio context xctx, while the training target remains the
original clean audio. Noise samples from the DNS challenge
corpus [22] are randomly selected and mixed at SNRs between
-5 dB and 20 dB. -->
在 fine-tuning 阶段，以概率 $P^{\text{ft}}_n$，随机添加噪声到音频中提取音频上下文 $x_{\text{ctx}}$，训练目标仍然是原始干净音频。噪声样本从 DNS challenge corpus 随机选择，混合 SNR 在 -5 dB 到 20 dB 之间。

## 实验（略）
