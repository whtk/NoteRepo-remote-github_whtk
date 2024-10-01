> preprint 2024.09，CUHK、网易伏羲，李海洲
<!-- 翻译 & 理解 -->
<!-- This paper introduces Easy One-Step Text-to-Speech (E1 TTS), an efficient non-autoregressive zero-shot text-to-speech system based on denoising diffusion pretraining and distribution matching distillation. The training of E1 TTS is straightforward; it does not require explicit monotonic alignment between the text and audio pairs. The inference of E1 TTS is efficient, requiring only one neural network evaluation for each utterance. Despite its sampling efficiency, E1 TTS achieves naturalness and speaker similarity comparable to various strong baseline models. Audio samples are available at e1tts.github.io. -->
1. 提出 Easy One-Step TTS（E1 TTS），非自回归 zero-shot TTS，基于 denoising diffusion 预训练和 distribution matching distillation
2. 训练简单，不需要文本和音频对之间的显式单调对齐
3. 推理高效，每个 utterance 只要一次 evaluation

## Introduction
<!-- Non-autoregressive (NAR) text-to-speech (TTS) models [1] gener- ate speech from text in parallel, synthesizing all speech units simul- taneously. This enables faster inference compared to autoregressive (AR) models, which generate speech one unit at a time. Most NAR TTS models incorporate duration predictors in their architecture and rely on alignment supervision [2]–[4]. Monotonic alignments between input text and corresponding speech provide information about the number of speech units associated with each text unit, guiding the model during training. During inference, learned duration predictors estimate speech timing for each text unit. -->
1. 大部分 NAR TTS 都包含 duration predictor，并依赖于 alignment supervision
<!-- Several pioneering studies [5], [6] have proposed implicit-duration non-autoregressive (ID-NAR) TTS models that eliminate the need for alignment supervision or explicit duration prediction. These models learn to align text and speech units in an end-to-end fashion using attention mechanisms, implicitly generating text-to-speech alignment. -->
2. 一些研究提出了 ID-NAR TTS 模型，不需要 alignment supervision 或 explicit duration prediction，采用 attention 机制端到端学习对齐
<!-- Recently, several diffusion-based [7] ID-NAR TTS models [8]–[14] have been proposed, demonstrating state-of-the-art naturalness and speaker similarity in zero-shot text-to-speech [15]. However, these models still require an iterative sampling procedure taking dozens of network evaluations to reach high synthesis quality. Diffusion distillation techniques [16] can be employed to reduce the number of network evaluations in sampling from diffusion models. Most distillation techniques are based on approximating the ODE sampling trajectories of the teacher model. For example, ProDiff [17] applied Progressive Distillation [18], CoMoSpeech [19] and FlashSpeech [20] applied Consistency Distillation [21], and VoiceFlow [22] and ReFlow-TTS [23] applied Rectified Flow [24]. Recently, a different family of distillation methods was discovered [25], [26], which directly approximates and minimizes various divergences between the generator’s sample distribution and the data distribution. Compared to ODE trajectory-based methods, the student model can match or even outperform the diffusion teacher model [26], as the distilled one- step generator does not suffer from error accumulation in diffusion sampling. -->
3. 基于 diffusion 的 ID-NAR TTS 可以实现 zero-shot TTS，在自然度和说话人相似度上达到 SOTA，但是需要多次的 evaluation，diffusion distillation 可以减少次数，且大多数 distillation 基于 ODE sampling trajectories
<!-- In this work, we distill a diffusion-based ID-NAR TTS model into a one-step generator with recently proposed distribution matching distillation [25], [26] method. The distilled model demonstrates better robustness after distillation, and it achieves comparable performance to several strong AR and NAR baseline systems. -->
4. 本文将基于 diffusion 的 ID-NAR TTS 模型 distill 成 one-step generator，采用 distribution matching distillation，实现和 AR/NAR baseline 系统相当的性能

## 背景

### Distribution Matching Distillation
<!-- Consider a data distribution p(x) on Rd. We can convolve the density p(x) with a Gaussian perturbation kernel qt(xt|x) = N (xt; αtx, σt2Id) to obtain the perturbed density pt(xt) := ∫p(x)qt (xt |x)dx, where αt, σt >0 ratio at each time t ∈ [0,1]. Various formulations of diffusion models exist in the literature [7], [24], most of which are equivalent to learning a neural network that approximates the score function sp(xt, t) := ∇xt log pt(xt) at each time t. -->
考虑位于 $\mathbb{R}^d$ 的数据分布 $p(x)$，将 $p(x)$ 与 Gaussian perturbation kernel $q_t(x_t|x) = N(x_t;\alpha_tx, \sigma_t^2I_d)$ 卷积，得到 perturbed density $p_t(x_t) := \int p(x)q_t(x_t|x)dx$，其中 $\alpha_t, \sigma_t >0$ 是每个时间 $t \in [0,1]$ 的 SNR。大部分 diffusion 模型等价于学习一个神经网络，逼近每个时间 $t$ 的 score function $s_p(x_t, t) := \nabla_{x_t} \log p_t(x_t)$。
<!-- Now, consider a generator function gθ(z) : Rd → Rd that takes in random noise Z ∼ N(0,Id) and outputs fake samples Xb := gθ(Z) with distribution qθ(x). Several studies [25], [27] have discovered that if we can obtain the two score functions sp (x) := ∇x log p(x) and sq(x) := ∇x logqθ(x), we can compute the gradient of the following KL divergence: -->
考虑 generator 函数 $g_{\theta}(z) : \mathbb{R}^d \rightarrow \mathbb{R}^d$，输入随机噪声 $Z \sim \mathcal{N}(0, I_d)$，输出 fake samples $\widehat{X} := g_{\theta}(Z)$，分布为 $q_{\theta}(x)$。如果可以得到两个 score functions $s_p(x) := \nabla_x \log p(x)$ 和 $s_q(x) := \nabla_x \log q_{\theta}(x)$，计算以下 KL 散度的梯度：
$$\nabla_\theta D_{\mathrm{KL}}\left(q_\theta(x)\|p(x)\right)=E\left[\left(s_q(\widehat{X})-s_p(\widehat{X})\right)\frac{\partial g_\theta(Z)}{\partial\theta}\right].$$
<!-- However, obtaining sp(x) and sq(x) directly is challenging. Instead, we can train diffusion models to estimate the score functions sp (xt , t) and sq(xt,t) of the perturbed distributions pt(xt) and qθ,t(xt) := R qθ (x)qt (xt |x)dx. Consider the following weighted average of KL divergence at all noise scales [25], [27]: -->
但是直接得到 $s_p(x)$ 和 $s_q(x)$ 很困难。可以训练 diffusion 模型估计 perturbed distributions $p_t(x_t)$ 和 $q_{\theta, t}(x_t) := R q_{\theta}(x)q_t(x_t|x)dx$ 的 score functions $s_p(x_t, t)$ 和 $s_{q, t}(x_t, t)$。考虑所有 noise scales 的 KL 散度的加权平均：
$$D_\theta:=E_{t\thicksim p(t)}\left[w_tD_{\mathrm{KL}}\left(q_{\theta,t}(x_t)\|p_t(x_t)\right)\right],$$
<!-- where wt ≥0 is a time-dependent weighting factor, and p(t) is the
distribution of time. Let W ∼N(0,Id) be an independent Gaussian
noise, and define Xt := αtX + σtW. Then, the gradient of the
weighted KL divergence can be computed as: -->
其中 $w_t \geq 0$ 是时间相关的权重因子，$p(t)$ 是时间分布。$W \sim \mathcal{N}(0, I_d)$ 是独立的高斯噪声，定义 $\widehat{X}_t\::=\:\alpha_t\widehat{X}\:+\:\sigma_tW$。加权 KL 散度的梯度可以计算为：
$$\nabla_\theta D_\theta=E_{t\thicksim p(t)}\left[w_t\alpha_t\left(s_q(\widehat{X}_t,t)-s_p(\widehat{X}_t,t)\right)\frac{\partial g_\theta(Z)}{\partial\theta}\right].$$
<!-- Given a pretrained score estimator sϕ(xt,t) ≈ sp(xt,t), the
procedure to distill it into a single-step generator gθ is described
in Algorithm 1. -->
给定预训练的 score estimator $s_{\phi}(x_t, t) \approx s_p(x_t, t)$，将其 distill 成 one-step generator $g_{\theta}$ 的过程如下：
![](image/Pasted%20image%2020241001114131.png)

<!-- Although the generator gθ can be randomly initialized in the-
ory, initializing gθ with sϕ leads to faster convergence and better
performance [25]. Several studies [28], [29] have discovered that
pretrained diffusion models already possess latent one-step generation
capabilities. Moreover, it is possible to convert them into one-step
generators by tuning only a fraction of the parameters [28], [29],
such as the normalization layers. -->
虽然 generator $g_{\theta}$ 理论上可以随机初始化，但是用 $s_{\phi}$ 初始化会更快收敛，性能更好。已有研究发现，预训练的 diffusion 模型已经具有 latent one-step generation 能力，可以通过调整部分参数（如 normalization layers）将其转换为 one-step generator。
<!-- While distribution matching distillation resembles generative ad-
versarial networks (GANs) [30] in its requirement for alternating
optimization, it has been empirically observed [26] to be significantly
more stable, requiring minimal tuning and avoiding the mode collapse
issue that often hinders GAN training. -->
distribution matching distillation 与 GAN 相似，需要交替优化，但是实验证明更稳定，避免了 GAN 训练中常见的 mode collapse 问题。

<!-- Rectified Flow -->
### Rectified Flow
<!-- Rectified Flow [24] is capable of constructing a neural ordinary
differential equation (ODE): -->
Rectified Flow 构建 ODE：
$$\mathrm{d}Y_t=v(Y_t,t)\mathrm{d}t,\quad t\in[0,1],$$
<!-- that maps between two random distributions X0 ∼π0 and X1 ∼π1,
by solving the following optimization problem: -->
实现两个随机分布 $X_0 \sim \pi_0$ 和 $X_1 \sim \pi_1$ 的映射，通过优化：
$$v(x_t,t):=\underset{v}{\operatorname*{\arg\min}}E\|v\left(\alpha_tX_1+\sigma_tX_0,t\right)-(X_0-X_1)\|_2^2,$$
<!-- where αt = t and σt = (1−t). In the special case where X0 ∼
N(0,Id) and X0 ⊥X1, the drift v(xt,t) is a linear combination of the score function s(xt,t) = ∇xt log pt(xt) and xt, where Xt :=
αtX1 + σtX0: -->
其中 $\alpha_t = t$，$\sigma_t = (1-t)$。特殊情况下，$X_0 \sim \mathcal{N}(0, I_d)$ 且 $X_0 \perp X_1$，drift $v(x_t, t)$ 是 score function $s(x_t, t) = \nabla_{x_t} \log p_t(x_t)$ 和 $x_t$ 的线性组合，其中 $X_t := \alpha_tX_1 + \sigma_tX_0$：
$$s(x_t,t)=-\frac{1-t}{t}v(x_t,t)-\frac{1}{t}x_t.$$
<!-- In the experiments, we trained all our diffusion models with the
Rectified Flow loss in Equation 5. Equation 6 allows us to apply
DMD to Rectified Flow models. -->
实验中，所有的 diffusion 模型使用 Rectified Flow loss。

## E1 TTS
<!-- E1 TTS is a cascaded conditional generative model, taking the
full text and partially masked speech as input, and outputs com-
pleted speech. The overall architecture is illustrated in Figure 2. E1
TTS is similar to the acoustic model introduced in [31] with the
modification that all speech tokens are generated simultaneously in
the first stage. Further more, we applied DMD to convert the two
diffusion transformers (DiTs) [32] to one-step generators, removing
all iterative sampling from the inference pipeline. We will describe
the components in the system in the following sections. -->
E1 TTS 是一个级联条件生成模型，输入完整文本和部分 mask 的语音，输出完整语音。整体架构如下图：
![](image/Pasted%20image%2020241001115221.png)

模型类似于[Autoregressive Diffusion Transformer forText-to-Speech Synthesis 笔记](Autoregressive%20Diffusion%20Transformer%20forText-to-Speech%20Synthesis%20笔记.md)，
但是所有 speech tokens 在第一阶段同时生成。采用 DMD 将两个 DiTs 转换为 one-step generators。

### mel 谱 Autoencoder
<!-- Directly training generative models on low-level speech represen-
tations such as Mel spectrograms [11] and raw waveforms [8] is
resource-consuming due to the long sequence lengths. We build a Mel
spectrogram autoencoder with a Transformer encoder and a Diffusion
Transformer decoder. The encoder takes log Mel spectrograms and
outputs continuous tokens in R32 at a rate of approximately 24Hz.
The decoder is a Rectified Flow model that takes speech tokens as
input and outputs Mel spectrograms. The encoder and decoder are
jointly trained with a diffusion loss and a KL loss to balance rate
and distortion. The Mel spectrogram autoencoder is fine-tuned for
the case where part of the spectrogram is known during synthesis to
enhance its performance in speech inpainting. For the decoder, we
appended layers of 2D convolutions after the transformer blocks to
improve its performance on spectrograms. Please refer to [31] for
further details regarding the training process and model architecture. -->
构建 mel 谱 autoencoder，包含 Transformer encoder 和 Diffusion Transformer decoder。Encoder 输入 log Mel spectrograms，输出 $\mathbb{R}^{32}$ 的 continuous tokens，大约 24Hz。Decoder 是 Rectified Flow 模型，输入 speech tokens，输出 Mel spectrograms。Encoder 和 decoder 一起训练，使用 diffusion loss 和 KL loss 平衡 rate 和 distortion。Mel spectrogram autoencoder 在合成时当部分 spectrogram 已知的情况下进行 fine-tuning，以增强 speech inpainting 性能。Decoder 在 transformer blocks 后添加 2D convolutions 层，提高在 spectrograms 上的性能。

<!-- Text-to-Token Diffusion Transformer -->
### Text-to-Token Diffusion Transformer
<!-- The Text-to-Token DiT is trained to estimate the masked part of
input speech tokens given the full text. During training, the sequence
of speech tokens is randomly split into three parts: the prefix part, the
masked middle part, and the suffix part. We first sample the length of
the middle part uniformly, and then we sample the beginning position
of the middle part uniformly. With 10% probability we mask the
entire speech token sequence -->
Text-to-Token DiT 给定完整文本，估计 mask 部分的 speech tokens。训练时，speech tokens 随机分为三部分：prefix、masked middle 和 suffix。首先均匀采样 middle 部分的长度，然后均匀采样 middle 部分的开始位置。10% 的概率 mask 整个 speech token 序列。
<!-- We adopted rotary positional embedding (RoPE) [33] in all Trans-
former blocks in E1 TTS. For the Text-to-Token model, we designed the positional embedding to promote diagonal alignment between text
and speech tokens, as illustrated in Figure 4. With RoPE, each token
is associated with a position index, and the embeddings corresponding
to the tokens are rotated by an angle proportional to their position
index. For text tokens, we assign them increasing integer position
indices. For speech tokens, we assign them fractional position indices,
with an increment of ntext/nspeech. This design results in an initial
attention pattern in the form of a diagonal line between text and
speech. Similar designs have proven effective in other ID-NAR TTS
models [5], [34]. -->
在所有 Transformer blocks 中采用 RoPE。对于文本 token，为递增的整数位置索引。对于 speech token，为分数位置索引，增量为 $\frac{n_{\text{text}}}{n_{\text{speech}}}$。
> 这种设计类似于 attention，在其他 ID-NAR TTS 模型中也被证明有效。

<!-- Duration Modeling -->
### Duration 建模
<!-- Similar to most ID-NAR TTS models, E1 TTS requires the total
duration of the speech to be provided during inference. We trained
a duration predictor similar to the one in [35]. The rough alignment
between text and speech tokens is first obtained by training an aligner
based on RAD-TTS [36]. Then a regression-based duration model is
trained to estimate partially masked durations. The duration model
takes the full text (phoneme sequence in our case) and partially
observed durations as input, then predicts unknown durations based
on the context. We observed that minimizing the L1 difference in total
duration [5], [6] works better than directly minimizing phoneme-level
durations, resulting in a lower total duration error. -->
训练类似于 [Voicebox- Text-Guided Multilingual Universal Speech Generation at Scale 笔记](Voicebox-%20Text-Guided%20Multilingual%20Universal%20Speech%20Generation%20at%20Scale%20笔记.md) 的 duration predictor。首先通过基于 RAD-TTS 的 aligner 获得 text 和 speech token 的粗略对齐，然后训练 duration model 估计部分 mask 的 durations。duration model 输入完整文本（phoneme sequence），部分 mask 的 durations，预测未知 durations。
> 最小化总 duration 的 L1 loss 比直接最小化 phoneme-level durations 效果更好。

### 推理
