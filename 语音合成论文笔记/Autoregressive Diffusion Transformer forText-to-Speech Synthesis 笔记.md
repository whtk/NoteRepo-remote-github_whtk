> preprint 2024,07，CUHK，李海洲
<!-- 翻译 & 理解 -->
<!-- Audio language models have recently emerged as a promising approach for var-
ious audio generation tasks, relying on audio tokenizers to encode waveforms
into sequences of discrete symbols. Audio tokenization often poses a necessary
compromise between code bitrate and reconstruction accuracy. When dealing with
low-bitrate audio codes, language models are constrained to process only a subset of
the information embedded in the audio, which in turn restricts their generative capa-
bilities. To circumvent these issues, we propose encoding audio as vector sequences
in continuous space Rd and autoregressively generating these sequences using a
decoder-only diffusion transformer (ARDiT). Our findings indicate that ARDiT ex-
cels in zero-shot text-to-speech and exhibits performance that compares to or even
surpasses that of state-of-the-art models. High-bitrate continuous speech represen-
tation enables almost flawless reconstruction, allowing our model to achieve nearly
perfect speech editing. Our experiments reveal that employing Integral Kullback-
Leibler (IKL) divergence for distillation at each autoregressive step significantly
boosts the perceived quality of the samples. Simultaneously, it condenses the
iterative sampling process of the diffusion model into a single step. Furthermore,
ARDiT can be trained to predict several continuous vectors in one step, signif-
icantly reducing latency during sampling. Impressively, one of our models can
generate 170 ms of 24 kHz speech per evaluation step with minimal degradation in
performance. Audio samples are available at ardit-tts.github.io. -->
1. Audio tokenization 通常需要在 code bitrate 和 reconstruction accuracy 之间权衡
2. 提出在连续空间 $\mathbb{R}^d$ 中编码音频，并使用 decoder-only diffusion transformer（ARDiT）自回归生成这些序列：
    1. ARDiT 在 zero-shot TTS 上性能甚至超过 SOTA
    2. 高码率连续表征可以实现几乎无损重构
    3. 采用 IKL 散度在每个自回归 step 上进行蒸馏可以提高质量，将 diffusion 迭代采样过程压缩为一步
    4. 模型可以生成 170 ms 的 24 kHz 语音

## Introduction
<!-- Autoregressive modeling of discrete audio tokens has recently achieved significant success across various audio generation tasks [1–15]. These models, often referred to as audio language models, compress audio waveforms into discrete tokens, predict them autoregressively, and then decode them back into waveforms. By discretizing audio signals into discrete tokens [16–22], a unified representation of audio and text is achieved, enabling seamless joint processing with language models. -->
<!-- Despite its success, discrete audio tokenization faces challenges. The theory of lossy compression
[23, 24] suggests a trade-off between the bitrate and reconstruction quality. Current state-of-the-
art neural audio codecs typically require a minimum of 1.5 kbps for high-fidelity reconstruction
of 16 kHz audio [1]. Using a codebook of size 1024, one second of audio would necessitate
1500/log2(1024) = 150 tokens, leading to long sequences that complicate audio language modeling.
Different strategies have been proposed to mitigate this, each with its own limitations: (i) A prevalent
approach assumes conditional independence among tokens [4, 5, 7], concurrently generating multiple
tokens to reduce autoregressive sampling steps. However, the effectiveness of this assumption depends on the quantization method employed. For example, Delayed Pattern [7, 25] works well
with residual vector quantization (RVQ) [17], but it might not work well with other techniques such
as finite scalar quantization (FSQ) [26]. An inappropriate independence assumption can negatively
impact generation performance [7]. (ii) Another approach involves limiting the bitrate and shortening
the token sequence length by encoding only a fraction of the total information in audios [2, 3, 8, 12–
15, 27]. This requires sophisticated disentangled representation learning to achieve a compressed,
task-specific representation for the audio language model. And it limits the model to generating only
partial audio information, hindering its performance on tasks that require high output bitrate. The
information gap must be filled by a cascade of generative models, complicating the audio generation
system. -->
<!-- Besides the trade-off between bitrate and reconstruction quality, audio tokenization also faces
challenges with gradient-based optimization in discrete distributions. While VAEs [28] and VAE-
GANs [29] with continuous latent variables can be trained using standard gradient optimizers via the
reparameterization trick [28], this is not the case for VQ-GAN models [30], which form the basis of
modern neural audio codecs. Effectively training VQ-GANs necessitates techniques like auxiliary
losses and codebook re-initialization [31–35]. -->
1. 离散 audio tokenization 有一些问题：
    1. 需要在 bitrate 和重构质量之间权衡
    2. 离散分布下的 gradient-based optimization，现在通常需要使用辅助 loss 和 codebook re-initialization
<!-- The complexities of audio tokenization can be avoided by representing speech as sequences of
vectors in Rd, termed as continuous tokens [36]. For a continuous token sequence [x1,···,xN],
several methods have been explored to model its conditional density p(xn|x<n): (i) One approach
is to apply flows based on finite compositions [37–41] for pθ(xn|x<n). This model constrains
the network architecture to ensure efficient computation of the Jacobian determinant and thereby
guarantee invertibility. (ii) An alternative is to represent pθ(xn|x<n) using a Mixture Density Network
(MDN) [42] that predicts parameters for continuous mixture distributions [43–49], such as a mixture
of Gaussians (MoG). However, due to the limited expressive power of mixture densities, p(xn|x<n)
needs to be simple enough to be accurately approximated. (iii) Efforts to model pθ(xn|x<n) with
generative adversarial networks (GANs) [50] have been made [51], but they often suffer from training
instability and mode dropping issues. (iv) Diffusion probabilistic models (DPMs) [52–54] are
proficient at modeling continuous densities. Their integration with autoregressive models could yield
impressive results [55–63]. However, DPMs require iterative processes for generating high-quality
samples. Combining diffusion sampling with autoregressive sequence sampling can result in a slow,
high-latency sampling process. -->
2. 可以通过将 speech 表示为 $\mathbb{R}^d$ 中的向量序列（continuous tokens）来避免这些问题，一些方法如下：
    1. 使用 finite compositions 的 flows 来建模 $p_{\theta}(x_n|x_{<n})$
    2. 使用 Mixture Density Network（MDN）来建模 $p_{\theta}(x_n|x_{<n})$
    3. 使用 GAN 来建模 $p_{\theta}(x_n|x_{<n})$
    4. 使用 Diffusion probabilistic models（DPMs）来建模 continuous densities
<!-- Recent discoveries in the distillation of diffusion models [64–68] have changed the situation. A family
of diffusion distillation methods demonstrates that we can effectively transform diffusion models into
single-step implicit generative models while preserving or even improving their generative modeling
performance. SSD-LM [55, 56] developed a method of integrating autoregressive sequence modeling
with diffusion models utilizing a decoder-only transformer. Compared to SSD-LM [55], SSD-2 [56]
carefully designed the attention mask that enhances the training efficiency of autoregressive diffusion
transformers (ARDiTs). However, it still suffers from slow speed and high computational cost during
inference. In our study, we propose to apply Distribution Matching Distillation (DMD) [64, 67] to
distill an autoregressive diffusion transformer for audio generation. To verify the performance of this
integrated approach, we apply it to zero-shot text-to-speech synthesis and speech editing tasks. -->
3. diffusion 的蒸馏可以将其转换为 single-step 模型：
    1. SSD-LM 将 autoregressive sequence modeling 与 diffusion models 结合，使用 decoder-only transformer
    2. SSD-2 设计了 attention mask，提高了 ARDiTs 的训练效率
    3. 本文提出将 Distribution Matching Distillation（DMD）用于音频合成中的自回归 DiT 蒸馏
<!-- Our contributions can be summarized as follows: -->
4. 贡献如下：
<!-- We introduce ARDiT for audio generation, a decoder-only diffusion transformer model that
eliminates the need for discrete tokenization of audio signals. -->
+ 提出 ARDiT，一个 decoder-only 的 DiT，不需要离散 tokenization
<!-- Leveraging fill-in-the-middle (FIM) [69] training, ARDiT excels in zero-shot text-to-speech
synthesis and speech editing, showcasing near-perfect speech editing capabilities on the
LibriTTS dataset. -->
+ 采用 FIM 训练，ARDiT 在 zero-shot TTS 和 speech editing 上表现很好
<!-- We distill ARDiT text-to-speech (TTS) models with DMD. After distillation, the student
models demonstrate enhanced perceptual naturalness compared to the teacher models, while
requiring only one network evaluation to generate one or more continuous tokens. One of
our distilled models achieves speech generation speeds of 170ms per network evaluation,
significantly reducing the inference latency. -->
+ 使用 DMD 蒸馏 ARDiT TTS 模型，蒸馏后的模型只需要一次 evaluation 就可以生成一个或多个 continuous tokens
<!-- Furthermore, we present a novel method for controlling the total duration of generated
speech in ARDiT TTS by manipulating the rotation angles of Rotary Position Embeddings
(RoPE) [70]. -->
+ 通过控制 RoPE 的旋转角度，控制 ARDiT TTS 生成语音总时长

## 相关工作（略）

## 方法

### 背景
<!-- Suppose X,Z are independent Rd-valued random variables with data density p(x) and Gaussian
density p(z) = N(0,Id). Let αt = (1−t) and σt = tfor t∈[0,1]. Let Xt = αtX+ σtZ. Define
the velocity field v(xt,t) : Rd ×[0,1] →Rd as: -->
假设 $X,Z$ 是独立的 $\mathbb{R}^d$ 值随机变量，其 PDF 为 $p(x)$，高斯密度为 $p(z) = \mathcal{N}(0, I_d)$。对于 $t \in [0,1]$，定义 $\alpha_t = (1-t)$，$\sigma_t = t$。令 $X_t = \alpha_tX + \sigma_tZ$，定义速度场 $v(x_t, t) : \mathbb{R}^d \times [0,1] \rightarrow \mathbb{R}^d$ 为：
$$v(x_t,t):=\underset{v}{\operatorname*{\arg\min}}E\left\|v(X_t,t)-(Z-X)\right\|_2^2=E[Z-X\mid X_t=x_t].$$
<!-- According to [104–106], we can sample p(x) by solving the following ODE in reverse: -->
可以通过求解以下 ODE 来采样 $p(x)$：
$$\mathrm{d}Y_t=v(Y_t,t)\mathrm{d}t,\quad Y_1\thicksim\mathcal{N}(0,I_d),\quad t\in[0,1].$$
<!-- Therefore we can obtain a deep generative model with sample density pθ(x) ≈p(x) by estimat-
ing v(xt,t) with vθ(xt,t) through minimizing Et∼U[0,1] ∥vθ(Xt,t)−(Z−X)∥2
2. This generative
model is referred to by various names in the literature [104–107]. In the following discussion, we
will refer to it as "Flow Matching" [105]. -->
通过估计 $v(x_t, t)$ 为 $v_{\theta}(x_t, t)$，通过最小化 $E_{t\thicksim U[0,1]}\left\|v_{\theta}(X_t,t)-(Z-X)\right\|_2^2$ 来获得生成模型，其采 PDF 为 $p_{\theta}(x) \approx p(x)$。
<!-- Suppose Xt ∼pt(xt). We can show that the score function s(xt,t) = ∇xt log pt(xt) can be
extracted from the velocity field v(xt,t) (See Appendix A). -->
假设 $X_t \thicksim p_t(x_t)$，可以证明 score function $s(x_t, t) = \nabla_{x_t} \log p_t(x_t)$ 可以从速度场 $v(x_t, t)$ 中得到：
$$v(x_t,t)=-\sigma_ts(x_t,t)-\frac{x_t+\sigma_t^2s(x_t,t)}{\alpha_t}=\frac{-1}{1-t}x_t+\frac{-t}{1-t}s(x_t,t).$$
<!-- This indicates that, akin to Diffusion Probabilistic Models (DPMs) [53, 54], Flow Matching models
also estimate the score function. -->
这表明，Diffusion Probabilistic Models（DPMs），Flow Matching 模型都是估计 score function。
<!-- Given a Flow Matching model vθ(xt,t) trained on p(x). With the ODE in equation 2, it establishes a
mapping fθ(w) : Rd →Rd that transforms Gaussian noises to data samples. Evaluating fθ is slow as
it involves solving an ODE. DMD can distill vθ into single step generator gξ : Rd →Rd, that maps
random noise W ∼N(0,Id) to X= gξ(W) with density pξ(x). Define Xt := αtX+ σtZwhere
Zis an independent Gaussian random variable. Suppose p(xt,t) is the density of Xt and pξ(xt,t) is
the density of Xt. Their Integral Kullback–Leibler (IKL) divergence [64] is defined as: -->
给定一个在 $p(x)$ 上训练的 Flow Matching 模型 $v_{\theta}(x_t, t)$。通过 ODE 建立了一个映射 $f_{\theta}(w) : \mathbb{R}^d \rightarrow \mathbb{R}^d$，将高斯噪声转换为数据样本。$f_{\theta}$ 计算涉及解 ODE，所以很慢。

DMD 可以将 $v_{\theta}$ 蒸馏为单步生成器 $g_{\xi} : \mathbb{R}^d \rightarrow \mathbb{R}^d$，将随机噪声 $W \sim \mathcal{N}(0, I_d)$ 映射到 $\widehat{X} = g_{\xi}(W)$，其密度为 $p_{\xi}(x)$。定义 $\widehat{X}_t := \alpha_t \widehat{X} + \sigma_t W$，其中 $W$ 是独立的高斯随机变量。假设 $p(x_t, t)$ 是 $X_t$ 的密度，$p_{\xi}(x_t, t)$ 是 $\widehat{X}_t$ 的密度。它们的 Integral Kullback–Leibler（IKL）散度定义为：
$$D_\xi:=D_{\mathrm{IKL}}\left(p_\xi(x_t,t)\|p(x_t,t)\right):=E_{t\thicksim\mathcal{U}[0,1]}\left[w_tD_{\mathrm{KL}}\left(p_\xi(x_t,t)\|p(x_t,t)\right)\right],$$
<!-- where wt ≥0 is the weighting factor for time t. Suppose s(xt,t) = ∇xt log p(xt,t) and sξ(xt,t) :=
∇xt log pξ(xt,t). Then according to [64, 67]: -->
其中 $w_t \geq 0$ 是时间 $t$ 的权重因子。假设 $s(x_t, t) = \nabla_{x_t} \log p(x_t, t)$ 和 $s_{\xi}(x_t, t) = \nabla_{x_t} \log p_{\xi}(x_t, t)$。有：
$$\nabla_\xi D_\xi=E_{t\thicksim\mathcal{U}[0,1]}\left[w_t\alpha_t\left(s_\xi(\widehat{X}_t,t)-s(\widehat{X}_t,t)\right)\frac{\partial g_\xi(W)}{\partial\xi}\right].$$
<!-- sξ(xt,t) and s(xt,t) are unknown, but we can approximate sξ(x
,
t)−s(xt,t) with Flow Matching
models vη(xt,t) and vθ(xt,t). Where vη is trained on samples of gξ by minimizing: -->
$s_{\xi}(x_t, t)$ 和 $s(x_t, t)$ 是未知的，但是可以用 Flow Matching 模型 $v_{\eta}(x_t, t)$ 和 $v_{\theta}(x_t, t)$ 近似：
$$\mathcal{L}_\eta:=E_{t\thicksim\mathcal{U}[0,1]}\left\|v_\eta(\widehat{X}_t,t)-(\widehat{Z}-\widehat{X})\right\|_2^2.$$
<!-- According to equation 3, assuming vη and vθ are well-trained, we have: -->
假设 $v_{\eta}$ 和 $v_{\theta}$ 训练好了，有：
$$v_\eta(x_t,t)-v_\theta(x_t,t)\approx\frac{-t}{1-t}\cdot\left(s_\xi(x_t,t)-s(x_t,t)\right).$$
<!-- DMD training freezes vθ and alternatively updates gξ and vη with Lξ and Lη. The generator gξ is
trained on minimizing the following loss function: -->
DMD 训练中，固定 $v_{\theta}$，采用以下 loss 函数交替更新 $g_{\xi}$ 和 $v_{\eta}$：
$$\mathcal{L}_{\xi}:=\underbrace{E_{t\sim\mathcal{U}[0,1]}\left\|\widehat{X}+\mathrm{sg}\left(v_\theta(\widehat{X}_t,t)-v_\eta(\widehat{X}_t,t)-\widehat{X}\right)\right\|_2^2}_{\mathcal{L}_{\mathrm{IKL}}}+\beta_{\mathrm{reg}}\cdot\underbrace{E\left\|g_\xi(W)-f_\theta(W)\right\|_2^2.}_{\mathcal{L}_{\mathrm{reg}}}$$
<!-- Here βreg > 0 is the weight of the L2 regression loss. And sg is the stop gradient operator. For
simplicity, we set wtαt = 2t/(1−t). In this case ∇ξLIKL ≈∇ξDξ. Impact of the weighting factor
wt is left for future study. -->
其中 $\beta_{\mathrm{reg}} > 0$ 是 L2 回归 loss 的权重，$\mathrm{sg}$ 是 stop gradient 操作符。为简单起见，设置 $w_t\alpha_t = \frac{2t}{1-t}$。此时，$\nabla_{\xi}\mathcal{L}_{\mathrm{IKL}} \approx \nabla_{\xi}D_{\xi}$。

### Contextual mel 谱 Autoencoder

结构如下：
![](image/Pasted%20image%2020241003161150.png)

<!-- We compress log Mel spectrograms into a sequence of continuous tokens with an autoencoder to
reduce the sequence length. Given random Mel spectrogram Y on RNframe×Dmel where Nframe is
the number of frames, and Dmel is the number of Mel filters, we encode Y into a sequence of continuous tokens Z on RNlatent×Dlatent where Nlatent = ⌊Nframe/4⌋and Dlatent = 16. The encoder
is a transformer [108] taking input Y and outputs µ,log σ ∈RNlatent×Dlatent . The encoder defines
the conditional density of Z given Y as qϕ(z|y) = n,dN(zn,d; µn,d,σ2
n,d). The decoder is a
conditional Flow Matching model vψ(yt; t,z) based on DiT [109] that recovers Y given Z. Define
the latent prior density p(z) := n,dN(zn,d; 0,1) on RNlatent×Dlatent . The encoder and decoder are
jointly optimized by minimizing: -->
将 log Mel spectrogram 用 autoencoder 压缩为连续 token 序列。给定随机 Mel spectrogram $Y \in \mathbb{R}^{N_{\text{frame}} \times D_{\text{mel}}}$，其中 $N_{\text{frame}}$ 是帧数，$D_{\text{mel}}$ 是 Mel 滤波器数，将 $Y$ 编码为连续 token 序列 $Z \in \mathbb{R}^{N_{\text{latent}} \times D_{\text{latent}}}$，其中 $N_{\text{latent}} = \left\lfloor\frac{N_{\text{frame}}}{4}\right\rfloor$，$D_{\text{latent}} = 16$。编码器是 transformer，输入 $Y$，输出 $\mu, \log\sigma \in \mathbb{R}^{N_{\text{latent}} \times D_{\text{latent}}}$。编码器定义了给定 $Y$ 的 $Z$ 的条件密度为 $q_{\phi}(z|y) = \mathcal{N}(z; \mu, \sigma^2)$。解码器是一个基于 DiT 的条件 Flow Matching 模型 $v_{\psi}(y_t; t, z)$，给定 $Z$，恢复 $Y$。定义潜在先验密度 $p(z) = \mathcal{N}(z; 0, 1)$，encoder 和 decoder 优化如下：
$$\mathcal{L}(\phi,\psi):=\beta_{\mathrm{M}}\cdot E\left[D_{\mathrm{KL}}\left(q_\phi(z|Y)\|p(z)\right)\right]+E_{W\sim\mathcal{N}(0,I)}\left\|v_\psi\left((1-t)Y+tW;t,Z\right)-(W-Y)\right\|_2^2.$$
<!-- Note that the first term E[DKL (qϕ(z|Y)∥p(z))] is a variational upper bound [110] of mutual infor-
mation I(Y; Z). So the weight βMI > 0 is controlling the trade-off between the coding rate and
reconstruction accuracy. In our experiments, the Mel spectrogram encoder emits 23.5 tokens per
second, and its theoretical bitrate is 1.7 kbps. For more details, please refer to Appendix C. -->
第一项 $E[D_{\mathrm{KL}}(q_{\phi}(z|Y)\|p(z))]$ 是互信息 $I(Y; Z)$ 的变分上界。所以权重 $\beta_{\mathrm{MI}} > 0$ 控制编码速率和重构精度之间的权衡。在实验中，Mel spectrogram encoder 每秒发出 23.5 个 token，其理论比特率为 1.7 kbps。
<!-- To enable conditional decoding when the Mel spectrogram target is partially known, the decoder is
fine-tuned on Mel spectrogram masked reconstruction. For more details, please refer to Appendix D.
During the inference stage of speech editing and zero-shot TTS, we provide the decoder with the
known Mel spectrogram frames. -->
为了在 Mel spectrogram 目标部分已知时进行条件解码，解码器在 Mel spectrogram masked reconstruction 上进行微调。在 speech editing 和 zero-shot TTS 的推断阶段，提供已知的 Mel spectrogram 帧给 decoder。

> 这部分本质是 autoencoder，将 Mel spectrogram 压缩为连续 token 序列，encoder 是 transformer，decoder 是 DiT。

<!-- Autoregressive Diffusion Transformers for Text-to-Speech Synthesis -->
### Autoregressive Diffusion Transformers 实现 TTS
<!-- In this part, we describe Autoregressive Diffusion Transformers (ARDiTs), and explain how they can
be utilized for text-to-speech synthesis. Suppose random Mel spectrogram Y is encoded into continu-
ous token sequence Z = [Z0;···; ZNlatent−1] on RNlatent×Dlatent . Suppose C = [C0,···,CNphone−1] on
ΣNphone is the phonetic transcript of Y. Where Σ is the set of all phonemes. -->
给定已经被 encode 的 Mel spectrogram $Y$，得到连续 token 序列 $Z = [Z_0; \cdots; Z_{N_{\text{latent}}-1}] \in \mathbb{R}^{N_{\text{latent}} \times D_{\text{latent}}}$。假设 $C = [C_0, \cdots, C_{N_{\text{phone}}-1}] \in \Sigma^{N_{\text{phone}}}$ 是 $Y$ 的 phoneme 序列，其中 $\Sigma$ 是所有音素的集合。
<!-- An ARDiT is semi-autoregressive, it samples from conditional density pθ(zi:i+B|c,z<i) with Flow
Matching through estimating the conditional velocity field vθ zi:i+B
t ; t,c,z<i . Here B ∈N+ is the
block size, and i∈N+ is the index of the first token in block zi:i+B = zi;···; zi+B−1 . Suppose
W is an independent RNlatent×Dlatent -valued random variable with density p(w) = n,dN(wn,d; 0,1).
Let Zt = (1−t)Z+ tW. The training loss of ARDiT would be: -->
ARDiT 是半自回归的，通过估计条件向量场 $v_{\theta}({z_t^{i:i+B}};t, c, z_{<i})$ 从 $p_{\theta}(z^{i:i+B}|c, z_{<i})$ 中采样。这里 $B \in \mathbb{N}_+$ 是 block 大小，$i \in \mathbb{N}_+$ 是 block 中第一个 token 的索引。假设 $W \in \mathbb{R}^{N_{\text{latent}} \times D_{\text{latent}}}$ 是独立随机变量，其密度为 $p(w) = \Pi_{n,d}\mathcal{N}(w_{n,d}; 0, 1)$。令 $Z_t = (1-t)Z + tW$。ARDiT 的训练 loss 为：
$$\mathcal{L}(\theta):=E_{i,t\sim\mathcal{U}[0,1]}\left\|v_\theta\left(Z_t^{i:i+B};t,C,Z^{<i}\right)-\left(W^{i:i+B}-Z^{i:i+B}\right)\right\|_2^2.$$
