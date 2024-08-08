> Microsoft Research & Microsoft Azure, USTC, preprint 2024
<!-- 翻译 & 理解 -->
<!-- While recent large-scale text-to-speech (TTS) models have achieved significant progress, they still fall short in speech quality, similarity, and prosody. Considering speech intricately encompasses various attributes (e.g., content, prosody, timbre, and acoustic details) that pose significant challenges for generation, a natural idea is to factorize speech into individual subspaces representing different attributes and generate them individually. Motivated by it, we propose NaturalSpeech 3, a TTS system with novel factorized diffusion models to generate natural speech in a zero-shot way. Specifically, 1) we design a neural codec with factorized vector quantization (FVQ) to disentangle speech waveform into subspaces of content, prosody, timbre, and acoustic details; 2) we propose a factorized diffusion model to generate attributes in each subspace following its corresponding prompt. With this factorization design, NaturalSpeech 3 can effectively and efficiently model intricate speech with disentangled subspaces in a divide-and-conquer way. Experiments show that NaturalSpeech 3 outperforms the state-of-the-art TTS systems on quality, similarity, prosody, and intelligibility, and achieves on-par quality with human recordings. Furthermore, we achieve better performance by scaling to 1B parameters and 200K hours of training data. -->
1. 现有的 TTS 在语言质量、相似度 和 韵律上还是不好
2. 语音包含了很多属性，比如内容、韵律、音色、声学细节，这些属性使得语音生成很难
3. 提出 NaturalSpeech 3，采用 factorized diffusion 进行 zero-shot 语音合成：
    1. 采用基于 factorized vector quantization (FVQ) 的 codec 将语音解耦为内容、韵律、音色、声学细节
    2. 提出 factorized diffusion，根据不同的 prompt 生成不同的属性
4. NaturalSpeech 3 在质量、相似度、韵律、可理解性上超过了现有的 TTS 系统，与人类录音质量相当

## Introduction
<!-- In recent years, significant advancements have been achieved in text-to-speech (TTS) synthesis. Traditional TTS systems [1, 2, 3, 4] are typically trained on limited datasets recorded in studios, and thus fail to support high-quality zero-shot speech synthesis. Recent works [5, 6, 7] have made considerable progress for zero-shot TTS by largely scaling up both the corpus and the model sizes. However, the synthesis results of these large-scale TTS systems are not satisfactory in terms of voice quality, similarity, and prosody. -->
<!-- The challenges of inferior results stem from the intricate information embedded in speech, since speech encompasses numerous attributes, such as content, prosody, timbre, and acoustic detail. Previ- ous works using raw waveform [8, 9] and mel-spectrogram [1, 2, 10, 7, 11] as data representations suffer from these intricate complexities during speech generation. A natural idea is to factorize speech into disentangled subspaces representing different attributes and generate them individually. However, achieving this kind of disentangled factorization is non-trivial. Previous works [12, 13, 6] encode speech into multi-level discrete tokens using a neural audio codec [14, 15] based on residual vector quantization (RVQ). Although this approach decomposes speech into different hierarchical representations, it does not effectively disentangle the information of different attributes of speech across different RVQ levels and still suffers from modeling complex coupled information. -->
1. 语音包含多种属性，之前的工作使用 raw waveform 和 mel-spectrogram 作为表征，但是这些表征在语音生成时会遇到复杂性问题，之前的工作采用 RVQ 将语音编码为多个 level 的 token，但是这种方法并不能很好地解耦不同属性的信息
<!-- To effectively generate speech with better quality, similarity and prosody, we propose a TTS system with novel factorized diffusion models to generate natural speech in a zero-shot way. Specifically, 1) we introduce a novel neural speech codec with factorized vector quantization (FVQ), named FACodec, to decompose speech waveform into distinct subspaces of content, prosody, timbre, and acoustic details and reconstruct speech waveform with these disentangled representations, leveraging information bottleneck [16, 17], various supervised losses, and adversarial training [18] to enhance disentanglement; 2) we propose a factorized diffusion model, which generates the factorized speech representations of duration, content, prosody, and acoustic detail, based on their corresponding prompts. This design allows us to use different prompts to control different attributes. The overview of our method, referred to NaturalSpeech 3, is shown in Figure 1. -->
2. 本文提出采用 factorized diffusion 模型进行 zero-shot 语音合成：
    1. 引入 FACodec，采用 factorized vector quantization (FVQ) 将语音解耦为内容、韵律、音色、声学细节，并使用 information bottleneck、各种监督 loss 和 对抗训练 来增强解耦
    2. 提出 factorized diffusion 模型，根据不同的 prompt 生成不同的属性
模型概览如下：
![](image/Pasted%20image%2020240808165539.png)
<!-- We decompose complex speech into subspaces representing different attributes, thus simplifying the modeling of speech representation. This approach offers several advantages: 1) our factorized diffusion model is able to learn these disentangled representations efficiently, resulting in higher quality speech generation; 2) by disentangling timbre information in our FACodec, we enable our factorized diffusion model to avoid directly modeling timbre. This reduces learning complexity and leads to improved zero-shot speech synthesis; 3) we can use different prompts to control different attributes, enhancing the controllability of NaturalSpeech 3. -->
3. 优势：
    1. factorized diffusion 模型能够高效地学习解耦表示，从而提高语音生成质量
    2. 通过 FACodec 解耦音色信息，避免直接建模音色，降低学习复杂度，提高 zero-shot 语音合成质量
    3. 可以使用不同的 prompt 控制不同的属性，增强了 NaturalSpeech 3 的可控性
<!-- Benefiting from these designs, NaturalSpeech 3 has achieved significant improvements in speech quality, similarity, prosody, and intelligibility. Specifically, 1) it achieves comparable or better speech quality than the ground-truth speech on the LibriSpeech test set in terms of CMOS; 2) it achieves a new SOTA on the similarity between the synthesized speech and the prompt speech (0.64 → 0.67 on Sim-O, 3.69 → 4.01 on SMOS); 3) it shows a significant improvement in prosody compared to other TTS systems with −0.16 average MCD (lower is better), +0.21 SMOS; 4) it achieves a SOTA on intelligibility (1.94 → 1.81 on WER); 5) it achieves human-level naturalness on multi-speaker datasets (e.g., LibriSpeech), another breakthrough after NaturalSpeech2. Furthermore, we demonstrate the scalability of NaturalSpeech 3 by scaling it to 1B parameters and 200K hours of training data. Audio samples can be found in https://speechresearch.github.io/naturalspeech3. -->
4. NaturalSpeech 3 在语音质量、相似度、韵律、可理解性上取得了显著的提升：
    1. 在 LibriSpeech 测试集上，语音质量与真实语音相当甚至更好
    2. 在相似度上取得了新的 SOTA（Sim-O 0.64 → 0.67，SMOS 3.69 → 4.01）
    3. 在韵律上相比其他 TTS 系统有显著提升（-0.16 MCD，+0.21 SMOS）
    4. 在可理解性上取得了 SOTA（1.94 → 1.81 WER）
    5. 在多说话人数据集上达到了人类级的自然度（例如 LibriSpeech），这是继 NaturalSpeech2 之后的又一突破
    6. 进一步展示了 NaturalSpeech 3 的可扩展性，将其扩展到 1B 参数和 200K 小时的训练数据

## 背景（略）

## NaturalSpeech 3

### 架构
<!-- In this section, we present NaturalSpeech 3, a cutting-edge system for natural and zero-shot text- to-speech synthesis with better speech quality, similarity and controllability. As shown in Figure 1, NaturalSpeech 3 consists of 1) a neural speech codec (i.e., FACodec) for attribute disentanglement; 2) a factorized diffusion model which generates factorized speech attributes. Since the speech waveform is complex and intricately encompasses various attributes, we factorize speech into five attributes including: duration, prosody, content, acoustic details, and timbre. Specifically, although the duration can be regarded as an aspect of prosody, we choose to model it explicitly due to our non-autoregressive speech generation design. We use our internal alignment tool to alignment speech and phoneme and obtain phoneme-level duration. For other attributes, we implicitly utilize the factorized neural speech codec to learn disentangled speech attribute subspaces (i.e., content, prosody, acoustic details, and timbre). Then, we use the factorized diffusion model to generate each speech attribute representation. Finally, we employ the codec decoder to reconstruct the waveform with the generated speech attributes. We introduce the FACodec in Section 3.2 and the factorized diffusion model in Section 3.3. -->
如上图，包含：
+ FACodec：用于解耦属性
+ factorized diffusion 模型：生成解耦的语音属性

将语音解耦为五个属性：duration、prosody、content、acoustic details 和 timbre
    + duration 可以看作是 prosody 的一部分，但还是选择显式建模它
    + 使用内部对齐工具对齐语音和音素，获得音素级的 duration
    + 对于其他属性，隐式地利用 factorized neural speech codec 学习解耦的语音属性子空间（content、prosody、acoustic details 和 timbre）

使用 factorized diffusion 生成每个 speech attribute 表征，最后使用 codec decoder 重构波形。

### FACodec for Attribute Factorization

#### FACodec 模型概述
<!-- We propose a factorized neural speech codec (i.e., FACodec3) to convert complex speech waveform into disentangled subspaces representing speech attributes of content, prosody, timbre, and acoustic details and reconstruct high-quality speech waveform from these. -->
提出 FACodec 将复杂的语音波形转换为解耦的子空间，表示内容、韵律、音色和声学细节，并从中重构高质量的语音波形。
<!-- As shown in Figure 2, our FACodec consists of a speech encoder, a timbre extractor, three factorized vector quantizers (FVQ) for content, prosody, acoustic detail, and a speech decoder. Given a speech x, 1) following [14, 5], we adopt several convolutional blocks for the speech encoder with a downsample rate of 200 for 16KHz speech data (i.e., each frame corresponding to a 12.5ms speech segment) to obtain pre-quantization latent h; 2) the timbre extractor is a Transformer encoder which converts the output of the speech encoder h into a global vector ht representing the timbre attributes; 3) for other attribute i (i = p, c, d for prosody, content, and acoustic detail, respectively), we use a factorized vector quantizer (FVQi) to capture fine-grained speech attribute representation and obtain corresponding discrete tokens; 4) the speech decoder mirrors the structure of speech encoder but with much larger parameter amount to ensure high-quality speech reconstruction. We first add the representation of prosody, content, and acoustic details together and then fuse the timbre information by conditional layer normalization [45] to obtain the input z for the speech decoder. We discuss how to achieve better speech attribute disentanglement in the next section. -->
如图：
![](image/Pasted%20image%2020240808171118.png)

包含：
+ speech encoder：给定语音 $x$，采用卷积 speech encoder 进行下采样，得到 pre-quantization latent $h$
+ timbre extractor：Transformer encoder，将 speech encoder 的输出转换为表示音色属性的全局向量 $h_t$
+ 三个 FVQ：prosody、content、acoustic detail：捕获细粒度的语音属性表示，获得相应的离散 token
+ speech decoder：结构与 speech encoder 相似，但参数更多，以确保高质量的语音重构。首先将韵律、内容和声学细节的表征相加，然后通过 conditional layer normalization 融合音色信息，得到 speech decoder 的输入 $z$

#### 属性解耦
<!-- Directly factorizing speech into different subspaces does not guarantee the disentanglement of speech. In this section, we introduce some techniques to achieve better speech attribute disentanglement: 1) information bottleneck, 2) supervision, 3) gradient reverse, and 4) detail dropout. Please refer to Appendix B.1 for more training details. -->
直接将语音解耦为不同的子空间并不能保证语音的解耦。这里介绍一些技术来实现更好的语音属性解耦：
<!-- Information Bottleneck. Inspired by [16, 17], to force the model to remove unnecessary information (such as prosody in content subspace), we construct the information bottleneck in prosody, content, and acoustic details FVQ by projecting the encoder output into a low-dimensional space (i.e., 8- dimension) and subsequently quantize within this low-dimensional space. This technique ensures that each code embedding contains less information, facilitating information disentanglement [32, 46]. After quantization, we will project the quantized vector back to original dimension. -->
+ information bottleneck：将 encoder 输出投影到低维空间（8 维），然后在这个低维空间内量化，强制模型去除不必要的信息，确保每个 code embedding 包含更少的信息，促进信息解耦。量化后，将量化向量投影回原始维度。
<!-- Supervision. To achieve high-quality speech disentanglement, we introduce supervision as auxiliary task for each attribute. For prosody, since pitch is an important part of prosody [37], we take the post-quantization latent zp to predict pitch information. We extract the F0 for each frame and use normalized F0 (z-score) as the target. For content, we directly use the phoneme labels as the target (we use our internal alignment tool to get the frame-level phoneme labels). For timbre, we apply speaker classification on ht by predicting the speaker ID. -->
+ supervision：为每个属性引入监督作为辅助任务。对于 prosody，由于 pitch 是 prosody 的重要部分，使用 post-quantization latent $z_p$ 预测 pitch 信息。对于 content，直接使用音素标签作为目标。对于 timbre，通过在 $h_t$ 预测说话人 ID 进行说话人分类。
<!-- Gradient Reversal. Avoiding the information leak (such as the prosody leak in content) can enhance disentanglement. Inspired by [47], we adopt adversarial classifier with the gradient reversal layer (GRL) [48] to eliminate undesired information in latent space. Specifically, for prosody, we apply phoneme-GRL (i.e., GRL layer by predicting phoneme labels) to eliminate content information; for content, since the pitch is an important aspect of prosody, we apply F0-GRL to reduce the prosody information for simplicity; for acoustic details, we apply both phoneme-GRL and F0-GRL to eliminate both content and prosody information. In addition, we apply speaker-GRL on the sum of zp, zc, zd to eliminate timbre.-->
+ gradient reverse：避免信息泄漏（比如内容中的韵律泄漏）可以增强解耦。对于 prosody，采用 phoneme-GRL（预测 phoneme label）消除内容信息；对于 content，由于 pitch 是韵律的重要方面，采用 F0-GRL 减少韵律信息；对于 acoustic details，同时采用 phoneme-GRL 和 F0-GRL 消除内容和韵律信息。此外，对 $z_p, z_c, z_d$ 的和采用 speaker-GRL 消除音色。
<!-- Detail Dropout. We have the following considerations: 1) empirically, we find that the codec tends to preserve undesired information (e.g., content, prosody) in acoustic details subspace since there is no supervision; 2) intuitively, without acoustic details, the decoder should reconstruct speech only with prosody, content and timbre, although in low-quality. Motivated by them, we design the detail dropout by randomly masking out zd during the training process with probability p. With detail dropout, we achieve the trade-off of disentanglement and reconstruction quality: 1) the codec can fully utilize the prosody, content and timbre information to reconstruct the speech to ensure the decouple ability, although in low-quality; 2) we can obtain high-quality speech when the acoustic details are given.--> 
+ detail dropout：codec 倾向于在 acoustic details 子空间中保留不必要的信息（比如内容、韵律）；没有 acoustic details，decoder 应该只用韵律、内容和音色重构语音。因此，设计 detail dropout，在训练过程中以概率 $p$ 随机屏蔽 $z_d$。通过 detail dropout，实现解耦和重构质量之间的权衡：
    + codec 可以充分利用韵律、内容和音色信息重构语音以确保解耦能力，尽管质量较低
    + 当给定 acoustic details 时，可以获得高质量的语音

### Factorized Diffusion Model

#### 模型概述
<!-- We generate speech with discrete diffusion for better generation quality. We have the following considerations: 1) we factorize speech into the following attributes: duration, prosody, content, and acoustic details, and generate them in sequential with specific conditions. Firstly, as we mentioned in Section 3.1, due to our non-autoregressive generation design, we first generate duration. Secondly, intuitively, the acoustic details should be generated at last; 2) following the speech factorization design, we only provide the generative model with the corresponding attribute prompt and apply discrete diffusion in its subspace; 3) to facilitate in-context learning in diffusion model, we utilize the codec to factorize speech prompt into attribute prompts (i.e., content, prosody and acoustic details prompt) and generate the target speech attribute with partial noising mechanism following [49, 13]. For example, for prosody generation, we directly concatenate prosody prompt (without noise) and target sequence (with noise) and gradually remove noise from target sequence with prosody prompt. -->
采用 discrete diffusion 生成语音：
+ 将语音解耦为 duration、prosody、content 和 acoustic details，按顺序生成
    + 由于非自回归生成设计，首先生成 duration
    + acoustic details 应该最后生成
+ 仅提供相应属性 prompt 给生成模型，并在其子空间中进行 discrete diffusion
+ 为了在 diffusion 中进行 in-context learning，采用 codec 将语音 prompt 解耦为属性 prompt，并使用 partial noising 生成目标语音属性
> 例如：对于 prosody 生成，直接拼接 prosody prompt（无噪声）和目标序列（有噪声），逐渐从目标序列中去除噪声

<!-- With these thoughts, as shown in Figure 3, we present our factorized diffusion model, which consists of a phoneme encoder and speech attribute (i.e., duration, prosody, content, and acoustic details) diffusion modules with the same discrete diffusion formulation: 1) we generate the speech duration by applying duration diffusion with duration prompt and phoneme-level textural condition encoded by phoneme encoder. Then we apply the length regulator to obtain frame-level phoneme condition cph; 2) we generate prosody zp with prosody prompt and phoneme condition cph; 3) we generate content prosody zc with content prompt and use generated prosody zp and phoneme cph as conditions; 4) we generate acoustic details zd with acoustic details prompt and use generated prosody, content and phoneme zp , zc , cph as conditions. Specifically, we do not explicitly generate the timbre attribute. Due to the factorization design in our FACodec, we can obtain timbre from the prompt directly and do not need to generate it. Finally, we synthesize the target speech by combining attributes zp , zc , zd and ht and decoding it with codec decoder. We discuss the diffusion formulation in Section 3.3.2. -->
如图：
![](image/Pasted%20image%2020240808173826.png)

包含：
+ phoneme encoder 
+ speech attribute diffusion 模块：duration、prosody、content、acoustic details
    + 生成 speech duration：采用 duration diffusion，输入为 duration prompt 和 phoneme encoder 得到的 phoneme-level 文本条件，然后采用 length regulator 得到 frame-level phoneme condition $c_{ph}$
    + 基于 prosody prompt 和 phoneme condition $c_{ph}$ 生成 prosody $z_p$
    + 基于 content prompt 生成 content prosody $z_c$，条件为前面得到的 prosody $z_p$ 和 phoneme $c_{ph}$
    + 基于 acoustic details prompt 生成 acoustic details $z_d$，条件为前面得到的 prosody、content 和 phoneme $z_p, z_c, c_{ph}$
    + 不显式生成 timbre 属性，因为可以直接从 prompt 中获得 timbre，无需生成
    + 最后，通过组合属性 $z_p, z_c, z_d$ 和 $h_t$ 合成目标语音，并使用 codec decoder 解码

#### diffusion 公式
<!-- Forward Process. Denote X = [xi]Ni=1 the target discrete token sequence, where N is the sequence length, Xp is the prompt discrete token sequence, and C is the condition. The forward process at time t is defined as masking a subset of tokens in X with the corresponding binary mask Mt = [mt,i]Ni=1, formulated as Xt = X ⊙ Mt , by replacing xi with [MASK] token if mt,i = 1, and otherwise leaving xi unmasked if mt,i = 0. mt,i ∼ Bernoulli(σ(t)) and σ(t) ∈ (0, 1] is a monotonically increasing
function. In this paper, σ(t) = sin( πt ), t ∈ (0, T ]. Specially, we denote X0 = X for the original 2T
tokensequenceandXT forthefullymaskedsequence. -->
前向过程：给定 $\mathbf{X}=[x_i]_{i=1}^N$ 目标 token 序列，$N$ 是序列长度，$\mathbf{X}_p$ 是 prompt token 序列，$C$ 是条件。时间 $t$ 的前向过程定义为用相应的 0/1 mask $\mathbf{M}_t=[m_{t,i}]_{i=1}^N$ 屏蔽 $X$ 中的一部分 token，公式为 $X_t=X\odot M_t$，如果 $m_{t,i}=1$，则用 [MASK] token 替换 $x_i$，否则保持不变。$m_{t,i}\sim \text{Bernoulli}(\sigma(t))$，$\sigma(t)\in(0,1]$ 是单调递增函数。本文中，$\sigma(t)=\sin(\frac{\pi t}{2T})$，$t\in(0,T]$。特别地，$\mathbf{X}_0=\mathbf{X}$ 是原始的 token 序列，$\mathbf{X}_T$ 是完全屏蔽的序列。
<!-- Reverse Process. The reverse process gradually restores X0 by sampling from reverse distribution q(Xt−∆t|X0, Xt), starting from full masked sequence XT . Since X0 is unavailable in inference, we use the diffusion model pθ, parameterized by θ, to predict the masked tokens conditioned on Xp and C, denoted as pθ(X0|Xt,Xp,C). The parameters θ are optimized to minimize the negative log-likelihood of the masked tokens: -->
反向过程：逐渐从完全屏蔽的序列 $\mathbf{X}_T$ 中采样，通过从反向分布 $q(\mathbf{X}_{t-\Delta t}|\mathbf{X}_0,\mathbf{X}_t)$ 开始。由于推断中不可用 $\mathbf{X}_0$，使用参数化为 $\theta$ 的 diffusion 模型 $p_\theta$ 预测在 $\mathbf{X}_p$ 和 $C$ 条件下的屏蔽 token，记为 $p_\theta(\mathbf{X}_0|\mathbf{X}_t,\mathbf{X}_p,C)$。参数 $\theta$ 通过最小化屏蔽 token 的负对数似然进行优化：
$$\mathcal{L}_{\text{mask}}=\underset{\mathbf{X}\in\mathcal{D},t\in[0,T]}{\operatorname*{\mathbb{E}}}-\sum_{i=1}^Nm_{t,i}\cdot\log(p_\theta(x_i|\mathbf{X}_t,\mathbf{X}^p,\mathbf{C})).$$
<!-- Then we can get the reverse transition distribution: -->
然后可以得到反向转移分布：
$$p(\mathbf{X}_{t-\Delta t}|\mathbf{X}_t,\mathbf{X}^p,\mathbf{C})=\mathbb{E}_{\mathbf{\hat{X}}_0\sim p_\theta(\mathbf{X}_0|\mathbf{X}_t,\mathbf{X}^p,\mathbf{C})}q(\mathbf{X}_{t-\Delta t}|\mathbf{\hat{X}}_0,\mathbf{X}_t).$$
<!-- Inference. During inference, we progressively replace masked tokens, starting from the fully masked sequence XT , by iteratively sampling from p(Xt−∆t |Xt , Xp , C). Inspire by [50, 51, 52], we first sample Xˆ0 from pθ(X0|Xt,Xp,C), and then sample Xt−∆t from q(Xt−∆t|Xˆ0,Xt), which involves remask ⌊N · σ(t − ∆t)⌋ tokens in Xˆ 0 with the lowest confidence score, where we define the confidence score of xˆi in Xˆ 0 to pθ (xˆi |Xt , Xp , C) if mt,i = 1, otherwise, we set confidence score of xi to 1, which means that tokens already unmasked in Xt will not be remasked. -->
推理：在推理过程中，从完全屏蔽的序列 $\mathbf{X}_T$ 开始，通过从 $p(\mathbf{X}_{t-\Delta t}|\mathbf{X}_t,\mathbf{X}^p,\mathbf{C})$ 中迭代采样逐步替换屏蔽 token。首先从 $p_\theta(\mathbf{X}_0|\mathbf{X}_t,\mathbf{X}^p,\mathbf{C})$ 中采样 $\mathbf{\hat{X}}_0$，然后从 $q(\mathbf{X}_{t-\Delta t}|\mathbf{\hat{X}}_0,\mathbf{X}_t)$ 中采样 $\mathbf{X}_{t-\Delta t}$，其中在 $\mathbf{\hat{X}}_0$ 中重新屏蔽 $\lfloor N\cdot\sigma(t-\Delta t)\rfloor$ 个置信度最低的 token，如果 $m_{t,i}=1$，则将 $X_{t,i}$ 的置信度设置为 $p_\theta(x_i|\mathbf{X}_t,\mathbf{X}^p,\mathbf{C})$，否则将 $x_i$ 的置信度设置为 1，这意味着在 $X_t$ 中已经解除屏蔽的 token 不会被重新屏蔽。
<!-- Classifier-free Guidance. Moreover, we adapt the classifier-free guidance technique [53, 54]. Specifically, in training, we do not use the prompt with a probability of pcfg = 0.15. In inference, we extrapolate the model output towards the conditional generation guided by the prompt gcond = g(X|Xp) and away from the unconditional generation guncond = g(X), i.e., gcfg = gcond + α · (gcond − guncond), with a guidance scale α selected based on experimental results. We then rescale it through gfinal = std(gcond) × gcfg/std(gcfg), following [55]. -->
Classifier-free Guidance：在训练中，以概率 $p_{\text{cfg}}=0.15$ 不使用 prompt。在推理中，通过 prompt $g_{\text{cond}}=g(\mathbf{X}|\mathbf{X}^p)$ 引导模型输出，远离无条件生成 $g_{\text{uncond}}=g(\mathbf{X})$，即 $g_{\text{cfg}}=g_{\text{cond}}+\alpha\cdot(g_{\text{cond}}-g_{\text{uncond}})$，其中 $\alpha$ 是根据实验结果选择的 guidance scale。然后通过 $g_{\text{final}}=\text{std}(g_{\text{cond}})\times g_{\text{cfg}}/\text{std}(g_{\text{cfg}})$ 重新缩放。

### 和 NaturalSpeech 系列的比较
<!-- NaturalSpeech 3 is an advanced TTS system of the NaturalSpeech series. Compared with the previous versions NaturalSpeech [4] and NaturalSpeech 2 [5], NaturalSpeech 3 has the following connections and distinctions: -->
与之前的 NaturalSpeech 和 NaturalSpeech 2 相比，NaturalSpeech 3 有以下联系和区别：
<!-- Goal. The NaturalSpeech series aims to generate natural speech with high quality and diversity. We approach this goal in several stages: 1) Achieving high-quality speech synthesis in single- speaker scenarios. To this end, NaturalSpeech [4] generates speech with quality on par with human recordings and only tackles single-speaker recording-studio datasets (e.g., LJSpeech). 2) Achieving high-quality and diverse speech synthesis on multi-style, multi-speaker, and multi-lingual scenarios. NaturalSpeech 2 [5] firstly focuses on speech diversity by exploring the zero-shot synthesis ability based on large-scale, multi-speaker, and in-the-wild datasets. Furthermore, NaturalSpeech 3 further achieves human-level naturalness on the multi-speaker dataset (e.g., LibriSpeech). -->
+ 目标：NaturalSpeech 系列旨在生成高质量和多样性的自然语音
    + 阶段 1：在单说话人场景中实现高质量语音合成。NaturalSpeech 生成与人类录音相当质量的语音，仅处理单说话人录音室数据集（例如 LJSpeech）
    + 阶段 2：在多风格、多说话人、多语言场景中实现高质量和多样性的语音合成。NaturalSpeech 2 首先通过探索基于大规模、多说话人和野外数据集的 zero-shot 合成能力来关注语音多样性。此外，NaturalSpeech 3 进一步在多说话人数据集（例如 LibriSpeech）上实现了人类级的自然度
<!-- Architecture. The NaturalSpeech series shares the basic components such as encoder/decoder for waveform reconstruction and duration prediction for non-autoregressive speech generation. Different from NaturalSpeech which utilizes flow-based generative models and NaturalSpeech 2 which leverages latent diffusion models, NaturalSpeech 3 proposes the concept of factorized diffusion models to generate each factorized speech attribute in a divide-and-conquer way. -->
+ 架构：NaturalSpeech 系列共享基本组件，如用于波形重构的 encoder/decoder 和用于非自回归语音生成的 duration 预测。不同于 NaturalSpeech 使用基于 flow 的生成模型和 NaturalSpeech 2 使用潜在扩散模型，NaturalSpeech 3 提出了 factorized diffusion 模型的概念，以分治的方式生成每个解耦的语音属性
<!--  Speech Representations. Due to the complexity of speech waveform, the NaturalSpeech series uses an encoder/decoder to obtain speech latent for high-quality speech synthesis. NaturalSpeech utilizes naive VAE-based continuous representations, NaturalSpeech 2 leverages the continuous representations from the neural audio codec with residual vector quantizers, while NaturalSpeech 3 proposes a novel FACodec to convert complex speech signal into disentangled subspaces (i.e., prosody, content, acoustic details, and timbre) and reduces the speech modeling complexity. -->
+ 语音表示：由于语音波形的复杂性，NaturalSpeech 系列使用 encoder/decoder 获得语音潜在表示进行高质量语音合成。NaturalSpeech 使用简单的 VAE-based 连续表示，NaturalSpeech 2 利用基于 RVQ 的神经音频编解码器的连续表示，而 NaturalSpeech 3 提出了 FACodec 将复杂的语音信号转换为解耦的子空间（韵律、内容、声学细节和音色），降低了语音建模的复杂性

## 实验和结果（略）
