> Preprint 2024，中科大、讯飞
<!-- 翻译 & 理解 -->
<!-- In recent years, diffusion-based text-to-music (TTM) generation has gained prominence, offering an innovative approach to synthesizing musical content from textual descriptions. Achieving high accuracy and diversity in this generation process requires extensive, high-quality data, including
both high-fidelity audio waveforms and detailed text descriptions, which often constitute only a small portion of available
datasets. In open-source datasets, issues such as low-quality
music waveforms, mislabeling, weak labeling, and unlabeled
data significantly hinder the development of music generation models. To address these challenges, we propose a novel
paradigm for high-quality music generation that incorporates
a quality-aware training strategy, enabling generative models to discern the quality of input music waveforms during
training. Leveraging the unique properties of musical signals,
we first adapted and implemented a masked diffusion transformer (MDT) model for the TTM task, demonstrating its
distinct capacity for quality control and enhanced musicality. Additionally, we address the issue of low-quality captions
in TTM with a caption refinement data processing approach.
Experiments demonstrate our state-of-the-art (SOTA) performance on MusicCaps and the Song-Describer Dataset. Our
demo page can be accessed at https://qa-mdt.github.io/. -->
1. 现有的 text-to-music (TTM) 需要高质量的数据，但是开源的数据集中存在低质量音频、错误标注、弱标注和无标注数据等问题
2. 本文提出 quality-aware training strategy，让生成模型在训练时识别输入音频的质量
    1. 先使用 masked diffusion transformer (MDT) 模型实现 TTM
    2. 使用 caption refinement data processing 解决 TTM 中低质量 caption 的问题
3. 在 MusicCaps 和 Song-Describer Dataset 上实现了 SOTA 

## Introduction
<!-- Text-to-music (TTM) generation aims to transform textual
descriptions of emotions, style, instruments, rhythm, and
other aspects into corresponding music segments, providing new expressive forms and innovative tools for multimedia creation. According to scaling law principles (Peebles
and Xie 2023; Li et al. 2024a), effective generative models require a large volume of training data. However, unlike image generation tasks (Chen et al. 2024a; Rombach
et al. 2021), acquiring high-quality music data often presents
greater challenges, primarily due to copyright issues and the
need for professional hardware to capture high-quality music. These factors make building a high-performance TTM
model particularly difficult. -->
1. TTM 将文字描述（情感、风格、乐器、节奏等）转换为音乐片段
2. 获取高质量音乐数据困难（版权）
<!-- In the TTM field, high-quality paired data of text and
music signals is scarce. This prevalent issue of low-quality
data, highlighted in Figure 1, manifests in two primary challenges. Firstly, most available music signals often suffer from distortion due to noise, low recording quality, or outdated recordings, resulting in diminished generated quality,
as measured by pseudo-MOS scores from quality assessment models (Ragano, Benetos, and Hines 2023). Secondly,
there is a weak correlation between music signals and captions, characterized by missing, weak, or incorrect captions,
leading to low text-audio similarity, which can be indicated
by CLAP scores (Wu* et al. 2023). These challenges significantly hinder the training of high-performance music generation models, resulting in poor rhythm, noise, and inconsistencies with textual control conditions in the generated
audio. Therefore, effectively training on large-scale datasets
with label mismatches, missing labels, or low-quality waveforms has become an urgent issue to address. -->
1. TTM 领域缺乏高质量的 text-music pair 数据，且：
    1. 音乐信号质量低
    2. 音乐信号和 caption 之间的相关性弱
<!-- In this study, we introduce a novel quality-aware masked
diffusion transformer (QA-MDT) to enhance music generation. This model effectively leverages extensive, opensource music databases, often containing data of varying
quality, to produce high-quality and diverse music. During training, we inject quantified music pseudo-MOS (pMOS) scores into the denoising stage at multiple granular ities to foster quality awareness, with coarse-level quality
information seamlessly integrated into the text encoder and
fine-level details embedded into the transformer-based diffusion architecture. A masking strategy is also employed to
enhance the spatial correlation of the music spectrum and
further accelerate convergence. This innovative approach
guides the model during the generation phase to produce
high-quality music by leveraging information associated
with elevated p-MOS scores. Additionally, we utilize large
language models (LLMs) and CLAP model to synchronize
music signals with captions, thereby enhancing text-audio
correlation in extensive music datasets. Our ablation studies
on public datasets confirm the effectiveness of our methodology, with the final model surpassing previous works in
both objective and subjective measures. The main contributions of this study are as follows: -->
2. 提出 quality-aware masked diffusion transformer (QA-MDT)：
    1. 在训练时，将 quantified music pseudo-MOS (pMOS) scores 注入到 denoising stage 来提高质量
    2. 使用 masking strategy 增强音乐频谱的空间相关性，加速收敛
    3. 使用 LLMs 和 CLAP model 同步音乐信号和 caption，提高 text-audio 相关性

## 相关工作（略）

## 预备知识
<!-- Latent diffusion model. Direct application of DMs to
cope with distributions of raw signals incurs significant
computational overhead (Ho, Jain, and Abbeel 2020;
Song, Meng, and Ermon 2020). Conversely, studies (Liu
et al. 2023c,b) apply them in a latent space with fewer
dimensions. The latent representation z0 is the ultimate
prediction target for DMs, which involve two key pro-
cesses: diffusion and reverse processes. In the diffusion
process, Gaussian noise is incrementally added to the
original representation at each time step t, described by √√
zt+1 = 1−βtzt + βtε, where ε is drawn from a standard normal distribution N (0, I ), and βt is gradu- ally adapted based on a preset schedule to progressively  -->
DM 计算开销大。将 DMs 用于低维 latent space 得到 LDM。latent representation $z_0$ 是 DMs 的最终预测目标，包括两个关键过程：diffusion 和 reverse processes。在 diffusion 过程中，每个时间步 $t$ 逐渐添加高斯噪声：$z_{t+1} = \sqrt{(1-\beta_t)}z_t + \sqrt{beta_t}\epsilon$，其中 $\epsilon$ 来自 $N(0, I)$。损失函数为：$\arg\min_\theta\mathbb{E}_{(\boldsymbol{z}_0,y),\epsilon}\left[\left\|\epsilon-D_\theta\left(\sqrt{\alpha_t}z_0+\sqrt{1-\alpha_t}\epsilon,t,y\right)\right\|^2\right]$，其中 $D_\theta$ 是估计噪声 $\epsilon$ 的 denoising 模型，$\alpha$ 为递增函数。reverse 过程中，通过 $z_{t-1}=\frac{1}{\sqrt{1-\beta_{t}}}\left(z_{t}-\frac{\beta_{t}}{\sqrt{1-\alpha_{t}}}\epsilon_{\theta}\right)+\sqrt{\frac{1-\alpha_{t-1}}{1-\alpha_{t}}\beta_{t}}\epsilon$，其中 $\epsilon_{\theta}$ 是估计的噪声。

<!-- 
Classifier-free guidance. Classifier-free guidance (CFG),
introduced by (Ho, Jain, and Abbeel 2020), increases the
versatility of DMs by enabling both conditional and unconditional generation. Typically, a diffusion model generates content based on specific control signals y within its
denoising function Dθ(zt, t, y). CFG enhances this mechanism by incorporating an unconditional mode Dθ(zt, t, ∅),
where ∅ symbolizes the absence of specific control signals. The CFG-enhanced denoising function is then expressed as DCFG
θ
(zt, t, y) = Dθ(zt, t, y) + w(Dθ(zt, t, y) −
Dθ(zt, t, ∅)), where w ≥ 1 denotes the guidance scale. During training, the model substitutes y with ∅ at a constant
probability puncond. In inference, ∅ might be replaced by a
negative prompt like “low quality” to prevent the model
from producing such attributes (Liu et al. 2023b) -->
CFG 增加了 DMs 的灵活性，使其能够进行有条件和无条件生成。CFG-enhanced denoising function 为 $DCFG_{\theta}(z_t, t, y) = D_{\theta}(z_t, t, y) + w(D_{\theta}(z_t, t, y) - D_{\theta}(z_t, t, \emptyset))$，其中 $w \geq 1$ 为 guidance scale。在训练时，模型以概率 $p_{\text{uncond}}$ 将 $y$ 替换为 $\emptyset$。在推理时，$\emptyset$ 可能被替换为负面提示，如“low quality”。

## 方法
<!-- Quality Information Injectio -->
### 质量信息注入
<!-- At the heart of our work lies the implementation of a pseudo- MOS scoring model (Ragano, Benetos, and Hines 2023) to meticulously assign music quality to quality prefixes and quality tokens. -->
使用 pseudo-MOS scoring model 给 quality prefixes 和 quality tokens 分配 music quality。
<!-- We define our training set as Do = {(Mi,Tio) | i = 1, 2, . . . , ND }, where each Mi represents a music signal and Tio is the corresponding original textual description. To optimize model learning from datasets with diverse au- dio quality and minimize the impact of low-quality audio, we initially assign p-MOS scores to each music track us- ing a model fine-tuned with wav2vec 2.0 (Baevski et al. 2020) on a dataset of vinyl recordings for audio quality as- sessment, and achieve the corresponding p-MOS set S = {s1 , s2 , . . . , sn }. These scores facilitate dual-perspective quality control for enhanced granularity and precision. -->
训练集 $D_o = \{(M_i, T_{io}) | i = 1, 2, \ldots, N_D\}$，其中 $M_i$ 表示音乐信号，$T_{io}$ 是对应的原始文本描述。对于每个 music track，使用 fine-tuned wav2vec 2.0 模型计算 p-MOS scores，得到 p-MOS set $S = \{s_1, s_2, \ldots, s_n\}$。

<!-- First, We analyze this p-MOS set S to identify a nega- tive skew normal distribution with mean μ and variance σ2. We define text prefixes based on s as follows: prepend “low quality” if s < μ−2σ, “medium quality” if μ−σ ≤ m ≤ μ+σ,and“highquality”ifs > μ+2σ.Thisin- formation is prepended before processing through the text encoder with cross-attention, enabling the initial separation of quality-related information. -->
首先分析 p-MOS set $S$ 识别 skew normal distribution，其中 mean 为 $\mu$，variance 为 $\sigma^2$。定义 text prefixes：
+ 如果 $s < \mu - 2\sigma$，则在 $s$ 前加上 “low quality”
+ 如果 $\mu - \sigma \leq m \leq \mu + \sigma$，则加上 “medium quality”
+ 如果 $s > \mu + 2\sigma$，则加上 “high quality”
> 这些信息在通过 text encoder 时与 cross-attention 一起处理，实现质量相关信息的初始分离。

<!-- To achieve a more precise awareness and control of wave- form quality, we synergize the role of text control with qual- ity embedding. We observed that the distribution of p-MOS in the dataset is approximately normal, which can be shown in Figure 1, allowing us to use the Empirical Rule to segment the data accordingly. Specifically, we define the quantiza- tion function Q : [0,5] → {1,2,3,4,5} to map the p-MOS scores to discrete levels based on the distance from the mean μ in terms of standard deviation σ: -->
为了更精确地控制 waveform quality，将 text control 与 quality embedding 结合。观察到 p-MOS 分布近似正态，可以使用 Empirical Rule 对数据进行分段。定义 quantization function $Q : [0, 5] \to \{1, 2, 3, 4, 5\}$，将 p-MOS scores 映射到离 mean $\mu$ 的标准差 $\sigma$ 的距离上：
$$Q(s)=\left\lfloor\frac{s-(\mu-2\sigma)}{\sigma}\right\rfloor+r$$
<!-- where r = 2 for s > μ, otherwise, r = 1. Subsequently, Q(s) is mapped to a d-dimensional quality vector embed- ding using the embedding function E, such that -->
对于 $s > \mu$，$r = 2$，否则 $r = 1$。然后，$Q(s)$ 通过 embedding function $E$ 映射到 $d$ 维 quality vector embedding：
$$q_{\mathrm{vq}}(s)=E(Q(s))\in\mathbb{R}^d,$$
<!-- This process provides finer granularity of control within the following model and facilitates the ability of interpolative quality control during inference, enabling precise adjust- ments in Rd . -->
从而可以更精细地控制模型。
<!-- Quality-aware Masked Diffusion Transformer -->
### Quality-aware Masked Diffusion Transformer
<!-- In a general patchify phrase with patch size pf × pl and overlap size of × ol, patchified token sequence X ={x ,x ,...,x } ⊂ Rpf×pl are obtained through spliting 12P F×L the music latent space Mspec ∈ R , as described in Sec- tion 5. The total number of patches P is given by: -->
patchify 阶段，patch size 为 $p_f \times p_l$，overlap size 为 $o_f \times o_l$，得到 patchified token sequence $X = \{x_1, x_2, \ldots, x_P\} \subset \mathbb{R}^{p_f \times p_l}$，其中 $P$ 为 patch 总数：
$$P=\left\lceil\frac{L-p_l}{p_l-o_l}+1\right\rceil\times\left\lceil\frac{F-p_f}{p_f-o_f}+1\right\rceil $$
<!-- A 2D-Rope position embedding (Su et al. 2024) is added to each patch for better modeling of relative position relationshipwhileabinarymaskm∈{0,1}P isappliedduring the training stage, with a variable mask ratio γ. This results in a subset of ⌊γP⌋ patches being masked that PPN m = i=1 i ⌊γP ⌋, leaving P − ⌊γP ⌋ patches unmasked. The subset of masked tokens is invisible in the encoder stage and replaced with trainable mask tokens in the decoder stage following the same strategy utilized in AudioMAE (Huang et al. 2022) and MDT (Gao et al. 2023). -->
每个 patch 添加 2D-Rope position embedding 建模相对位置。

训练时，binary mask $m \in \{0, 1\}^P$，mask ratio 为 $\gamma$，其中 $\lfloor\gamma P\rfloor$ 个 patches 被 mask，剩下的 $P - \lfloor\gamma P\rfloor$ 个 patches 不被 mask。被 mask 的 tokens 在 decoder 阶段被可训练的 mask tokens 替换。
<!-- The transformer we use consists of N encoder blocks,
M decoder blocks, and an intermediate layer to replace the
masked part with trainable parameters. We treat the em-
bedding of the quantized p-MOS score as a prefix token,
concatenated with each stage’s music tokens. Let Xk =
[xk1,xk2,...,xkP ] ∈ RP×d represent the output of k-th en-
coder or decoder block, where the initial input of the encoder
X0 =z =αz +√1−αε,andthefinaldecoderblock tt0t
estimate XN+M = z0 = [x1,x2,...,xP ]. For k < N, indicating the encoder blocks, the sequence transformation focuses only on unmasked tokens:
 -->
使用 N 个 encoder blocks，M 个 decoder blocks 和一个 intermediate layer（用于替换被 mask 的部分）。将 quantized p-MOS score 的 embedding 视为 prefix token，与每个阶段的 music tokens 拼接。$X^k = [x_{1}^k, x_{2}^k, \ldots, x_{P}^k] \in \mathbb{R}^{P \times d}$ 表示第 $k$ 个 encoder 或 decoder block 的输出，其中 encoder 的初始输入 $X^0 = z_t = \alpha_t z_0 + \sqrt{1-\alpha_t}\epsilon$，decoder 的最终输出 $X^{N+M} = z_0 = [x_1, x_2, \ldots, x_P]$。对于 $k < N$，即 encoder blocks。整个过程仅关注未被 mask 的 tokens：
$$[q_{\mathrm{vq}}^{k+1};X^{k+1}]=\text{ Encoder}^k\left(\left[q_{\mathrm{vq}};X^k\odot(\mathbf{1-m})\right]\right),$$
<!-- where m ∈ {0, 1}P is the mask vector, with 1 indicating masked positions and 0 for visible tokens. -->
其中 $m \in \{0, 1\}^P$ 是 mask vector，1 表示 mask 位置，0 表示可见 tokens。
<!-- /For N < k < N + M , indicating the decoder blocks, the full sequence including both unmasked tokens and learnable masked tokens is considered: -->
对于 $N < k < N + M$，即 decoder blocks。考虑包括未被 mask 和可学习 mask tokens 的完整序列：
$$[q_{\mathrm{vq}}^{k+1};X^{k+1}]=\text{Decoder}^k\left(\left[q_{\mathrm{vq}};X^k\right]\right),$$
<!-- where the previously masked tokens are now subject to pre- diction and refinement. In the decoding phase, the portions that were masked are gradually predicted, and throughout this entire phase, the quality token qvq(s) is progressively infused and optimized. Subsequently, the split patches are unpatchified while the overlapped area is averaged to recon- struct the output noise and every token contributes to calcu- lating the final loss: -->
先前被 mask 的 tokens 现在被预测和优化。在解码阶段，被 mask 的部分逐渐被预测，整个过程中，quality token $q_{\mathrm{vq}}(s)$ 逐渐被注入和优化。然后，split patches 被 unpatchified，overlap 区域被平均以重构输出 noise，每个 token 贡献于计算最终损失：
$$\mathcal{L}(\theta)=\mathbb{E}_{(z_0,q_{\upsilon q},y),\epsilon}\left[\left\|\epsilon-D_\theta\left(\sqrt{\alpha_t}z_0+\sqrt{1-\alpha_t}\epsilon,t,q_{vq},y\right)\right\|^2\right]$$
<!-- In the inference stage, the model can be guided to generate high-quality music through CFG: -->
推理阶段，通过 CFG 引导模型生成高质量音乐：
$$\begin{aligned}D_\theta^{\mathrm{High}}(z_t,t,q_{vq}^{\mathrm{high}},y)&=D_\theta(z_t,t,q_{vq}^{\mathrm{high}},y)+\\&w\left(D_\theta(z_t,t,q_{vq}^{\mathrm{high}},y)-D_\theta(z_t,t,q_{vq}^{\mathrm{low}},\emptyset)\right)\end{aligned}$$
<!-- indicate quantified p-MOS for guiding the model in a balance between generation quality and di-versity. -->
其中 $q_{vq}^{\text{high}}$ 和 $q_{vq}^{\text{low}}$ 表示 quantified p-MOS，用于平衡生成质量和多样性。

<!-- Music Caption Refinement -->
### Music Caption Refinement
