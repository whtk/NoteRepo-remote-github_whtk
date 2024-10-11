> Microsoft Corporation, USA，preprint 2024.6

1. 提出 Embarrassingly Easy TTS（E2 TTS），非自回归 zero-shot TTS
    1. 文本输入转为 character sequence（带 filler token）
    2. 基于 audio infilling 任务训练 flow-matching-based mel spectrogram generator
    3. 无需额外模块（如 duration model, grapheme-to-phoneme）或 MAS 对齐
2. E2 TTS 达到了与 Voicebox 和 NaturalSpeech 3 相当的 zero-shot TTS 性能

## Introduction

1. 早期的 zero-shot TTS 采用 speaker embedding 作为 condition
2. VALL-E 将 zero-shot TTS 看成语言建模问题，提高了 speaker similarity
1. 但是 AR 语言模型 zero-shot TTS 有一些局限性：
    1. 推理延迟
    2. 需要找到最佳 tokenizer（文本和音频都需要）
    3. 对于长序列，需要一些技巧来稳定
2. 一些 fully NAR zero-shot TTS 模型效果不错：
    1. NaturalSpeech 2 和 3 采用 diffusion 估计 neural audio codec 的 latent vectors
    2. Voicebox 和 MatchaTTS 使用 flow-matching 
3. 但 NAR 模型很难建模 文本 和 音频 的对齐
    1. NaturalSpeech 2、3 和 Voicebox 使用 frame-wise phoneme alignment
    2. MatchaTTS 使用 MAS
    3. E3 TTS 使用 cross-attention，需要 U-Net
4. 本文发现这些技术不是必要的，有时甚至有害
5. TTS 中关于文本 tokenizer 的选择
    1. AR 模型需要仔细选择 tokenizer
    2. 大多数 NAR 模型假设文本和输出之间的单调对齐
    3. 模型对输入格式有约束，通常需要 text normalizer
    4. 当模型基于 phonemes 训练时，还需要进行 grapheme-to-phoneme
6. 提出 E2 TTS，一个简单的 NAR zero-shot TTS 系统
    1. E2 TTS 仅包含两个模块：flow-matching-based mel spectrogram generator 和 vocoder
    2. 文本输入转为 character sequence，带 filler tokens 以匹配输入和输出 mel-filterbank sequence 的长度
    3. mel 谱 generator 由 Transformer 和 U-Net skip connections 组成，使用 speech-infilling 任务训练
    4. E2 TTS 达到了与 Voicebox 和 NaturalSpeech 3 相当的 zero-shot TTS 性能

## E2 TTS

结构如图：
![](image/Pasted%20image%2020240801151059.png)

### 训练

训练流程如图 a，假设有训练样本 $s$，其文本为 $y = (c_1, c_2, ..., c_M)$，其中 $c_i$ 表示第 $i$ 个 character。首先，提取 mel-filterbank 特征 $\hat{s} \in \mathbb{R}^{D \times T}$，$D$ 为特征维度，$T$ 表示序列长度。然后，创建一个 extended character sequence $\tilde{y}$，在 $y$ 后面添加特殊的 filler token $\langle F \rangle$ 使得 $\tilde{y}$ 的长度等于 $T$：
$$\hat{y}=(c_1,c_2,\ldots,c_M,\underbrace{\langle F\rangle,\ldots,\langle F\rangle}_{(T-M)\mathrm{~times}}).$$

然后训练一个 spectrogram generator，由 vanilla Transformer 和 U-net skip connection 组成，基于 speech infilling 任务训练。具体来说，模型训练学习分布 $P(m \odot \hat{s} | (1 - m) \odot \hat{s}, \hat{y})$，其中 $m \in \{0, 1\}^{D \times T}$ 表示 0-1 mask，$\odot$ 是 Hadamard 乘积。E2 TTS 使用 conditional flow-matching 学习这种分布。

### 推理

上图 b 为推理过程。假设有一个音频 prompt $s^{aud}$ 和其文本 $y^{aud} = (c_1^\prime, c_2^\prime, ..., c_{M^{aud}}^\prime)$，用于模仿说话者特征。同时，有一个文本 prompt $y^{text} = (c_1^{\prime\prime}, c_2^{\prime\prime}, ..., c_{M^{text}}^{\prime\prime})$。
> 还需要目标语音的持续时间，这里可以任意确定，其由帧长 $T^{gen}$ 表示。

首先，从 $s^{aud}$ 提取 mel-filterbank 特征 $\hat{s}^{aud} \in \mathbb{R}^{D \times T^{aud}}$。然后，创建一个 extended character sequence $y^{\prime}$，通过连接 $y^{aud}$、$y^{text}$ 和重复的 $\langle F \rangle$ 得到：
$$\hat{y}^{\prime}=(c_{1}^{\prime},c_{2}^{\prime},\ldots,c_{{M^{{\mathrm{aud}}}}}^{\prime},c_{1}^{\prime\prime},c_{2}^{\prime\prime},\ldots,c_{{M^{{\mathrm{text}}}}}^{\prime\prime},\underbrace{{\langle F\rangle,\ldots,\langle F\rangle}}_{{\mathcal{T}\mathrm{~times}}}),$$
其中 $\mathcal{T} = T^{aud} + T^{gen} - M^{aud} - M^{text}$，确保 $y^{\prime}$ 的长度等于 $T^{aud} + T^{gen}$。

然后 mel spectrogram generator 基于学习到的分布 $P(\hat{s}^{gen} | [\hat{s}^{aud}; z^{gen}], \hat{y}^{\prime})$ 生成 mel-filterbank 特征 $\hat{s}^{gen}$，其中 $z^{gen}$ 是一个全零矩阵，形状为 $D \times T^{gen}$，$[\cdot]$ 是在 $T^*$ 维度上的连接操作。生成的 $\hat{s}^{gen}$ 通过 vocoder 转换为语音。

### 基于 flow-matching 的 mel spectrogram generator

E2 TTS 使用 conditional flow-matching，将简单初始分布 $p_0$ 转换目标分布 $p_1$。用神经网络实现，参数化为 $\theta$，训练的模型用来估计时间相关的向量场 $v_t(x; \theta)$，其中 $t \in [0, 1]$。从向量场可以得到一个 flow $\phi_t$，将 $p_0$ 转为 $p_1$。训练目标函数为 conditional flow matching：
$$\mathcal{L}^{\mathrm{CFM}}(\theta)=\mathbb{E}_{t,q(x_1),p_t(x|x_1)}\left\|u_t(x|x_1)-v_t(x;\theta)\right\|^2,$$
其中 $p_t$ 是时刻 $t$ 的概率路径，$u_t$ 是 $p_t$ 的向量场，$x_1$ 表示与训练数据对应的随机变量，$q$ 是训练数据的分布。训练时，从训练数据采用最优传输路径构建概率路径和向量场 $p(x|x_1)=N(x|t x_1, (1 - (1 - \sigma) t)^2 I)$ 和 $u(x|x_1)= (x_1 - (1 - \sigma_{\min}) x)/(1 - (1 - \sigma_{\min}) t)$。推理时，用 ODE solver 从初始分布 $p_0$ 开始生成 log mel-filterbank 特征。

采用与 [Voicebox- Text-Guided Multilingual Universal Speech Generation at Scale 笔记](Voicebox-%20Text-Guided%20Multilingual%20Universal%20Speech%20Generation%20at%20Scale%20笔记.md) 相同的模型架构，除了 frame-wise phoneme 序列替换为 $\hat{y}$。具体来说，使用 Transformer 和 U-Net style skip connection 作为 backbone。mel spectrogram generator 的输入为 $m \odot \hat{s}, \hat{y}, t, s_t$（噪声）。首先，$\hat{y}$ 转为 character embedding 序列 $\tilde{y} \in \mathbb{R}^{E \times T}$。然后，$m \odot \hat{s}, s_t, \tilde{y}$ 堆叠形成形状为 $(2 \cdot D + E) \times T$ 的张量，经过线性层输出形状为 $D \times T$ 的张量。最后，将 time embedding 表示 $\tilde{t} \in \mathbb{R}^D$ 附加到输 tensor，得到 $D \times (T + 1)$ 的 tensor 后输入 Transformer。Transformer 训练输出一个向量场 $v_t$，目标函数为 $L^{\mathrm{CFM}}$。

### 和 Voicebox 的比较

从 Voicebox 角度看，E2 TTS 用 character 替换了 frame-wise phoneme 序列，简化模型，不需要 grapheme-to-phoneme、phoneme aligner 和 phoneme duration 模型。

E2 TTS 的 mel spectrogram generator 可以看作是 grapheme-to-phoneme、phoneme duration model 和 Voicebox 的 audio model 的联合模型。这种联合建模提高了自然度，同时保持了说话者相似性和可懂性。

### E2 TTS 变体

为了在推理过程中消除对音频 prompt 的文本的要求，提出 E2 TTS X1，如图：
![](image/Pasted%20image%2020240801160943.png)

E2 TTS X1 假设可以得到音频的 masked 区域的文本，用于 $y$。在推理过程中，构建 extended character sequence $y^{\prime\prime}$ 时不包含 $y^{aud}$：
$$\tilde{y}^{\prime}=(c_1^{\prime\prime},c_2^{\prime\prime},\ldots,c^{\prime\prime}_{M^{text}},\underbrace{\langle F\rangle,\ldots,\langle F\rangle}_{\mathcal{T}\text{ times}})$$

其余过程与 E2 TTS 相同。

训练时，可以通过多种方式得到 maske 区域的文本。一种方法是在训练时对 masked 区域进行 ASR。在实验中，使用 Montreal Forced Aligner 确定每个训练数据样本中单词的开始和结束时间。确保 masked 区域不会切断单词。

第二个 E2 TTS X2 可以在推理过程中指定单词的发音。在训练时，将 $y$ 中的单词偶尔替换为括号中的 phoneme 序列。在推理时，将目标单词替换为括号中的 phoneme 序列：
![](image/Pasted%20image%2020240801162717.png)

实现上，用 CMU pronouncing dictionary 中的 phoneme 序列替换单词。可以发现，这里 $y$ 仍然是一个简单的 character 序列，而 character 是单词还是 phoneme 由括号的存在和内容决定。
> 替换时保留单词周围的标点符号，允许模型在单词被替换为 phoneme 序列时使用这些标点符号。

## 实验（略）
