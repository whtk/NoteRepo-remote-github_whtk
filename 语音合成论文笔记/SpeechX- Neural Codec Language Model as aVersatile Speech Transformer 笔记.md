> TASLP 2024，Microsoft

1. 提出 SpeechX，实现 zero-shot TTS 和各种 speech transformation 任务
2. SpeechX 将 neural codec LM 和多任务学习结合，使用 task-dependent prompting，实现统一和可扩展的建模，可以利用文本输入进行 speech enhancement 和 transformation 任务
3. 实验表明 SpeechX 在 zero-shot TTS、noise suppression、target speaker extraction、speech removal 和 speech editing 等任务上表现良好

> 也是基于 VALL-E 进行多任务学习，引入额外 tokens 指定任务。但是可以处理噪声。且这里的多任务是指一些语音上的任务，不是多模态的任务。

## Introduction

1. 现有的基于 codec LM 的一些限制举例：
    1. speech editing 模型只能处理 clean signals，无法保留背景音
    2. denoising 模型需要 noisy signal 两侧有 clean speech segments，限制了实际应用
    3. target speaker extraction 任务没有使用过 generative 模型
    4. 传统语音增强方法使用回归模型，需要为每个任务训练专门的模型，不适用于多样的 acoustic disturbances
2. audio-text-based 模型需要统一 generation 和 transformation 能力，能够处理多样的 speech generation 任务，具有一下特性：
    1. Versatility：能够处理多种任务，包括 zero-shot TTS 和 speech transformation，如 speech enhancement 和 speech editing
    2. Robustness：对各种 acoustic distortions 都要 robust，适用于真实场景
    3. Extensibility：使用灵活的架构，支持 task 的扩展，如增加 input tokens 或模块
3. 提出 SpeechX，能够处理多种任务，包括 zero-shot TTS、noise suppression、speech removal、target speaker extraction 和 speech editing：
    1. 基于 text 和 acoustic inputs 生成 neural codec model 的 codes，即 acoustic tokens
    2. 在多任务学习中引入额外 tokens 实现多任务处理

## 相关工作（略）

## 方法

### 概览

架构如图：
![](image/Pasted%20image%2020241021114341.png)

模型基于 [VALL-E- Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers 笔记](VALL-E-%20Neural%20Codec%20Language%20Models%20are%20Zero-Shot%20Text%20to%20Speech%20Synthesizers%20笔记.md)，使用 Transformer 架构的 neural codec LM，根据 text prompt $\mathcal{T}$ 和 acoustic prompt $\mathcal{A}$ 生成 neural code sequence $O$，即 acoustic tokens。

text prompt $\mathcal{T}$ 是通过 grapheme-to-phoneme conversion 音素，包含语义信息，称为 semantic tokens。acoustic prompt $\mathcal{A}$ 包含输入语音的 acoustic 信息，通过 neural codec encoder 转为 acoustic tokens。为了指定任务，在 acoustic prompt 中加入额外 tokens。输出 $\mathcal{O}$ 是目标 codes，通过 decoder 转为 waveform。

采用 [EnCodec- High Fidelity Neural Audio Compression 笔记](../语音领域其他论文笔记/EnCodec-%20High%20Fidelity%20Neural%20Audio%20Compression%20笔记.md) 作为 codec，得到 75Hz 采样率的 1024 维 codes。

### Neural Codec LM

SpeechX 使用 auto-regressive (AR) 和 non-auto-regressive (NAR) Transformer，AR 用于生成 EnCodec 的第一个 quantization 层的 codes，NAR 用于生成二到八层的 codes。

输出 $\mathcal{O}$ 表示为矩阵 $\mathbf{O}=[o_{t,l}]\in \mathbb{N}^{T\times L}$，$o_{t,l}$ 表示第 $l$ 层的第 $t$ 个 code。AR 模型由 Transformer decoder 组成，通过最小化第一层 code 的负对数似然进行优化：
$$\mathcal{L}_{A R}=-\sum_{t=1}^{T}\log{P}(o_{t,1}|\mathcal{T},\mathcal{A},\mathbf{o}_{<t,1};\theta_{A R}),$$
其中 $\mathbf{o}_{<t,1}=[o_{1,1},\dots,o_{t-1,1}]$。text 和 acoustic tokens 通过不同的 embedding 投影得到。

AR 模型的条件为 acoustic prompt $\mathcal{A}$、text prompt $\mathcal{T}$ 和过去的 acoustic history $\mathbf{o}_{<t,1}$。与 VALL-E 不同，VALL-E 的 AR 模型只与 $\mathcal{T}$ 和 $\mathbf{o}_{<t,1}$ 有关，acoustic prompt 推理时作为 $\mathbf{o}_{<t,1}$ 的一部分。
> 进行 zero-shot TTS 推理时，SpeechX 不需要 audio prompt 的文本。VALL-E 需要将 audio prompt 的文本和 text prompt 连接为 $\mathcal{T}$，audio prompt 序列为 $\mathbf{o}_{<t,1}$。SpeechX 推理时 $\mathcal{T}$ 就是 text prompt，模型可以生成 codec sequence 而不需要 audio prompt 对应的文本。

得到第一层 codes 后，NAR 模型根据 text 和 acoustic prompts 以及前 $l - 1$ 层的 codes 生成第 $l$ 层的 codes。重复 $l=2,\dots,8$。NAR 模型训练时最小化负对数似然：
$$\mathcal{L}_{N A R}=-\sum_{l=2}^{8}\log P(\mathbf{o}_{:,l}|\mathcal{T},\mathcal{a},\mathbf{o}_{:,<l};\theta_{N A R})$$
其中 $\mathbf{o}_{:,l}$ 表示第 $l$ 层的所有 codes，$\mathbf{o}_{:,<l}=[\mathbf{o}_{:,1},\dots,\mathbf{o}_{:,l-1}]$。为了让单个 NAR 模型处理每一层，前 $l-1$ 层的 acoustic tokens $\mathbf{o}_{:,<l}$ 进行 embeded 并求和。

### 基于任务的 prompting

采用基于任务的 prompting 来处理不同的任务：
![](image/Pasted%20image%2020241021161117.png)

noise suppression 任务：从 noise-corrupted observation $s+n$ 中提取干净语音$s$，$n$ 为噪声。采用特殊 token `<ns>`，得到 acoustic prompt $A=[<ns>,C(s+n)]$。text prompt $\mathcal{T}$ 为文本，可选。目标输出是干净音频的 acoustic token 序列 $C(s)$。

speech removal 任务：从噪声语音信号中中移除语音，保留背景噪声。采用特殊 token `<sr>`，得到 acoustic prompt $A=[<sr>,C(s+n)]$。目标输出是噪声信号的 acoustic token 序列 $C(n)$。

target speaker extraction 任务：从混合语音中提取目标说话人的干净语音$s_1$，目标说话人通过短的 enrollment 音频$s'_1$ 确定。acoustic prompt 为 $A=[C(s'_1),<tse>,C(s_1+s_2)]$。目标输出是 $C(s_1)$。

zero-shot TTS 任务：通过输入文本和 enrollment 语音$s$ 生成 speech signal $s^\prime$。acoustic prompt 为 $A=C(s)$。模型根据输入文本生成 acoustic tokens，转为 waveform。

clean speech editing 任务：修改输入语音以匹配输入文本。将输入语音$s$ 分为三部分，$s_{pre}$、$s_{mid}$ 和 $s_{post}$，其中 $s_{mid}$ 为要编辑部分。acoustic prompt 为 $[C(s_{pre}),<soe>,<mask>,<eoe>,C(s_{post})]$，token `<soe>`、`<mask>`、`<eoe>` 用于指定任务和编辑部分。目标输出是 neural codes 序列 $[C(s_{pre}),C(s_{edit}),C(s_{post})]$，其中 $[s_{pre},s_{edit},s_{post}]$ 的内容与输入文本匹配。

noisy speech editing 任务：在噪声语音上操作，修改语音内容而保留背景噪声。acoustic prompt 为 $[C(s_{pre}+n_{pre}),<soe>,C(s_{mid}+n_{mid}),<eoe>,C(s_{post}+n_{post})]$。目标输出是 neural codes 序列 $[C(s_{pre}+n_{pre}),C(s_{edit}+n_{mid}),C(s_{post}+n_{post})]$，模型需要在编辑过程中区分 speech 和 noise。

在实际编辑场景中，输入文本通常通过 ASR 获得，然后由用户编辑。此时可以确定 `<soe>` 和 `<eoe>` 的位置。

> 基于任务的 prompting 使 SpeechX 在推理时能够唯一确定输出，灵活性高，可以添加新任务，只需初始化新引入的 task-specific tokens 的 embedding。

### 模型训练

训练时，每次 update 的时候随机采样任务。对于 noise suppression、speech removal 和 target speaker extraction 任务，50% 概率包含 text prompt，使模型能够看到 text 和 text-less 场景。

为了实现生成，首先只训练 zero-shot TTS，然后使用所有任务进行多任务学习。使用已有的 VALL-E 模型 checkpoint 来初始化模型。
> 注意：这里两者的训练数据不一样：SpeechX 的训练样本显式地包含一个不同的 enrollment 音频，而 VALL-E 不包含。

多任务训练阶段，特殊 tokens 进行随机初始化。

## 评估（略）

## 实验（略）
