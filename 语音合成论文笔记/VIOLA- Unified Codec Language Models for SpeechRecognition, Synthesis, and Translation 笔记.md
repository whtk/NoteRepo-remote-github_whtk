> TASLP 2024.07 发表，2023.05 arxiv，Microsoft

1. 提出 VioLA，用单个自回归 Transformer decoder 统一文本和语音的跨模态任务，如 speech-to-text、text-to-text、text-to-speech 和 speech-to-speech 任务，采用多任务学习框架来实现 conditional codec LM 任务：
    1. 先使用 codec 将所有的语音转为离散 token，从而将所有任务转为基于 token 的序列转换问题，可用一个 conditional LM 处理  
    2. 将任务 ID（TID）和语言 ID（LID）引入到模型，增强处理不同语言和任务的能力
2. 实验结果表明，VioLA 支持单模态和跨模态任务，decoder-only 模型的性能相比于 strong baselines 有可比甚至更好的性能

> 其实就是把 VALL-E 拓展到不同任务，然后用多任务的训练损失函数。

## Introduction

1. [VALL-E- Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers 笔记](VALL-E-%20Neural%20Codec%20Language%20Models%20are%20Zero-Shot%20Text%20to%20Speech%20Synthesizers%20笔记.md) 只能实现单个的语音合成任务，本文探索：一个 decoder-only 生成模型是否可以实现语音识别、合成和翻译
2. 提出了多语言多模态自回归 Transformer 模型 VioLA，将 speech-to-text、text-to-text、text-to-speech 和 speech-to-speech 任务统一为 conditional codec LM 任务，如图：
![](image/Pasted%20image%2020241010220921.png)
    1. 通过 [EnCodec- High Fidelity Neural Audio Compression 笔记](../语音领域其他论文笔记/EnCodec-%20High%20Fidelity%20Neural%20Audio%20Compression%20笔记.md) 将连续语音转为离散 codes，从而将语音表征视为文本 token，并使用 decoder-only 模型优化多模态任务
    2. 引入任务 ID（TID）和语言 ID（LID）以增强区分不同语言和任务的能力
3. 在多任务学习框架下训练 VioLA 模型：ASR、MT 和 TTS，实验结果表明 VioLA 可以有效用于单模态和跨模态任务，且保持 in-context 学习能力

## 相关工作（略）

## 多任务 Codec LM

### 背景

模型基于 VALL-E，包含两部分：
+ 自回归 codec LM 基于语义 token（音素序列）预测每帧的第一个 code（声学 token）
+ 非自回归 codec LM 并行生成其他 7 层 code

本文将 VALL-E 从单个 TTS 任务扩展到更多语音任务，提升自回归 codec LM 的能力。

### 问题描述
VioLA 将所有语音任务视为 conditional codec LM，核心是将所有语音数据转为离散 token，从而将 speech-to-text、text-to-text 和 text-to-speech 任务转为基于 token 的序列转换任务。

给定数据集 $\mathcal{D} = \{\mathbf{x}_i, \mathbf{y}_i\}$，其中 $t \in \{\text{asr}, \text{mt}, \text{tts}\}$ 表示不同任务，$i$ 是训练样本索引，$\mathbf{x}$ 和 $\mathbf{y}$ 分别表示：
+ ASR 任务的音频和文本
+ MT 任务的源文本和目标文本
+ TTS 任务的文本和音频

数据集中的音频和文本通过 EnCodec 和 G2P 工具转为离散 codes 和 phonemes。

具体来说，ASR 和 TTS 数据的音频 $\mathbf{x}^{\text{asr}}$ 和 $\mathbf{y}^{\text{tts}}$ 转为离散 acoustic token，$\text{EnCodec}(\mathbf{x}^{\text{asr}}) = \mathbf{A}^{\text{asr}, 1:8} \in \mathbb{R}^{T_A \times 8}$，$\text{EnCodec}(\mathbf{y}^{\text{tts}}) = \mathbf{A}^{\text{tts}, 1:8} \in \mathbb{R}^{T_A \times 8}$，其中 $\mathbf{A}$ 是 8 层 acoustic token 矩阵，$T_A$ 是下采样后的语音长度。量化后，EnCodec decoder 从 acoustic tokens 重构波形，记为 $\text{Decodec}(\mathbf{A}) \approx \hat{\mathbf{y}}$。文本数据通过 G2P 工具转为 phoneme 序列，分别记为 $\mathbf{S}^{\text{asr}}$、$\mathbf{S}^{\text{mt,s}}$、$\mathbf{S}^{\text{mt,t}}$ 和 $\mathbf{S}^{\text{tts}}$，其中 $T_S$ 是 phoneme 序列长度，$s$ 和 $t$ 分别表示源语言和目标语言。

本文目标是利用 conditional codec LM 建模各种任务，模型通过最大化 $p(\mathbf{y}|\mathbf{x}, \theta) = p(\mathbf{S}^{\text{asr}}|\mathbf{A}^{\text{asr}, \theta}) + p(\mathbf{S}^{\text{mt,t}}|\mathbf{S}^{\text{mt,s}}, \theta) + p(\mathbf{A}^{\text{tts}}|\mathbf{S}^{\text{tts}}, \theta)$ 进行优化，类似 GPT 中的生成式预训练方法，采用多任务学习框架。训练后的 VioLA 可以实现所有 ASR、MT、TTS、级联 speech-to-text 翻译和级联 speech-to-speech 翻译任务。

### 模型框架

#### 多任务自回归 codec LM

模型核心是自回归 codec Transformer LM，架构如下：
![](image/Pasted%20image%2020241010224157.png)

包含：
+ embedding encoding 模块
+ Transformer decoder 模块
+ prediction layer

还引入 language ID 和 task ID 区分不同语言和任务。

模型通过多任务优化，使用 ASR 数据（$\mathbf{A}^{\text{asr}, 1:8}, \mathbf{S}^{\text{asr}}$）、MT 数据（$\mathbf{S}^{\text{mt,s}}, \mathbf{S}^{\text{mt,t}}$）和 TTS 数据（$\mathbf{S}^{\text{tts}}, \mathbf{A}^{\text{tts}, 1}$）作为图中的输入和输出。输入包括 semantic 和 acoustic tokens，首先通过 embedding encoding，然后用 Transformer decoder 建模 semantic 和 acoustic tokens 的关系。输入的所有 token 可以 attend 所有位置的输入 token，输出中的每个 token 可以 attend 前面的输出和所有输入 token。最后使用 prediction layer 得到离散 vocabulary 的索引。

### Pre-Net 和 Pos-Net

使用 embedding encoding 模块更好地表示 acoustic tokens 和 semantic tokens。对于 semantic tokens $\mathbf{S} \in \mathbb{R}^{T_S}$，包括 $\mathbf{S}^{\text{asr}}, \mathbf{S}^{\text{mt,s}}, \mathbf{S}^{\text{mt,t}}, \mathbf{S}^{\text{tts}}$，使用 semantic embedding matrix $\mathbf{E}_S$ 得到的 semantic embedding。同样，language ID ${L} \in \mathbb{R}^1$ 通过 ${E}_L$ 嵌入到 semantic embedding 中：
$$\boldsymbol{X}_\mathbf{S}=\mathbf{E}_S(\boldsymbol{S})+\mathbf{E}_L(L)$$
其中 $\mathbf{X}_\mathbf{S} \in \mathbb{R}^{T_S \times D}$，如 $X^{\text{asr}}_S, X^{\text{mt,s}}_S, X^{\text{mt,t}}_S, X^{\text{tts}}_S$，是通过 embedding encoding 模块得到的 semantic 特征，$D$ 是 hidden states 的维度。

对于 acoustic tokens $\mathbf{A}^{\text{asr}, 1:8}$ 和 $\mathbf{A}^{\text{tts}, 1}$，通过 acoustic embedding matrix $\mathbf{E}_{A, i}$ 得到 acoustic embedding，其中 $i$ 表示第 $i$ 层 acoustic tokens。此外，将所有层的 acoustic embedding 平均得到 mixed embedding feature 表示多层 acoustic embedding。最后将 mixed embedding feature 和 single embedding feature 输入单向 LSTM 得到 acoustic 特征 $\mathbf{X}_A \in \mathbb{R}^{T_A \times D}$：
$$\begin{aligned}\boldsymbol{X}_\mathrm{A}^\mathrm{asr}&=\mathrm{LSTM}\left(\sum_{i=1}^8\mathrm{E}_{A,i}(\boldsymbol{A}^\mathrm{asr,i})/8\right)+\mathrm{E}_L(L)\\\boldsymbol{X}_\mathrm{A}^{\mathrm{tts},1}&=\mathrm{LSTM}\left(\mathrm{E}_{A,1}(\boldsymbol{A}^{\mathrm{tts},1})\right)+\mathrm{E}_L(L)\end{aligned}$$

对于 speech synthesis 任务，上述自回归 codec LM 只得到第一层 acoustic tokens。参考 VALL-E，采用另一个非自回归 codec Transformer LM 生成所有层 acoustic tokens。与原始的非自回归 codec LM 不同，这里的非自回归 codec LM 引入 LSTM 模块编码 acoustic tokens。

### 训练目标

多任务自回归 codec LM 通过 speech-to-text 识别、text-to-text 翻译和 text-to-speech 合成任务进行优化。引入额外的 task ID（如 basr、bmt、btts）完成不同任务。

speech-to-text 识别任务需要根据原始语音的 acoustic tokens $A^{\text{asr}, 1:8}$ 预测 phoneme 序列 $\mathbf{S}^{\text{asr}}$。训练采用 teacher forcing 和自回归策略，ASR task ID 为 $b_\text{asr}$：
$$\begin{aligned}
\mathcal{L}_{asr}& =p\left(\boldsymbol{S^\mathrm{asr}}|\boldsymbol{A^\mathrm{asr}},\boldsymbol{b_\mathrm{asr}};\theta\right)  \\
&=\prod_{t=0}^{T_\mathrm{S}}p\left(S_t^\mathrm{asr}|\boldsymbol{A}^\mathrm{asr},1:8,b_\mathrm{asr},\boldsymbol{S}_{<t}^\mathrm{asr},\theta\right)
\end{aligned}$$
acoustic tokens、task ID 和 semantic tokens 拼接后输入到 LM。

text-to-text 翻译任务是将源 phoneme 序列 $\mathbf{S}^{\text{mt,s}}$ 翻译为目标语言 phonemes $\mathbf{S}^{\text{mt,t}}$。类似 speech recognition，MT 任务通过最大化以下概率进行优化：
$$\begin{aligned}
\mathcal{L}_{mt}& =p\left(\boldsymbol{S^\mathrm{mt,t}}|\boldsymbol{S^\mathrm{mt,~s}},b_\mathrm{mt};\theta\right)  \\
&=\prod_{t=0}^{T_\mathrm{S}}p\left(S_t^{\mathrm{mt,~t}}|\boldsymbol{S^{\mathrm{mt,s}}},b_{\mathrm{mt}},\boldsymbol{S_{<t}^{\mathrm{mt,t}}},\theta\right)
\end{aligned}$$

text-to-speech 合成任务通过自回归方式生成第一层 quantized acoustic tokens $\mathbf{A}^{\text{tts}, 1}$，如下：
$$\begin{aligned}
\mathcal{L}_{tts}& =p\left(\boldsymbol{A^\mathrm{tts,1}}|\boldsymbol{S^\mathrm{tts}},b_\mathrm{tts};\theta\right)  \\
&=\prod_{t=0}^{T_\mathrm{A}}p\left(A_n^{\mathrm{tts},1}|\boldsymbol{S}^{\mathrm{tts}},b_{\mathrm{tts}},\boldsymbol{A}_{<t}^{\mathrm{tts},1},\theta\right)
\end{aligned}$$

多任务学习框架优化自回归 codec LM：
$$\mathcal{L}=\mathcal{L}_{asr}+\mathcal{L}_{mt}+\mathcal{L}_{tts}$$
这三个任务同时训练。

### 推理

训练后，模型可以用于 ASR、MT、TTS 任务。speech synthesis 用相同或不同语言的 speech prompts，对于一个文本合成五个音频，根据下面的策略进行选择：
+ 策略1：根据 speaker similarity（SS）分数选择得分最高的 utterance
+ 策略2：根据 speaker similarity 和 word error rate（WER）分数的组合

## 实验（略）
