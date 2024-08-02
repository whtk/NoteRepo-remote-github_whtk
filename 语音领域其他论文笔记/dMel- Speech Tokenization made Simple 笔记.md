> Apple，2024.7 preprint

1. 现有的语音 tokenization 方法要么建模 semantic token，从而丢失 acoustic 信息；要么建模 acoustic token，从而丢失 semantic 信息
2. 且多种 token 类型使得架构复杂，需要额外的预训练
3. 本文表明，离散 mel-filterbank channels 得到的简单表示（dMel）比其他现有的语音 tokenization 方法表现更好
4. 采用 transformer decoder-only 架构进行 speech-text 建模，可以用于 ASR 和 TTS 任务

## Introduction

1. 语音需要有效的 tokenization 方法将连续信号离散化
2. 现有的 tokenization 方法分为 semantic tokens 和 acoustic tokens：
    1. semantic tokens 从自监督预训练中提取，语音信号先编码为 representations，然后用 k-means 方法聚类为 semantic tokens
    2. acoustic tokens 从音频压缩模型中获得，训练用于将语音信号压缩为 codebook indices，优先 acoustic reconstruction 但丢失 semantic 信息
3. AudioLM 组合两个 token，是三阶段模型：semantic modeling、coarse acoustic modeling 和 fine acoustic modeling，设计 coarse-to-fine 模型以匹配 acoustic tokens 的 residual 结构，但多阶段结构复杂且训练和推理慢
4. 本文提出 dMel，直接将 mel-filterbanks energies 离散化为 ordinal bins，发现离散化 Mel spectrograms 对 mel-filterbank vocoders 重构波形影响不大，采用两种 vocoders 重构 mel-filterbanks 和 dMel，发现 WER 相似，表明离散化 Mel 对信息内容影响有限
5. dMel 保留了频率和强度信息，也保留了 semantic 和 acoustic 信息，无需额外 tokenization 或预训练模型
    1. Mel-filterbanks 是可解释的表征，保留了 semantic 和 acoustic 信息，离散化影响有限
    2. dMel 是基于原始 acoustic space 的 model-free 表征，可以被任何 mel-filterbank vocoder 转换为波形，不像其他 tokenization 方案，表征与 encoder 和 decoder 紧密耦合
    3. dMel 的不同 channel 之间没有 coarse-to-fine acoustic tokens 的复杂层次依赖，可以在每帧使用简单 decoder-only transformer 架构独立建模
6. 使用 dMel，可以用单个 decoder-only 的模型，在 ASR 和 TTS 任务上取得好结果

## 方法
unified transformer decoder-only 模型架构如图：
![](image/Pasted%20image%2020240801210755.png)

### dMel


记 tensor 为 $\mathbf{X}$，$\mathbf{X}_{i,...}$ 表示 tensor $\mathbf{X}$ 的 $(i, ...)$-th component。speech tokenizer 输入为 speech signal $x$，计算 mel-filterbanks $\mathbf{M}$：
$$\mathbf{M}=Mel(\mathbf{x})$$

其中 $Mel(·)$ 表示计算 mel-filterbanks 的函数，$\mathbf{M} \in \mathbb{R}^{T \times N}$，$N$ 是 filterbanks 数量，$T$ 是帧数。

将 mel-filterbanks $M$ 离散化为 speech tokens，采用 codebook $\mathbf{C}$。本文采用线性离散化，codebook $\mathbf{C} \in \mathbb{R}^{2^K}$，值均匀分布在 mel-filterbanks 值范围内：
$$m=\min_{t,i}(\mathbf{M}_{t,i}),\quad M=\max_{t,i}(\mathbf{M}_{t,i})\quad\delta=\frac{M-m}{2^K},\\\mathbf{C}=\begin{bmatrix}m,\:m+\delta,\:m+2\delta,\:\ldots,\:m+(2^K-1)\delta\end{bmatrix}.$$

实际中，计算整个数据集中 mel-filterbanks 的最小值 $m$ 和最大值 $M$ 来定义 codebook $\mathbf{C}$。然后将每个 channel $i=1...N$ 的每个频率帧 $t=1...T$ 的幅度 $\mathbf{M}_{t,i}$ 映射到 codebook $\mathbf{C}$ 的 bin index：
$$\mathbf{S}_{t,i}=\text{Discretize}(\mathbf{M}_{t,i})=\text{argmin}_j|\mathbf{M}_{t,i}-\mathbf{C}_j|$$

其中 $\mathbf{S} \in \mathbb{B}^{T \times N}$ 表示离散化 mel-filterbanks（dMel），$\mathbb{B} = \{j|j = 1, 2, 3, \ldots 2^K\}$，$\mathbf{S}_t \in \mathbb{B}^N$ 是第 $t$ 个 speech token。由于 codebook $\mathbf{C}$ 有 $2^K$ 个不同值，因此 bin 数量 $|\mathbb{B}| = 2^K$，每个 speech token 用 $N \cdot K$ 位表示，每 $K$ bit 表示一个频率 channel。

为了从 speech tokens $\mathbf{S}$ 重构 speech signal $\mathbf{x}$，首先通过 codebook $\mathbf{C}$ 将 bin indices 转换回 mel-filterbanks：
$$\hat{\mathbf{M}}_{t,i}=\mathbf{C}_{\mathbf{S}_{t,i}}.$$

然后用 vocoder 将重构的 mel-filterbanks $\hat{\mathbf{M}}_{t,i}$ 转到时域信号 $\mathbf{x}$。vocoder 独立训练，不是 transformer decoder-based 模型的一部分。

### Unified Speech-Text Transformer Decoder

采用 unified transformer decoder-only 模型，输入 speech 和 text tokens，生成目标序列的 output tokens。模型在 speech 和 text pair 数据集上端到端训练，学习 ASR 和 TTS 任务的联合表示。

对于文本数据，采用 character-level tokenizer 将输入文本转为 text tokens 序列。text tokens 经过 embedding 层，$Embed(·) : \{j|j = 1, 2, 3, \ldots L\} \rightarrow \mathbb{R}^D$，$D$ 是 embedding 维度，$L$ 是词表大小。speech token embedding 维度与 text token embedding 维度相同。
> 使用 character-level tokenizer 可以减小词表长度，提高模型泛化能力，character tokens 可以捕获对 ASR 和 TTS 任务有用的特征。

对于语音，采用 dMel speech tokenizer 将输入语音转为 speech tokens 序列。speech tokens $\mathbf{S} \in \mathbb{B}^{T \times N}$ 经过可学习的 embedding 层，$Embed(·) : \mathbb{B} \rightarrow \mathbb{R}^d$，和可学习的线性层，$Linear(·) : \mathbb{R}^{N \times d} \rightarrow \mathbb{R}^D$，得到 speech token 表征 $\mathbf{E} \in \mathbb{R}^{T \times D}$：
$$\begin{array}{rcl}\mathbf{E^{\prime}}_{t}&=&\mathsf{Concatenate}([\mathsf{Embed}(\mathbf{S}_{t,1}),\mathsf{Embed}(\mathbf{S}_{t,2}),\ldots,\mathsf{Embed}(\mathbf{S}_{t,N})])\\\mathbf{E}_{t}&=&\mathsf{Linear}(\mathbf{E^{\prime}}_{t}),\end{array}$$

其中 $\mathbf{E}_t$ 是 speech token 表征。对于每个时间帧 $t$，一个 speech token $\mathbf{S}_t$ 并行独立处理每个频率 channel $i$，然后所有频率 channel 的 embedding 堆叠形成帧 $t$ 的一个向量表示 $\mathbf{E^{\prime}}_t$。最后，speech token embeddings $\mathbf{E}_t$ 输入 decoder-only transformer 模型。

也比较了 HuBERT-KM 和 SpeechTokenizer。
> 这两个 tokenizer 的区别在于 codebook 大小和 codes 维度。对于 HuBERT-KM 和 SpeechTokenizer，speech tokens 通过可学习的线性层从其维度映射到 text embedding 维度 $D$，然后输入 decoder-only transformer 模型。

为了建模多说话人，将 speaker embeddings 作为 transformer decoder 的输入。speaker embedding 来自独立的 dvector 模型。speaker embeddings 通过可学习的线性层映射到 speech 和 text token embeddings 维度 $D$。
> 对 ASR 任务，speaker representation 可选，对 TTS 任务是必须的。在训练时，对 text-to-speech 使用，对 speech-to-text 忽略。

transformer decoder 在 speech-text pair 数据集上端到端训练：
    + 对于 TTS，输入序列由 speaker embedding、text tokens 和 speech tokens 构成
    + 对于 ASR ，输入序列由 speech tokens 和 text tokens 构成
    + 都采用 causal masking，基于前面的 token 预测下一个 token
    + 损失是，预测的 token 和 GT token 之间的交叉熵
    + 对于 ASR 任务跳过 speech tokens，对于 TTS 任务跳过 text tokens

> 注意，dMel tokenizer 的所有频率 channel 在时间帧 $t$ 预测独立并并行。

为了捕获输入序列 token 间的相对距离，采用 multiplicative relative positional embedding RoPE。模型可以学习 speech tokens、text tokens 和 speaker embeddings 之间的位置关系。
> 对于 positional embeddings，不区分 text、speech 和 speaker embeddings，在所有的 token 中都是 global positions。

此外，由于音频帧有强局部相关性，高度冗余，从而很难生成长文本，即所谓的 exposure bias。为了减轻 exposure bias，采用 span-masking，随机 mask 多个 speech frames。模型在 masked context 上预测下一个 token。这种 context-masking 策略有助于模型在缺失信息的情况下生成准确的 speech tokens，提高鲁棒性和泛化性。
> 其强制模型关注 text，而不是复制先前推理的 speech tokens
> 也发现 span-masking text tokens 可以提升 ASR 任务

## 实验

使用开源数据集进行实验：
    + LibriSpeech：英语语音记录（960h，16kHz），各种说话人（~2k）   
    + LibriTTS：从 LibriSpeech 衍生，适当的句子拆分、文本规范化和保持 24kHz，约 500h
    + VCTK：44 小时英语语音（108 说话人）
    + LJSpeech：16kHz 单说话人英语音频，来自 LibriVox

> LibriSpeech 用于训练 ASR、TTS 和 joint 模型，LibriTTS、VCTK 和 LJSpeech 用于训练 TTS 模型

训练三种不同大小的 decoder-only transformers：Small、Base 和 Large。默认使用 Base 模型。所有模型使用 pre-LayerNorm，residual、attention 和 embedding dropout 设置为 0.1，positional embedding dropout 设置为 0.3。dMel 每个 channel 使用 16 个离散 bin，text 使用 character 词表，speaker embedding dvector 维度为 512。

结果见原论文。
