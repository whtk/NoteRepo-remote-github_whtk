> preprint 2024.10，开源非营利性组织 Kyutai

1. 提出 Moshi，是一个 speech-text foundation model，实现全双工的语音对话
2. 现有的语音对话需要各种 pipeline，包括 VAD、ASR、文本对话和 TTS
3. Moshi 将语音对话看成是 speech-to-speech 生成，其从文本的 LM 出发，生成 speech tokens，同时并行建模自己的语音和用户的语音
4. 进一步将 hierarchical semantic-to-acoustic token 生成扩展到首先预测时间对齐的文本 token 作为音频 token 的 prefix

> 重点在于整套 text+audio token 的组织方式。

## Introduction

1. 现有的对话系统还是不如自然对话：
    1. pipeline 太复杂，延迟很高
    2. 语言理解和生成是在 textual domain，忽略了 non-written 信息
    3. 模型基于 turn-based，其假定对话是一系列明确定义的单说话人片段，无法处理打断、重叠的语音和 backchanneling（非打断的插话）
2. 提出 Moshi，是一个 speech-text foundation model 和实时语音对话系统，采用一个更小的 audio LM 来增强 text LLM，直接在 audio domain 处理输入和输出，同时利用 text LLM 的知识和推理能力；设计了一个 streaming、hierarchical 架构；引入了第一个 multi-stream audio LM，显式地将输入和输出音频流联合成两个自回归 token stream 来消除 speaker turn 的概念，从而允许训练模型在具有任意动态的自然对话上，包括重叠和打断；最后得到的模型是第一个全双工的实时对话 LLM；贡献如下：
    + 提出 Helium，一个 7B 参数的 text LLM，使用 2.1T tokens 数据预训练
    + 训练 Mimi codec，，使用 RVQ 将 audio 转换为 Moshi 预测的离散 token；将语义信息蒸馏到 acoustic tokens 的第一级，并引入改进的训练技巧
    + 提出 Moshi，audio LM 架构，将 Helium 与一个较小的 Transformer 模型结合起来，以 hierarchical 和 streaming 的方式预测 audio tokens；将其扩展到并行模拟多个 audio stream，实现全双工对话
    + 引入 Inner Monologue，训练和推理方式，通过在 audio tokens 之前预测时间对齐的文本 tokens，提高生成语音的真实性和语言质量


## 相关工作（略）

## 模型

### 概览

架构如图：
![](image/Pasted%20image%2020241019104516.png)

基于 Helium LLM，采用 Inner Monologue 训练和推理方式，使得模型可以利用文本模态的知识，同时实现 speech-to-speech；设计为 multi-stream 架构，可以同时说话和听取用户，无需显式建模 speaker turns；提出 Mimi codec，通过 RVQ 和 knowledge distillation 将语义和 acoustic 信息合并为单一 tokenizer；为了联合模拟 Moshi 和用户的 audio streams，以及 Moshi 的 text tokens，使用 Depth Transformer 来支持 streaming 推理。

### Helium Text LLM

Helium 是一个基于 Transformer 架构的自回归语言模型，对原始架构进行了如下修改：
+ 采用 RMS normalization
+ 采用 RoPE 位置编码
+ 采用 FlashAttention 实现高效训练
+ 将 feed-forward blocks 架构改为 Gated Linear Units，使用 SiLU 作为门控函数

Tokenizer 基于 SentencePiece 的 unigram 模型，包含 32,000 个元素。使用 AdamW 优化器，固定学习率，然后使用余弦学习率衰减。

从高质量数据源（如 Wikipedia、Stack Exchange 和大量科学文章）开始，再用 CommonCrawl 爬取数据。数据预处理包括：
+ 去重：从 WET 文件开始，只包含 CommonCrawl 提取的网页文本。其包括很多 boilerplate，是对每个 shard（每次爬取有 100 个 shard）进行去重，去除这些 boilerplate。
+ 语种识别：在去重后，使用 fastText 进行语种识别，只保留英文数据。
+ 质量过滤：训练 fastText 分类器，对高质量数据源和随机 CommonCrawl 网页的行进行分类。分类器有 9 个类别，对应不同的高质量来源，如 Wikipedia 或 Wikibooks，以及 StackExchange 的子集。计算每行的平均分数（按长度加权）得到聚合分数，保留分数高于某个阈值的文档。

### Audio Tokenization

Mini codec 是一个带有离散 bottleneck 的自编码器，将 waveform 离散化为 audio tokens（acoustic tokens）semantic tokens 与 acoustic tokens 不同，不可以重构音频，但与内容相关。从而可以在没有文本条件是，使用 semantic audio tokens 作为 acoustic tokens 的 prefix 来生成可理解的语音。然而，这种混合 tokenization 方法不适用于实时生成。semantic tokens 只能离线计算。而且用不同的 encoder 生成 acoustic 和 semantic tokens 增加了计算量。因此，Mimi 使用 distillation 将非因果的语义信息转移到 causal model 生成的 tokens 中，实现流式编解码。

baseline 参考了 [SoundStream- An End-to-End Neural Audio Codec 笔记](../语音领域其他论文笔记/SoundStream-%20An%20End-to-End%20Neural%20Audio%20Codec%20笔记.md) 和 [EnCodec- High Fidelity Neural Audio Compression 笔记](../语音领域其他论文笔记/EnCodec-%20High%20Fidelity%20Neural%20Audio%20Compression%20笔记.md)，包含 SeaNet autoencoder 和 RVQ。encoder 将波形投影到 latent representation，decoder 将 latent representation 投影回 24kHz 音频。latent space 通过 RVQ 离散化。baseline 模型使用重构和对抗损失训练。Mimi 的主要改变如下：
+ 在 bottleneck 前后添加 Transformer 模块
+ Transformer 有 8 层，8 heads，RoPE 位置编码，250 帧的有限上下文，GELU 激活函数，模型维度 512，MLP 维度 2048
+ 使用 LayerScale 初始化，对角线值为 0.01
+ 两个 Transformer 使用因果 masking，从而兼容 streaming 推理

Mimi 是因果的，可以流式编解码。初始帧大小和总步长都是 80ms，即给定 80ms 的音频帧，Mimi 输出一个 latent timestep，可以解码为 80ms 的输出音频。

优化：使用 AdamW 优化器，学习率 8x10−4，动量衰减 0.5，梯度平方衰减 0.9，权重的指数移动平均衰减 0.99。batch size 128，随机 12s 窗口训练 4M steps，Transformer 的上下文限制在 10s。

量化率：8 个 quantizers，每个 codebook 大小为 NA = 2048。12.5Hz，比特率 1.1kbps。latent dimension 512，RVQ 前后投影到 256 和 512。使用 quantizer dropout 提供比特率可扩展性。训练时只有 50% 的时间应用量化。

仅使用对抗训练，去除重构损失，只保留特征损失和判别器损失。去除重构损失会显著降低客观指标，但主观评估音频质量有显著提升。

与 SpeechTokenizer 类似，从 WavLM 中蒸馏语义信息到 RVQ 的第一级。WavLM 将 16kHz 波形投影到 1024 维度的 embeddings，Mimi 将 24kHz 波形投影到 512 维度。训练时，通过下采样输入波形到 16kHz，计算 WavLM embeddings，然后平均池化到 12.5Hz。线性投影到第一个 RVQ 级的输出，与 decoder 的实际 embedding 平行。计算第一个 quantizer 的输出和转换后的 WavLM embeddings 之间的余弦距离进行蒸馏。蒸馏损失与重构和对抗损失冲突，显著提高了第一个 quantizer 的语音区分度，但也对音频质量产生负面影响。提出 split RVQ，将语义信息蒸馏到普通 VQ，然后并行应用 7 级 RVQ。两者的输出求和，这样两者都可以用于重构，去除 acoustic 信息应该保留在 semantic quantizer 的残差中的约束。

### Generative Audio Modeling

下面拓展 Helium 模型，使其支持 Mimi codec 的 audio tokens。同时模拟两个 audio stream，一个代表用户，一个代表系统。最后使用 Inner Monologue 联合建模文本和音频模态。

令 $U \in \{1,...,N\}^S$ 为离散随机序列，$N$ 为选择的个数，$S$ 为序列长度。令 $U_0 = 0$ 为一个确定的初始 token 值。自回归建模是通过估计所有步骤 $1 \leq s \leq S$ 的条件分布 $P[U_s|U_0,...U_{s-1}]$ 来估计联合分布 $P[U_1,...,U_S]$。

建模语音时，使用 tokenized text 比 audio tokens 可以实现更紧凑的表征。
> 使用 Mimi codec，Q = 8 codebooks，帧率 12.5hz，需要 100  step 来生成 1s 的音频。5 分钟音频需要 30,000 个 step，无法实现 streaming。而语音可以用每秒 3 到 4 个文本 token 表示。

本文需要建模多个子序列。将这些子序列堆叠为 $V_{s,k}$，其中 $1 \leq s \leq S$，$1 \leq k \leq K$。对于每个 $1 \leq s \leq S$，$1 \leq k \leq K$，$V_{s,k} \in \{1,...,N_k\}$，$N_k$ 是第 $k$ 个子序列的选择个数。可以将 K 个序列展平为一个序列，预测数量增加 K 倍。

RQ-Transformer 由两个 Transformer 模型组成，一个 Temporal Transformer 和一个较小的 Depth Transformer，如图：
![](image/Pasted%20image%2020241020101239.png)

记 $Tr_\text{Temp}$ 为 Temporal Transformer，$Tr_\text{Depth}$ 为 Depth Transformer。对所有 $s \leq S$，记 $V_s = (V_{s,1},...,V_{s,K})$ 为所有子序列在步骤 $s$ 的联合值。对于给定的 step $1 \leq s \leq S$，Temporal Transformer 将 $(V_0,...,V_{s-1})$ 映射到一个 temporal context vector：
$$z_{s}={\mathrm{Tr}}_\mathrm{Temp}\left({{V}}_{0},\dots,{{V}}_{s-1}\right)\in\,{\mathbb{R}}^{d}$$

如果进一步取一个子序列索引 $1 < k \leq K$，Depth Transformer 将 $z_s$ 与 $(V_{s,1},...,V_{s,k-1})$ 映射到 logits estimate：
$$l_{s,k}=\mathrm{Tr}_{\mathrm{Depth}}(z_{s},V_{s,1},\dots,V_{s,k-1})\in\mathbb{R}^{N_{k}}.$$

定义 $l_{s,1} = \mathrm{Lin}(z_s) \in \mathbb{R}^{N_1}$，其中 $\mathrm{Lin}$ 是一个专门的线性层。训练 $Tr_\text{Temp}$，$Tr_\text{Depth}$ 和 $\mathrm{Lin}$，使得 $\mathrm{softmax}(l_{s,k})$ 是 $V_{s,k}$ 在前一步的所有子序列和当前步的前一子序列的条件分布的近似：
$$\begin{array}{r l}
\{\mathrm{softmax}(l_{s,1})\}&\approx\mathbb{P}\left[V_{s,1}|V_{0},\dots,V_{s - 1}\right]\\
\{\mathrm{softmax}(l_{s,k})\}&\approx\mathbb{P}\left[V_{s,k}|V_{0},\dots,V_{s - 1}, V_{s,1},\dots,V_{s,k - 1}\right]\quad\mathrm{if~}k>1.
\end{array}$$

Temporal Transformer 的步数始终等于 $S$，而不是 $K \cdot S$，Depth Transformer 的步数最多为 $K$。Temporal Transformer 在每个步骤 $s$ 输入为 $K$ 个 embedding table 的和，表示 $V_{s-1}$ 的值。对于 $1 < k \leq K$，Depth Transformer 输入为 $z_s$ 和单个 embedding $V_{s,k-1}$ 的和。

#### Audio Modeling

Mimi 输出 $Q$ 个子序列，每秒 12.5 个步骤的音频。记这些序列为 $A_{t,q} \in \{1,...,N_A\}$，$1 \leq t \leq T$，$T = 12.5 \cdot \text{duration}$，$1 \leq q \leq Q$，$Q = 8$。将音频子序列插入 RQ-Transformer 模型中。
> 注意：第一个 codebook 对应 semantic 信息，其他 codebooks 对应 acoustic 特征。

Acoustic delay：在 semantic 和 acoustic tokens 之间的引入轻微延迟可以得到更稳定的生成。因为可以减少给定时间步的子序列之间的依赖，从而使用较弱的模型来近似联合分布 $P[V_{s,k}|V_0,...,V_{s-1}]$。

在 semantic 和 acoustic 特征之间引入 1 或 2 步的延迟使得 Temporal Transformer 可以建模 semantic 和 acoustic 特征之间的相互依赖。对所有 step $s$，有：
$$\begin{array}{l l}
\{V_{s,1}=A_{s,1}\\
\{V_{s,q}=A_{s-\tau,q}\qquad\mathrm{if}\ s\ge\tau + 1,q>1\\
\{V_{s,q}=0\qquad\mathrm{if}\ s< \tau + 1,q=1
\end{array}$$

本文的 semantic token 与 acoustic tokens 一起生成，从而可以流式建模 semantic 和 acoustic tokens。

#### Multi-stream Modeling

模拟两个说话人的对话：给定两个音频流 $(A_{t,q})$ 和 $(A^\prime_{t,q})$，两者都进行 acoustic delay，然后拼接得到 $V$。
> 具体工作时，$A$ 表示 Moshi，$A^\prime$ 表示用户。

#### Inner Monologue

通过建模 Moshi 读出来的语音的文本表征可以提高生成语音质量。定义 text stream $W \in \{1,...,N_W\}^T$，其通过将 语音对应的文本使用 SentencePiece tokenizer 得到。将 $W$ 插入 $V$ 的第一个子序列，作为生成 semantic tokens 的 prefix。
> 这里不使用用户的文本，因为实时获取用户的文本很困难，依赖外部 ASR 系统。

对齐文本和音频 tokens：将文本 tokens 与音频 tokens 对齐到 12.5Hz 的帧率。通过 Whisper 得到 word-level timestamp，将文本中的第 $i$ 个单词映射到第 $n_i$ 个文本 tokens $w_{i,j}$，$j \leq n_i$，以及开始索引 $t_i$，定义为开始时间戳除以 12.5 Hz。定义两个特殊 tokens：PAD 和 EPAD，不出现在任何单词 tokens 中。构建 $W$ 如：当一个单词开始时，$(W_t)$ 包含其文本 tokens 和直到下一个单词之前的 PAD。EPAD 插入到下一个单词之前，表示填充结束。
> 虽然不是必需的，但我们观察到这为模型提供了有用的指导，将结束单词的决策和下一个单词的选择分成两个步骤。

首先，序列 $(W_t)$ 初始化为 PAD tokens，$W_t \leftarrow \text{PAD}$。然后，对每个单词 $i$ 和其开始索引 $t_i$，更新 $W$：
$$\left.\left\{\begin{array}{ll}W_{t_i-1}&\leftarrow\mathrm{EPAD}\\W_{t_i+j}&\leftarrow w_{i,j}&\forall j\leq n_i.\end{array}\right.\right.$$

如果 $t_i = 1$，则在索引 1 处插入 EPAD，并移动文本 tokens。如果 EPAD token 会覆盖前一个单词的文本 token，则不插入 EPAD token。由于文本 tokens 比音频 tokens 更紧凑，通常在 $W_t$ 中单词之间没有重叠。在英语对话中，padding tokens 占大约 65%。

通过在文本序列 $(W_t)$ 和音频 tokens $(A_{t,q})$ 之间引入多一些延迟，可以控制 LM 在哪种模态下决定生成音频的内容：
+ 音频在文本之前，文本的内容将由前几步的音频决定。仅采样文本 tokens，输入 GT audio tokens 可以变成流式的 ASR 模型，且包含 precise word level alignment。
+ 文本在音频 tokens 之前，音频的内容由文本内容决定。给定适当填充的文本 tokens 序列，可以获得流式 TTS 模型。

Moshi 的联合序列建模：将 multi-stream 和 inner monologue 结合，定义最终的序列 $V$：
$$\begin{cases}V_{s,1}&=W_s&\text{aligned text tokens.}\\V_{s,2}&=A_{s,1}&\text{semantic tokens of Moshi.}\\V_{s,1+q}&=A_{s-\tau,q}&\mathrm{if}\quad s\geq\tau+1,1<q\leq Q&\text{delayed acoustic ok. of Moshi.}\\V_{s,1+Q+1}&=A_{s,1}^{\prime}&&\text{semantic tokens of }other.\\V_{s,1+Q+q}&=A_{s-\tau,q}^{\prime}&\mathrm{if}\quad s\geq\tau+1,1<q\leq Q&\text{delayed acoustic tok. of }other,&\end{cases}$$

总共 $K = 2Q + 1$ 个 streams，实验中 $Q = 8$。具体流程如下图：
![](image/Pasted%20image%2020241020112735.png)

Moshi 的推理：在训练时，模型在任何 step $s$ 输入 $0,V_1,...,V_{s-1}$，输出估计的概率分布 $\hat{V}_s(0,V_1,...,V_{s-1})$。在推理时，对所有与 Moshi 输出对应的子序列索引 $k$，从 $V_{s,k}$ 中采样：$k=1$ 对应 Moshi 的文本 tokens，$k \in \{2,...,2+Q\}$ 对应 Moshi 的音频 tokens。实际使用时，丢掉预测的 user audio，而使用实际的 user audio。
> Moshi 可以在任何时候 spear 和 listen，甚至同时进行。当 user 说话而 Moshi 保持沉默时，Moshi 的 audio tokens 解码为 “natural silence” 为几乎无声的波形，而非固定的、明确定义的值；同时，Moshi 的文本 stream 将填充 PAD tokens。

## 数据集和训练

### 文本数据

训练数据集由高质量数据源和 CommonCrawl 过滤的 web 数据混合而成。12.5% 数据来自以下精选数据源：Wikipedia、Wikibooks、Wikisource、Wikinews、StackExchange 和科学文章集合 pes2o。剩余 87.5% 数据来自 CommonCrawl。

### 语音数据

使用 700 万小时的无监督音频数据集，大部分包含英语语音。使用 large-v3 Whisper 进行转录。数据用于音频预训练阶段。所有音频重新采样为 24kHz 单声道。

为了实现 multi-stream，模型需要同时具备听和说的能力。使用 Fisher 数据集，包含 2000 小时的电话对话，每个对话人在不同的 channel 中录制，可以为 Moshi 提供 ground-truth separated stream。原始音频采样率为 8kHz，使用 AudioSR 将其上采样到 24kHz。

最后使用 170 小时对话（每个说话人来自单独的 channel）微调模型来提高质量。称为 supervised multi-stream 数据集。不直接在此数据集上训练 Moshi，而是用于训练真实的 multi-stream TTS 模型，并在真实对话文本上微调 Helium。

对于 Fisher 和最后的数据集，随机选择一个说话人作为主要说话人（即 Moshi 说话），另一个说话人作为第二个 audio stream。Fisher 的 text stream 只包含主要说话人的文本。


### 语音-文本 Instruct 数据（略）

### 训练阶段和超参数（略）

## 实验（略）

## 安全
