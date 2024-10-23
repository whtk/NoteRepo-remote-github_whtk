> ICLR 2024，CUHK、CMU、MSRA、ZJU

1. 提出 UniAudio，采用 LLM 得到多种类型的音频（语音、声音、音乐、歌唱）：
    1. 首先对所有音频和其他条件模态进行 tokenization
    2. 将 source-target pair 拼接成单个序列
    3. 使用 LLM 预测下一个 token
2. 提出一个多尺度 Transformer 来处理 RVQ 进行 tokenization 中得到的长序列
3. 数据集用 165K 小时音频，参数 1B
4. 实验表明，UniAudio 在 11 个音频生成任务中表现良好

> 同样地方式，也是 多任务 + 多尺度 Transformer 架构

## Introduction

1. 提出 UniAudio，基于不同的输入模态采用 LLM 技术生成多种类型音频（语音、声音、音乐、歌唱），功能如下：
    1. 音频和其他输入模态都被 tokenized 为离散序列，对于音频，使用一个通用的 codec 进行 tokenization
    2. 拼接 source-target pair 为单个序列
    3. 使用 LLM 预测下一个 token
    4. 设计 多尺度 Transformer 架构，分别建模帧间和帧内的相关性：
        1. global Transformer 建模 inter-frame 相关性
        2. local Transformer 建模 intra-frame 相关性
2. UniAudio 的构建过程分为两个阶段：
    1. 首先，UniAudio 在多个音频生成任务上联合训练以获得足够的先验
    2. 通过微调，训练模型可以支持更多 unseen 音频生成任务

## UniAudio

### Tokenization

#### Audio

UniAudio 将所有类型的音频使用 codec 来 tokenize 为单一的统一模态。音频信号可以表示为 $\mathbf{x}\in[-1,1]^{d*f_s}$，其中 $d$ 为持续时间，$f_s$ 为采样率。音频 codec 通过 encoder-decoder 结构和量化模块压缩 $\mathbf{x}$ 并恢复为 $\hat{\mathbf{x}}$：
$${\mathrm{h}}=\mathrm{Encoder}(\mathbf{x})\in\mathcal{R}^{T*L};\quad{\hat{\mathrm{h}}}=\mathrm{Quantization}(\mathrm{h});\quad{\hat{\mathbf{x}}}=\mathrm{Decoder}(\mathbf{h})$$
其中 $T$ 表示 encoder 下采样后的音频帧数，$L$ 为特征维度。给定任意帧的 hidden output $\mathbf{h}_t$，通过 RVQ 生成整数向量 $\mathbf{z}_t = [z_{t}^1,...,z_t^{n_q}]$，其中 $n_q$ 为 VQ 层数。每个元素 $z_t^k$ 是所有预训练和固定的第 $k$ 层quantizer vector $\{\mathbf{q}^*_k\}$ 中与残差的最小 L2 距离的索引。得到离散的表征 $\mathbf{z}_t$ 后，通过 decoder 重构音频 $\hat{\mathbf{x}}_t$：
$$z_{t}^{k}=\arg\operatorname*{min}\operatorname{Distance}(\mathbf{h}_{t}-\sum_{j=1}^{k-1}\mathbf{q}_{j}^{z_{t}^{j}},\mathbf{q}_{k}^{m});\quad{\hat{\mathbf{h}}}_{t}=\sum_{j=1}^{n_{q}}\mathbf{q}_{j}^{z_{t}^{j}};\quad1\leq k\leq n_{q}$$

所有音频帧的离散表示 $\mathbf{z}\in\mathbf{Z}^{T\times n_q}$ 是一个矩阵，将其简单地展平为一个序列，其中每个帧的 $n_q$ 元素是连续的。实验中设置 $n_q=3$。

#### 其他模态

其他输入模态的 tokenization 如下：

+ Phoneme：有多种来源：
    1. 只有文本时，通过发音字典的文本到音素映射获得音素序列；
    2. 只有语音时，通过 DNN-HMM 系统的 beam search 获得带有持续时间信息的音素序列；
    3. 有文本和语音时，通过 DNN-HMM 系统的强制对齐获得带有持续时间信息的音素序列

+ MIDI：MIDI 用于歌声合成任务，包含 F0 和持续时间信息。使用持续时间信息展平 F0 序列，得到帧级 F0 序列

+ Text：文本表征 为 从预训练文本 LLM 中得到的 连续 embeddings，其包含丰富的语义信息。

+ Semantic Token：从 SSL 模型的连续 embeddings 中通过 K-means 聚类得到离散表征

### 统一任务格式

所有任务都可以统一为序列建模任务，交由 LLM 处理：target audio 和 conditions 先转换为序列，然后拼接为 `[conditions, target]` 格式。

支持 11 种任务，每个任务的序列格式如下：
![](image/Pasted%20image%2020241023161015.png)

由于每个任务的独特配置，一些条件子序列在 tokenization 期间需要特定的预处理操作。对于音频，这些操作主要用于破坏数据，如在 tokenization 前在原始音频中添加噪声、混响和混合其他说话人的语音。

为了避免歧义，插入一些特殊的离散 token（用  `<>` 括起来）：
+ 序列开始和结束
+ 某个模态的子序列开始和结束
+ 任务标识符
> 例如，对于一个 text-to-sound 任务序列，生成基于文本描述的目标音频，整个序列如下：`<start> <sound_task> <text_start> text_sequence <text_end> <audio_start> audio_sequence <audio_end> <end>`


### 多尺度 Transformer

将离散音频 token 建模为展开的序列，但是这样的序列长度为 $T\times n_q$，导致 Transformer 空间复杂度很大的。因此提出多尺度 Transformer，分别处理帧间和帧内相关性。架构如下：
![](image/Pasted%20image%2020241023161657.png)

多尺度 Transformer 将 patch（即每个连续的 $n_q$ token）作为全局建模单元，然后在每个 patch 内部处理 token。
> 注意：全局和局部 Transformer 都是因果的。

对于 audio token 序列，每个 patch 包含 $n_q$ 个连续的音频 token，表示一帧：
+ 首先，每个 patch 在 embedding 阶段由对应 embeddings 的和向量表示
+ 然后 global Transformer 逐帧预测：预测帧 $x_t$ 时，输出包括帧 $x_{t-1}$ 和所有先前内容的连续表征，后续被用于 local Transformer
+ 给定 hidden representation $\mathbf{h}_t$，使用 global Transformer 对应于帧 $x_{t-1}$ 的 hidden output 预测帧 $x_t$ 的离散 token $z_t$。
    + 每个 token $z_k^t$ 都自回归地依赖于其先前 token $\{z_t^j|j<k\}$，使用 local Transformer 自回归预测 patch 序列 $\mathbf{z}_t$。
    + 相当于 global Transformer 产生的 vector 作为 patch-level context

多尺度 Transformer 架构也适用于除音频外的离散和连续序列。对于除音频外的所有离散 token（音素、语义、MIDI 和特殊 token），每个 token 有独立的语义，对应一个 patch。把这些离散 token 重复 $n_q$ 次以填充每个 patch。连续文本 embeddings 也重复 $n_q$ 次。此外，其 embedding 过程替换为 linear transform，而 local Transformer 的预测目标是连续的特殊 token `<continuous_token>`。

多尺度 Transformer 可以降低计算复杂度：
+ global Transformer 的等效序列长度从 $T\times n_q$ 降低到 $T$，使得全局建模成本与 $n_q$ 无关，因此可以采用更大的 $n_q$
+ intra-patch 计算转移到 local Transformer。local Transformer 只处理短序列（固定为 $n_q$ 的长度），参数比 global Transformer 少

## 实验（略）
