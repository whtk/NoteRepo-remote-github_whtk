> Google，ICLR 2024 reject。。。

1. 提出 SoundStorm，非自回归音频生成模型：
    1. 输入是 AudioLM 的 semantic token
    2. 依赖 bidirectional attention 和 confidence-based parallel decoding 生成对应于 codec 的 token
2. 相比于 AudioLM，在相同质量和更高一致性下 更快（在 TPU-v4 上 0.5 秒内生成 30 秒音频）
3. 可以生成长序列

## Introduction

1. 解决长音频 token 序列生成问题有三种方法：
    + 高效的 attention 机制
    + 非自回归、并行解码
    + 可以适应 codec 产生的 token 的自定义架
2. 作者认为，音频 token 序列的特殊结构有助于长序列音频建模
3. 提出 SoundStorm，解决长音频 token 生成问题：
    + 提出适应音频 token 的层次结构的架构
    + 是并行、非自回归、基于 confidence 的解码方法
4. SoundStorm 依赖于双向 attention-based Conformer，训练预测 SoundStream 生成的 masked 音频 token：
    + 输入端，将相同 SoundStream frame 的 token embeddings 相加，使得 self-attention 的内部序列长度与 SoundStream frame 数相同，与 RVQ 中的 quantizers 数无关
    + 输出 embeddings 由每个 RVQ level 的独立 heads 处理，预测 masked 目标 tokens
    + 推理时，给定条件，SoundStorm 从所有音频 token 开始，逐 level 填充 masked tokens，并行预测多个 token
5. SoundStorm 可以作为 AudioLM 的 acoustic generator，替代 stage two 和 stage three，生成音频比 hierarchical autoregressive acoustic generator 快，效果更好

## 相关工作（略）

## 方法

SoundStorm 输入是离散 token 序列，输出是 SoundStream tokens，可以解码为音频波形。假设 conditioning signal 与 SoundStream frames 对齐或可以上采样到相同帧率。例如，AudioLM、SPEAR-TTS 或 MusicLM 中使用的语义 token 序列。这里将 SoundStorm 作为 [AudioLM- a Language Modeling Approach to Audio Generation 笔记](AudioLM-%20a%20Language%20Modeling%20Approach%20to%20Audio%20Generation%20笔记.md) 中的 acoustic generator，替代 coarse 和 fine acoustic modeling stages。

> 并行解码就需要，输入和输出的维度相同，所以需要把 condition signal 进行上采样对齐。

### 架构

模型架构如图：
![](image/Pasted%20image%2020240607102930.png)

输入端，将 time-aligned conditioning tokens 与 SoundStream tokens 在 frame level 交错，相同 frame 对应的 embeddings 相加（包括 conditioning token 的 embedding），再送给 Conformer。Conformer 中的双向 self-attention 的序列长度由 SoundStream frames 数决定（通常每秒 50 帧），与 RVQ levels Q 数无关。输出端，使用 Q 个 dense layers 作为 heads 生成目标 SoundStream tokens。

### Masking

将 MaskGIT 的 masking 和 confidence-based 并行解码方案扩展到 RVQ 产生的 token 序列。此方法可以看作是按照 coarse-to-fine 的顺序策略。
> 好处在于：这种顺序遵循 RVQ 层次结构之间的条件依赖性，同时利用了 finer levels 的 tokens 在给定所有 coarser levels 的 tokens 时的条件独立性。finer levels 的 tokens 负责局部、细节的音频细节，因此可以并行采样而不损失音频质量。

设计训练的 masking 方案。为了实现 voice prompting，随机采样一个时间步 $t \in \{1, \ldots, T\}$，其中 $T$ 表示最大序列长度，此时间步之前的 token 不 mask。conditioning tokens 不 mask。设 $Y \in \{1, \ldots, C\}^{T \times Q}$ 表示 SoundStream tokens，其中 $C$ 表示每个 RVQ level 中的 codebook 大小。masking 方案如下：
+ 采样 prompt delimiter 时间步 $t \sim U\{0, T-1\}$
+ 采样当前 RVQ level $q \sim U\{1, Q\}$
+ 采样 mask $M \in \{0, 1\}^T$，根据 cosine schedule 采样 masking ratio $p = \cos(u)$，其中 $u \sim U[0, \pi/2]$，iid 采样 $M_i \sim \text{Bernoulli}(p)$
+ 在当前 RVQ level $q$（mask $Y_{t', q}$，如果 $M_{t'} = 1$ 且 $t' > t$）和所有 finer RVQ levels（$Y_{>t, >q}$）mask 选定的非 prompt tokens 

给定 masked token 序列，使用 ground-truth tokens 作为 target 训练模型，loss 仅在 q-th RVQ level 的 masked tokens 上计算。如下图所示，$T=4, Q=3, t=0, q=2$。

### 迭代并行解码

给定 conditioning signal，解码方案从所有 SoundStream tokens 开始，除了 prompt 的 token。然后，按 coarse-to-fine 顺序逐 RVQ level 采样 tokens，只有当 1, ..., q 级的 tokens 都采样完毕时才会继续到 q+1 级。在 RVQ level 内，使用 confidence-based sampling 方案。即，进行多次 forward pass，在每次迭代 $i$，根据 confidence score 保留 $p_i$ 个候选 masked 位置，其中 $p_i$ 遵循 cosine schedule。但是在每个 RVQ level 的最后一次迭代使用 greedy decoding，而不是 confidence-based sampling，这样可以提高感知质量。

> 逐 RVQ level 进行解码，可以利用 finer levels 的条件独立性假设，即可以并行采样多个 finer tokens，因为它们代表局部、细节的音频细节。随着解码到 finer RVQ levels，可以显著减少 forward pass 数量。

## 实验（略）
