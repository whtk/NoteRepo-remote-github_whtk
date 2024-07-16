> Interspeech 2024，NTT

1. 提出采用 mixture of adapters (MoA) 的 TTS：
    1. 将 MoA 引入到非自回归 TTS 模型的 decoder 和 variance adapter 中
    2. 通过选择与 speaker embeddings 相关的 adapters 来增强 zero-shot 能力
2. 可以在保持高质量语音合成的同时，减少参数量，提高推理速度

## Introduction

1. 用 lightweight TTS 在 zero-shot 情况下合成高质量语音仍然具有挑战性
2. 目前的方法都无法满足 lightweight zero-shot TTS 的需求
    1. PortaSpeech 和 LightGrad 只适用于单说话人 TTS
    2. LightTTS 只训练了几百个说话人，不足以实现 zero-shot TTS
3. MoE 和 MoA 可以提高模型能力，同时保持训练和推理效率
    1. MoA 主要用于 NLP
    2. ADAPTERMIX 将 MoA 用于 TTS，但是没有涉及 zero-shot TTS
4. 提出了一种基于 MoA 的 lightweight zero-shot TTS
    1. MoA 与 speaker embeddings 相关联
    2. 在推理时，可以根据说话人特征调整网络配置
    3. 通过在大型训练数据集上训练，可以在推理时覆盖各种说话人特征
    4. MoA 模块使用了不到 40% 的参数，推理速度提高了 1.9 倍

## 方法

Zero-shot TTS 模型包含三部分：
+ TTS 模型（encoder 和 decoder）
+ speaker-embedding extractor：基于 SSL speech model
+ vocoder

本文主要关注 TTS 模型和 MoA 模块。

### Backbone SSL-based TTS 模型

模型采用 SSL-based embedding extractor 处理输入语音序列，包括：
+ SSL model
+ embedding module：将 SSL 特征转为 speaker embedding，包含：
    + weighted-sum
    + bidirectional GRU
    + attention
    
在推理时，可以单独使用 embedding extractor 计算 speaker embedding。

### 基于 MoA 的 Speaker embedding

下图 b 为 decoder 的 FFT block，c 为 predictors（pitch, energy, duration predictors）和 MoA module。MoA module 包含 N 个 lightweight bottleneck adapters，每个包含两个 feed-forward layers 和一个 trainable gating network，用于根据 speaker embeddings 确定 adapters 的权重。
![](image/Pasted%20image%2020240715115701.png)


MoA module 计算如下：
$$\mathrm{MoA}(\mathbf{x},\mathbf{x}_\mathbf{e})=\mathbf{x}+\sum_{i=1}^Ng_i(\mathbf{x}_\mathbf{e})\cdot\text{Adapter}_i(\mathbf{x})$$

其中，$\mathbf{x} \in \mathbb{R}^D$ 为输入，$\mathbf{x}_\mathbf{e} \in \mathbb{R}^{D_{emb}}$ 为 speaker embedding，$\text{Adapter}_i : \mathbb{R}^D \rightarrow \mathbb{R}^D$ 表示 N 个 adapters，$g_i : \mathbb{R}^{D_{emb}} \rightarrow \mathbb{R}^N$ 是一个 trainable gating network。MoA 有两种实现方式：
    1. dense MoA：对所有 adapters 求和
    2. sparse MoA：只保留 top-k 的 $g_i$ 权重，其他设置为 0

用多任务目标函数来训练模型，其中 loss 包括 MSE losses 和 importance loss：
$$\begin{aligned}
L_{importance}(\mathbf{X})& =\left(\frac{\sigma(\text{Importance}(\mathbf{X}))}{\mu(\text{Importance}(\mathbf{X}))}\right)^2  \\
\mathrm{Importance}(\mathbf{X})& =\sum_{\mathbf{x_e}\in\mathbf{X}}g_i(\mathbf{x_e}) 
\end{aligned}$$
其中，$\mathbf{X} \in \mathbb{R}^{n \times D}$ 为 speaker embeddings 的 batch，$\mu$ 和 $\sigma$ 分别为序列的平均值和标准差。

## 实验

数据集：960 小时的日语

基于 FastSpeech2 模型，encoder 4 层，decoder 6 层，四种配置：
    + Small (S)：14M 参数
    + Medium Small (M/S)：19M 参数
    + Medium (M)：42M 参数
    + Large (L)：151M 参数


MoA 模块插入 Small 模型，两种 MoA 实现：
    + sparse MoA：8 个 adapters，top-k 为 3
    + dense MoA：3 个 adapters

参数量和速度比较如下：
![](image/Pasted%20image%2020240715144131.png)


输入和目标序列分别为 303 维的语言向量和 80 维的 mel-spectrograms，帧移为 10.0 ms。使用 HuBERT BASE 作为 SSL model，将 16 kHz 的原始音频输入转为 768 维序列，embedding modules 将其转为与 decoder 维度相同的固定长度向量。使用 HiFi-GAN 生成波形。

## 结果

客观评估：MCD，F0 RMSE 和 phoneme durations RMSE，结果如下：

![](image/Pasted%20image%2020240715144919.png)

上图显示了各模型的表现，随着参数量的减少，性能下降。

Proposed(d) 和 Proposed(s) 在所有指标上表现优于 S 和 M/S，表明插入 MoA 模块可以提高性能，且提升大于简单增加参数。与 M 相比，Proposed(d) 和 Proposed(s) 表现更好或相当，参数量不到 40%，推理速度提高了 1.9 倍。

主观评估：自然度和相似度，结果如下：
![](image/Pasted%20image%2020240716110224.png)


Proposed(s) 在自然度和相似度上均优于 S，M 和 Proposed(d)。这表明使用 MoA 可以提高性能。Proposed(d) 和 Proposed(s) 的差异表明，稀疏 gating 和更多的 adapters 可以提高性能。结果进一步表明，Proposed(s) 在专业说话人中的表现明显优于其他模型。

计算了测试数据中每个说话人的 adapters 权重的相关系数，如图：
![](image/Pasted%20image%2020240716112655.png)


Proposed(s) 在相似特征的说话人之间具有更高的相关系数，这表明专门的 adapters 被适当地选择。Proposed(d) 似乎无法区分专业和非专业说话人。总体来说，Proposed(s) 可以获得更具表现力的专家来处理更广泛的说话人。
