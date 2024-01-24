> InterSpeech 2020，三星电子

1. 提出端到端 TTS，可以在 CPU 上实现实时 TTS
2. 包含 attention-based 自回归 seq2seq 声学模型 +  LPCNet vocoder 来生成波形
3. 声学模型采用 Tacotron 1 和 2，用了 purely location-based attention 来确保稳定性
4. 推理时，decoder 展开，然后以流式的方式进行声学特征的生成

## Introduction

1. 大部分的语音合成都包含  acoustic model，这类模型通常参数很多，计算耗时，从而难以实现实时的语音合成

## 相关工作（略）

### 提出的方法

本文重点在于优化声学模型以实现在 CPU 上实时应用。基于已有的工作，贡献如下：
+ 用于纯自回归模型的流式推理方法
+ 轻量化的端到端 TTS 架构
+ 采用鲁棒的对齐模型，减轻 attention 的问题

许多模型的推理时间取决于序列的长度，对于长序列生成很慢。提出的声学模型可以在 50ms 的延迟下，在单线层 CPU 上 31 times faster than real-time。

## 方法

### 声学模型架构

声学模型将输入序列映射到声学特征序列， 然后采用 vocoder 合成波形，本质是一个直接修改 Tacotron 1 和 2 的 seq2seq 模型。

encoder 将输入序列转化为可学习的 embedding 向量，做两层的做两层的 pre-net 和 CBHG 模块 来产生最终的 encoder 表征，decoder 端，输入的声学帧也通过  pre-net，在每个 decoding step 通过 RNN 产生 hidden state $h_i$。

输出的 acoustic frames 通过一层卷积层进行预测，decoding 结束后，通过 post-net 然后进行 residual 连接。和 Tacotron 2 一样，也用了 stop token 来判断结束。

### 对齐模型

已有论文发现，纯 location-based GMM attention 可以泛化到任意长度的序列，且不会违反对齐的单调性。

本文采用GMM的一个变体，将混合高斯分布替换为 将混合高斯分布替换为MoL，称之为MoL attention。为计算对齐，直接采用 logistic distribution 的 CDF，其计算简单且等效于 sigmoid 函数：
 $$F(x;\mu,s)=\frac1{1+e^{-\frac{(x-\mu)}s}}=\sigma\left(\frac{x-\mu}s\right)$$ 对于每个 decoder step $i$，encoder 中对第 $j$ 个 time step 的对齐概率计算如下：
 $$\begin{aligned}
a_{ij}& =\sum_{k=1}^Kw_{ik}\left(F(j+0.5;\mu_{ik},s_{ik})-F(j-0.5;\mu_{ik},s_{ik})\right)  \\
 c_i&=\sum_{j=1}^Na_{ij}e_j 
\end{aligned}$$
其中 context vector 为 encoder representations 的加权和。

mixture 在每个 time step 的参数计算如下：
$$\begin{gathered}
\mu_{ik}=\mu_{i-1k}+\exp(\hat{\mu}_{ik}) \\
s_{ik}=\exp(\hat{s}_{ik}) \\
w_{ik}=softmax(\hat{w}_{ik}) 
\end{gathered}$$
然后通过两成全卷积来预测：
$$(\hat{\mu}_{ik},\hat{s}_{ik},\hat{w}_{ik})=W_2\tanh(W_1(h_i))$$

### vocoder 

用的是别人改进的 [LPCNet- Improving Neural Speech Synthesis Through Linear Prediction 笔记](../LPCNet-%20Improving%20Neural%20Speech%20Synthesis%20Through%20Linear%20Prediction%20笔记.md)。

### 流式推理

语音合成时，必须先生成 acoustic frame 然后输入到 vocoder 中，类似于 FastSpeech 可以生成整个序列然后一起通过 vocoder，但是对于类似 CPU 这种设备，合成很耗时（不能并行）。

提出的流式推理如下图：
![](image/Pasted%20image%2020240123214629.png)

LPCNet 是自回归的，所以可以在收到第一帧的 acoustic frame 就开始生成。

为了减少延迟，以分组的方式获得连续的 acoustic frame，然后并行通过 post-net。每个 decoder step 的输出 frame 放在 buffer 中，然后以 larger chunks 形式输入到 post-net 中，每个 chunk 中 frame 的数量可以实现 latency 和 RTF 的trade-off。数量小，则 latency 小，但是总的 window frames 的数量多，从而增加了 RTF。这里选择 frame 数为 100，对应 1s 的音频。

上述方法的唯一要求是，CPU 处理速度 faster than real time。

## 实验和结果（略）
