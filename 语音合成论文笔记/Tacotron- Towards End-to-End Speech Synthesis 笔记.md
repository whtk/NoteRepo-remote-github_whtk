> Google，Interspeech 2017

1. 提出 Tacotron，直接从文字开始的端到端的语音合成模型
2. 给定 (text, audio) 对，模型可以从零开始训练
3. 在 US English 合成上实现 3.82 的 MOS
4. 在 frame level 产生语音，比 sample-level 的自回归模型快

## Introduction

1. 当代的基于统计参数的 TTS 系统是复杂的，且都分开训练，存在误差的累积，通常包含：
	1. 文本前端：提出语言特征
	2. duration model
	3. acoustic feature prediction model
	4. complex signal-processing-based vocoder
2. 集成的端到端 TTS 的好处是：
	1. 减轻了 laborious feature engineering
	2. 允许丰富的条件和不同的属性加入
	3. 更容易适应到新数据
	4. 更鲁棒
3. TTS 的挑战在于，需要处理大量的 variations，且输出是连续的，输出长度远大于输入（可能会导致误差的快速累积）
4. 本文提出 Tacotron：基于 seq2seq 的端到端 TTS 模型
	1. 输入 characters，输出 spectrogram
	2. 可以从零开始训练
	3. 不需要 phoneme-level 的对齐

## 相关工作（略）

## 模型架构

![](image/Pasted%20image%2020230827095057.png)

Tacotron 的 backbone 是带有 attention 的 seq2seq 模型，包含：
+ encoder
+ attention-based decoder
+ post-processing net

### CBHG 模块

![](image/Pasted%20image%2020230827100346.png)

CBHG  包含：
+ 一组 1-D convolutional filters
+ highway networks
+ GRU

输入序列首先通过 $K$ 个 1-D convolutional filters 的集合，其中 $C_k$ 表示第 $k$ 个集合中包含的 filters 数量。这些 filters 显示地建模 local and contextual information，卷积的输出堆叠在一起，通过在时间轴上进行 max pooling 来增加 local invariances。

然后把处理后的序列通过一些固定宽度的 1-D convolutions，然后输出通过 residual connections 和原始的输入（最最最开始的输入）相加。

所有的卷积层都使用了 Batch normalization，卷积的输出再通过一个 multi-layer highway network 来提取 high-level 的特征。

最后，在顶端放一个双向的 GRU 来提取序列的 forward 和 backward context 特征。

### encoder

encoder 的目标是提取鲁棒的文本表征，输入为 字符序列，每个字符都表示为 one-hot，然后 embedding 到一个连续的向量中，然后对每个 embedding 采取一系列的非线性变换（称为 pre-net），在本文中，使用带有 dropout 的 bottleneck layer 作为 pre-net，CBHG 模块再把 pre-net 的输出转换为最终的表征。

### decoder

采用 content-based tanh attention decoder，在每个 time step，一个 stateful recurrent layer 产生 attention 的 query。

把 context vector 和 attention RNN cell 的输出 拼接起来作为 decoder RNN 的输入，使用带有 vertical residual connections 的堆叠的 GRU 作为 decoder，residual connections 可以加速收敛。

对于 decoder 的 target，虽然可以预测 raw spectrogram，但对于学习语音信号与文本之间的对齐而言，是一种冗余度很高的表征（这也是使用 seq2seq 的动机）。于是采用了不同的 target，即 80-band mel-scale spectrogram，bands 数更少。

采用的是简单的全连接层来预测 target，且每次预测 $r$ 帧而非 1 帧来加速训练和推理。
> 因为邻近的帧相关性很大，而且每个字符可能对应多个帧

第一个 decoder step 的输入为 all-zero frame（图中的 GO），推理时，在 step $t$，前一个 step  $r$ 帧的最后一帧作为输入。训练时，每隔 $r$ 帧将 GT 作为 decoder 的输入。输入帧在输入到 decoder 之前还会先通过 pre-net，且 pre-net 中的 dropout 对于模型的泛化非常重要。
> 其实 dropout 的作用有点类似于 scheduled sampling

### 后处理网络和波形合成

后处理网络是将 seq2seq target 转换为 可以用于波形合成的 target。由于采用 Griffin-Lim 作为合成器，post-processing net 预测从线性频率下的 spectral magnitude。

且 post-processing net 可以看到整个 decoder 的输出序列，本文采用 CBHG 作为 post-processing net。
> 其实 post-processing net 也可以用于预测其他的，如 vocoder parameters 或者当作 WaveNet-like neural vocoder 来直接合成波形。

采用 Griffin-Lim 来合成波形，发现将 predicted magnitude 乘以 1.2 倍可以减少伪影，可能是因为 harmonic enhancement effect。50 次迭代之后可以收敛。

## 模型细节和实验（略）
