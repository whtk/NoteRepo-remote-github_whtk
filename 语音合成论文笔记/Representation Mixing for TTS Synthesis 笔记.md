> Bengio，2019

1. 基于字符和音素的 TTS 很强，但是两者只能选一个
2. 提出 representation mixing 混合表征法，可以在一个 encoder 中组合多种语言信息，也就是可以选其中一个也可以混合使用

## Introduction

1. TTS 是给定语言序列 $l$，来生成音频特征序列 $a$，且两者的长度和维度一般不同，因此需要对齐，本文通过联合学习对齐这两种类型的信息来解决对齐问题

### 数据表征

音频特征采用对数mel谱，语言特征可以是粗粒度如 grapheme（字素，或者说字符），细粒度的包含发音信息的如 phoneme（音素）

### 表征混合的 motivation

某些情况下是需要发音知识的，如文本中存在多音字。

如果没有外部的发音信息，TTS 通常会产生很模糊的输出。所以就需要组合 grapheme 和 phoneme 到一个单一的 encoder 中。

## 表征混合

系统的输入包含数据序列 $l_j$ 和掩码序列 $m$。
其中 $l_j$ 包含 字符序列 $l_c$ 和音素序列 $l_p$ 的混合；mask （0/1）用于给出来自哪个序列。训练的时候，在 word level 随机混合（将所有的空格和标点看成字符）。

例如，字符序列为 "the cat"，音素序列为 "@d@ah @k@ah@t"，这里的@ 用于分隔音素，那么训练的时候可能是：
+ "the @k@ah@t", 对应的 mask 为 $[0,0,0,0,1,1,1]$
+ "@d@ah cat"，对应的 mask 为 $[1,1,0,0,0,0]$
这其实可以看成是一种数据增强，同时可以平滑化字符和音素信息，使其不过度依赖任何一种表征。

### 将 embedding 进行组合

混合序列 $l_j$ 分别通过两个 embedding 矩阵来得到 $e_c,e_p$，然后通过 mask 序列生成混合表征 $e_j$，然后再通过 mask 自己的 embedding $e_m$ 进一步得到 $e_f$，最后作为后面的模型的输入：$$\begin{aligned}
& e_j=(1-m) * e_c+m * e_p \\
& e_f=e_m+e_j
\end{aligned}$$
如图：
![](../../Pasted%20image%2020230609225334.png)

### Stacked Multi-scale Residual Convolution（多尺度残差卷积网络）

![](../../Pasted%20image%2020230609225551.png)

前面得到的 embedding $e_f$ 输入到 Stacked Multi-scale Residual Convolution（SMRC）子网络中，其包含多个多尺度的卷积层，每个多尺度卷积层的多尺度依次在 channel 维度上concatenate 1×1、3×3和5×5 的kernel。然后再通过残差连接，最后接 BN。

然后将得到的结果通过 BLSTM 层，得到模型的 encoder 部分。

### Noisy Teacher Forcing 的重要性

音频信息通过 带有 dropout 的 pre-net （训练和测试的时候都要）。可以提高模型在生成时候的稳健性。

### 基于 Attention 的 RNN decoder

通过 LSTM network 网络驱动 ，采用 Gaussian mixture 注意力方法，基于上下文和 pre-net 来实现注意力激活。

后面的 LSTM decoder 以prenet激活、注意力激活和之前层的隐藏状态为条件。

最终的 hidden state 通过投影层来匹配音频帧的维度。

### 截断的 BPTT

decoder 采用截断的 BPTT 进行更新，仅处理相关音频序列的一个子序列，同时复用语言序列，直到相关音频序列结束：
![](../../Pasted%20image%2020230609230910.png)
其实方法很简单，就是重复的部分可以 batch 计算，其他的部分就单独计算。

最后采用 L-BFGS 从mel谱生成语音。

## 实验

基于 LJSpeech 进行训练，每个 word 选择字符还是音素的概率是 0.5。

研究三个问题：
1. 以 仅基于字符的模型作为 baseline，表征混合（RM）是否有改进
2. RM是否通过在固定PWCB上训练的 baseline 上对未知单词（PWCB）推理进行 character backoff 来改善音素
3. 在用表征混合训练的模型中，PWCB 是否优于基于字符的模型

结果：
![](../../Pasted%20image%2020230609231902.png)

+ 混合表征模型的质量更好