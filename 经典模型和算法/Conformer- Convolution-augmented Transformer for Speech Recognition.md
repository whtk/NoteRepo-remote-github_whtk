> InterSpeech 2020，Google


1. 将CNN和Transformer结合起来，建模局部和全局依赖
2. 提出 Conformer，在ASR上的效果优于Transformer和CNN（独立模型），实现了SOTA
3. 在 LibriSpeech 中测试，不使用语言模型时，有 2.1%/4.3% 的WER，在test/test-other 上使用外部语言模型时实现了1.9%/3.9%的WER

>第一个 CNN + Transformer，大部分的创新都来自于重组（很多模块都是来自于已经提出的论文的，稍微有修改），说明了精心设计模型对最终的效果影响很大。

## Introduction

1. 背景：Transformer 被广泛应用，CNN 已经被用于 ASR
2. 但是局限在于，自注意提取局部特征能力不强，但是 CNN 很强；但是 CNN 又不太能提取动态的全局信息
3. 已经有研究表明，将卷积和自注意机制结合可以提高性能，可以同时利用位置相关的局部特征和内容相关的全局特征
4. 本文提出了一种新型的将自注意和 CNN 结合的方式：
	1. 自注意建模全局信息
	2. 卷积捕获局部信息
	3. 称为 Conformer，如图：![](./image/Pasted%20image%2020230308104555.png)
	4. ASR 的效果优于 Transformer Transducer

## Conformer Encoder

首先使用 convolution subsampling layer（下采样） 处理输入，然后接几层 conformer block，每个 conformer block 包括四个模块：
+ feed-forward
+ self-attention
+ convolution
+ second feed-forward

### Multi-Headed Self-Attention 模块
采用的是 Transformer 中的 Multi-Head Attention with Relative Positional Embedding 方法：![](./image/Pasted%20image%2020230308110813.png)
相对位置编码使得 MHA 对不同长度的输入泛化性更强。

采用 pre-norm 残差单元 + dropout 进行正则化，从而可以训练更深的网络。

### 卷积
（参考 [Lite Transformer]）卷积包括：a pointwise convolution and a gated linear unit，然后接一维深度卷积，最后接 Batch Norm。如图：![](./image/Pasted%20image%2020230308111214.png)

### Feed Forward 模块
如图：![](./image/Pasted%20image%2020230308111515.png)
原始的 Transformer 是两个 Linear 层+ 激活函数，然后进行 residual 和 LayerNorm，本文则继续使用 Pre-Norm 结构，同时在使用 Swish 激活和 Dropout 来进行正则化。

### Conformer Block

conformer 结构包括 FF+ MHA + CNN + FF，形成所谓的sandwich 结构（前后都是 FF 层），采用了 half-step feed-forward ，设 Conformer $i$ 的输入为 $x_i$，输出为 $y_i$ ，则计算过程为：$$\begin{aligned}
\tilde{x}_i & =x_i+\frac{1}{2} \operatorname{FFN}\left(x_i\right) \\
x_i^{\prime} & =\tilde{x}_i+\operatorname{MHSA}\left(\tilde{x}_i\right) \\
x_i^{\prime \prime} & =x_i^{\prime}+\operatorname{Conv}\left(x_i^{\prime}\right) \\
y_i & =\operatorname{Layernorm}\left(x_i^{\prime \prime}+\frac{1}{2} \operatorname{FFN}\left(x_i^{\prime \prime}\right)\right)
\end{aligned}$$
消融实验表明，使用 half-step residual 的连接相比于原始的 FF 能够显著提升效果。

然后实验试出来的——MHA 后面接 CNN 的效果是最适合 ASR 的。

## 实验

### 数据
LibriSpeech，970 小时 和 额外的 800M 个单词的文本（用于建立语言模型）。

特征：80 维 的 MelSpectrogram，25 ms win size，stride 10 ms，采用 SpecAugment 数据增强，F = 27。

### Conformer Transducer
一共实现了三个模型，分别具有10M、30M和118M参数：![](./image/Pasted%20image%2020230308115214.png)
同时 Decoder 层都是单层的 LSTM。

P(drop) = 0.1，采用 l2 regularization，Adam 优化器， β1 = 0.9, β2 = 0.98。
采用 transformer learning rate schedule，10k warm-up steps，peak learning rate $0.05 / \sqrt{\mathrm{d}}$，其中 $\mathrm{d}$ 是conformer encoder 的维度。

对于语言模型，采用三层的 LSTM，一共 1K 个 WPM tokens。
### 结果
![](./image/Pasted%20image%2020230308144757.png)
1. 没有 LM 时，M 模型已经和 SOTA 有竞争性
2. 加入 LM 后可以实现 SOTA

### 消融实验
![](./image/Pasted%20image%2020230308144944.png)
1. 卷积的影响最大
2. Macaron FFN 次之
3. swish 激活可以加速收敛

进一步比较，Macaron-net Feed Forward 的影响：![](./image/Pasted%20image%2020230308145304.png)

Attention Head 数量的影响：![](./image/Pasted%20image%2020230308145331.png)
16 是最好的。

卷积 kernel size 的影响：![](./image/Pasted%20image%2020230308145359.png)
趋势是kernel 越大效果越好，增加到 17 和 32 时，再往后性能变差。