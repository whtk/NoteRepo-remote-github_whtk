> 微软亚洲工程院、电子科大，AAAI，2019

1. 端到端的 TTS 如 Tacotron2 可以实现 SOTA 性能，但是有两个问题：
	1. 训练和推理时效率很低
	2. 基于 RNN 很难建模长时依赖
2. 本文提出用 MHA 替代 Tacotron2 中的 RNN 和 attention 部分，此时 encoder 和 decoder 的 hidden states 可以并行构造，从而提升训练效率；同时由于任意两个时间的输入是通过 self-attention 连接的，也可以解决长时依赖问题
3. Transformer TTS  输入为 phoneme 序列，输出为 mel 谱，然后通过 WaveNet vocoder 生成波形
4. 实验表明，相比于 Tacotron2 快 4.25 倍，且可以实现 SOTA 的性能

## Introduction

1. 传统 TTS 包含两部分：
	1. 前端用于文本分析和语言特征提取
	2. 后段基于语言特征进行声学参数建模、韵律建模和语音合成
2. 主流的方法是 concatenative 和 parametric 的语音合成，但是很复杂，且弄出一个好的语言特征很耗时耗力，而且还效果不好
3. Tacotron 和 Tacotron 2 的提出可以简化此过程，仅需一个神经网络就行。然后通过 vocoder 来生成音频。其采用的是端到端的架构，包含 encode 和 decoder：
	1. encoder 将 words or phonemes 映射到语义空间然后生成  encoder hidden states 序列
	2. decoder 将这些 hidden state 作为 context information，使用 attention 机制构造 decoder hidden state 然后生成 mel 谱
	3. encoder 和 decoder 都是 RNN 结构
4. 但是 RNN 是序列生成，导致无法并行。而 NMT 中提出了 Transformer 来替代 RNN
5. 本文将 Tacotron2 和 Transformer 组合起来得到新的 端到端 TTS 系统，使用 MHA 替代 encoder 和 decoder 中的 RNN 和 attention 模块；且 MHA 可以从不同的角度使用不同的 head 来构造 context vector
6. 实现 4.39 MOS（人类真实录音 MOS 为 4.44）；而且 Tacotron2 快 4.25 倍

## 背景

### seq2seq 模块

### Tacotron 2

结构如下：
![](image/Pasted%20image%2020230828214306.png)

### Transformer

结构如下：
![](image/Pasted%20image%2020230828214332.png)


## 基于 Transformer 的 TTS

在 TTS 中用 Transformer 有两个优点：
+ 没有循环结构 ，可以并行训练
+ self attention 可以实现类似于 global context 的效果，建立长时依赖关系
在韵律合成方面很有用，因为韵律不仅取决于邻近的几个单词，而是整个序列的语义。

结构如下：
![](image/Pasted%20image%2020230828214722.png)

### Text-to-Phoneme 转换

英语的发音有一些特定的规则，采用神经网络在训练时学习这种规则，但是有时候因为有些规则出现太少或数据太少又学不到这些规则，于是创造了一个 rule system，用它进行 text-to-phoneme 转化。

### Scaled Positional Encoding

Transformer 没有循环或卷积结构，所以如果打乱输入的顺序结果是不变的。因此需要通过  triangle positional embeddings 加入相对 位置信息：
$$\begin{aligned}
PE(pos,2i)& =\sin(\frac{pos}{10000^{\frac{2i}{d_{model}}}})  \\
PE(pos,2i+1)& =\cos(\frac{pos}{10000^{\frac{2i}{d_{model}}}}) 
\end{aligned}$$
其中，$pos$ 为 time step index，$2i,2i+1$ 为 channel index，由于在 NMT 中，source and target language 都来自语言空间，所以 scales  是相似的，但是在 TTS 中不一样，因为 source domain 是文本，target domain 是 mel 谱，采用固定的 positional embedding 不太好，于是设置 triangle positional embeddings 的权重是可以训练的，所以可以自适应 encoder 和 decoder  的 pre-net 输出，即：
$$x_i=prenet(phoneme_i)+\alpha PE(i)$$
其中，$\alpha$ 为可训练的参数。

### Encoder Pre-net

Tacotron2 中，采用三层的 CNN 来提取 text embedding， Transformer TTS 也有，称为 encoder pre-net，每个 phoneme 都对应一个可训练的 512 维的 embedding，每个卷积的输出层都是 512 个 channel，然后接 batch norm + relu。
> 由于 relu 的输出范围是 $[0,+\infty)$， triangle positional embeddings 的范围又是在 $[-1,1]$，在非负 embedding 中添加以 0 为中心的 positional information 会导致不以原点为中心的波动，从而损害模型性能。Hence we add a linear projection for center consistency。

### Decoder Pre-net

mel 谱 首先通过由两个全连接层+relu 的神经网络，称为 decoder pre-net，由于 phoneme 是可训练的，而 mel 谱 是固定的，因此这个 pre-net 可以把 mel 谱 投影到某种和 phoneme embedding 类似的子空间，从而可以计算 （phoneme, mel f rame）之间的相似度。

同时也添加了额外的 linear projection，一方面为了  center consistency，另一方面为了和 triangle positional embeddings 维度一致。

### Encoder

Tacotron2 中的 encoder 是双向 RNN，将其替换为 Transformer encoder，由于 MHA 可以将一个 attention 分到好几个子空间（head 数）所以可以建模不同方面的帧关系，且可以直接建模帧之间的长时依赖。同时还可以实现并行计算。

### Decoder

Tacotron2 中 decoder 是一个带有 location-sensitive attention 的两层的 RNN，将其替换为 Transformer decoder，从而带来两个不同：
+ 自注意力机制，其效果和 encoder 中的类似
+ 采用的是 MHA 而不再是 location-sensitive attention，MHA 可以得到更好的  context vectors

### Mel Linear, Stop Linear and Post-net

和 Tacotron2 一样，采用两个不同的  linear projections 来预测 mel 谱 和 stop token，采用了 五层的 CNN 来预测 residual
> 关于代码实现，transformer 模块 和 post-net 是分开训练的

## 实验

