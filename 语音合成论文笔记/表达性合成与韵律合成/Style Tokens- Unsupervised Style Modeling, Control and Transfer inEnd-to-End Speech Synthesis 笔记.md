> Google，ICML，2018

1. 提出 global style tokens（GST），是一组和 Tacotron 联合训练的 embedding，embedding 训练时没有 explicit labels，因此可以学习很大范围的 acoustic expressiveness
2. 其产生的 soft interpretable “labels” 可以用来控制合成的速度、风格等多方面，也可以用于 风格迁移
3. 当在 noisy unlabeled 数据中训练时，GST 可以分解 噪声 和 speaker identity

## Introduction

1. 为了得到 human-like speech，TTS 系统必须建模韵律，包含 paralinguistic information（副语言信息）, intonation（语调）, stress（重音）, and style（风格）
2. 本文专注于风格建模，能够给定文本选择语音的风格（风格其实也很难定义，一般包括意图、情感、音调、流畅性等）
3. 风格建模的挑战有：
	1. 没有关于 prosodic style 的准确的度量
	2. 高动态范围导致的建模困难
4. 本文在 Tacotron 中引入 “global style tokens” (GSTs) ，不用任何 prosodic labels 即可训练，架构可以产生 soft interpretable “labels”，可用于多种风格控制和迁移任务

## 模型架构

基于 Tacotron，输入为 grapheme 或 phoneme，输出 mel 谱，然后通过 vocoder 合成音频。对于 Tacotron，vocoder 的选择并不影响韵律，韵律主要还是有 seq2seq 建模。

GST 模型如下：
![](image/Pasted%20image%2020230831210531.png)

包含 reference encode、style attention、style embedding 和 Tacotron。

### 训练

训练时的信号流如下：
+ reference encoder：来自 [Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron 笔记](Towards%20End-to-End%20Prosody%20Transfer%20for%20Expressive%20Speech%20Synthesis%20with%20Tacotron%20笔记.md) 从音频信号生成一个固定长度的 vector，称为 reference embedding，训练时的音频信号是 GT
+ reference embedding 通过 style token layer，把这个 embedding 作为内部的 attention 的 query，这里的 attention 不是用于学习 alignment，而是学习 reference embedding 和 一组 randomly initialized embeddings 中的每个 token 之间的相似性，而这些随机初始化的 embedding 即称为 GST，是在所有的 训练过程中共享的
+ attention 模块的输出为，当前 reference embedding 下对每个 GST 中的 tokens 的注意力权重，然后用这些权重对 GST 进行加权，得到一个 style embedding，把它作为  text encoder 每个 time step 的条件
+ style token layer 是和其他模块联合训练的，其损失为重构损失。从而得到的 GST 没有任何显式的 label 

### 推理

推理时有两种方式：
1. 直接基于 GST 中的某个特定的 token 作为 style embedding，从而不需要任何参考音频（图右）
2. 也可用一个额外的音频信号通过 Reference encoder 得到权重然后计算 style embedding，此时可以实现特定的风格迁移（图左）

## 模型细节

### Tacotron

具体细节见 [Tacotron- Towards End-to-End Speech Synthesis 笔记](../Tacotron-%20Towards%20End-to-End%20Speech%20Synthesis%20笔记.md) ，有一些细节不同：
+ phoneme 作为输入加速训练
+ decoder 中的 GRU 换成两层的 256-cell LSTMs，采用 zoneout 进行正则化
+ decoder 输出 80维的 log mel spectrogram ，每次输出两帧；然后通过  dilated convolution network 得到 linear spectrogram，通过 Griffin-Lim 算法合成波形

最终可以得到 4.0 的 MOS。

### Style Token 架构

#### REFERENCE ENCODER

reference encoder 由一堆卷积层和 RNN 组成：
+ 输入为 log-mel spectrogram
+ 通过 6个  2-D convolutional layer，每个卷积适使用 $3\times 3$ 的 filter 、 $2\times 2$ stride、 batch normalization 和 ReLU 激活，输出 channel 分别为 2, 32, 64, 64, 128 和 128
+ 然后通过一个 128-unit unidirectional GRU，最后一个 step 的输出状态作为  reference embedding

#### STYLE TOKEN LAYER

 style token layer 由一组 style token embeddings 和 一个 attention 模块组成，实验用的 token 数量为 10，每个 token embedding 都是 256 维的用于匹配 text encoder state。

在进行 attention 之间对每个 GST 通过 tanh 激活可以提高多样性。输出的时候采用 softmax 得到一些列权重。

实验了一些将 style embedding 作为条件的方法，发现直接将它加到 text encoder state 中的效果是最好的。

关于 attention 的选择，有 Dot-product attention, location-based attention 等，最后发现 Transformer 中的 multi-head attention 效果是最好的。

## Model Interpretation

### 端到端的聚类/量化

GST 模型其实也可以看成一个端到端的模型，用于将 reference embedding 解耦为一系列的基向量或者说 soft clusters（即 style token），每个 style token 的贡献通过  attention score 来体现。

GST-layer 有点类似于 VQ-VAE encoder，也是根据输入学习一个量化后的表征。

### Memory

GST embedding 可以看成是一种外部的 memory，用于存储从数据中训练的风格信息。训练时，reference signal 引导 memory 的写入，推理的时候则读取 memory。

## 相关工作（略）

## 实验：风格控制和迁移

