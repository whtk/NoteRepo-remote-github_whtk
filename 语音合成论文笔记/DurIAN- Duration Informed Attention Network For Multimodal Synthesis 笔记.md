> 腾讯 AI Lab，interspeech 2020

1. 提出多模态合成系统，同时产生高质量的自然语音和面部表情
2. 提出 Duration Informed Attention Network (DurIAN)，是一个自回归模型，输入文本和输出声学特征之前的 alignment 通过 duration model 学习
3. 提出了在 WaveRNN 上的 multi-band 的并行生成策略

## Introduction

1. DurIAN 将传统的参数系统和端到端系统组合起来，能够实现自然地鲁棒的语音生成
2. 现有端到端系统的生成不稳定的原因主要来自于 端到端的 attention，DurIAN 的核心就是将这个替换为参数合成系统中的 alignment model，而这个 model 包含一个 duration 预测 model，也可以被用于驱动面部表情生成，而无需语音和 face 的 parallel data
3. 主要贡献：
	1. 将 Tacotron 2 中的 attention 机制替换为 alignment model
	2. 采用 skip encoder 结构来 encode phoneme 序列 和 中文韵律下的  hierarchical prosodic structure
	3. 提出简单高效的 fine-grained style control 方法
	4. 提出了 multi-band synchronized parallel WaveRNN  算法

## DurIAN

输入来自文本的 符号序列，输出为 mel 谱 或者面部建模参数。
![](image/Pasted%20image%2020230915165640.png)

包含：
+ skip encoder ，用于 encode phoneme 序列和 prosodic structures
+ alignment model，用于对齐 phoneme 序列和 目标声学帧
+ 自回归的 decoder，用于一帧一帧地生成声学特征或者面部建模特征
+ post-net：预测没有被 decoder 捕获的 residual

skip encoder 输入为 $x_{1:N}$，输出 hidden state：
$$\mathrm{h_{1:N^{\prime}}=skip_{-}encoder(x_{1:N}),}$$
这里的 $N$ 是包含 phoneme 和 他们之间的 prosodic boundaries 的长度，$N^\prime$ 是不包含 prosodic boundaries 的输入 phoneme 序列的长度。得到的 hidden states 会根据每个 phoneme 的 duration 进行拓展，从而生成 frame aligned hidden state $e_{1:T}$：
$$\mathrm{e_{1:T}=state\_expand(h_{1:N^{\prime}},d_{1:N^{\prime}}),}$$
其中，$T$ 为声学帧的数量。然后用于训练自回归的生成模型：
$$\mathrm{y_{1:T}^{\prime}=decoder(e_{1:T}),}$$
最后通过 post-net 来预测 residual：
$$\mathrm{r_{1:T}=postnet(y_{1:T}^{\prime}).}$$
网络通过最小化 l1 loss 来训练：
$$\mathrm{L}=\sum_{n=1}^T|\mathrm{y}-\mathrm{y'}|+\sum_{n=1}^T|\mathrm{y}-(\mathrm{y'}+\mathrm{r})|$$
其中的 duration model 是独立训练的。

### Skip Encoder

prosodic structure 可以提高合成语音的泛化性，主要是通过插入特殊的 symbols 来代表不同 level 的 prosody boundaries：
![](image/Pasted%20image%2020230915212839.png)

网络的其他组成和 Tacotron 1 差不多。

### Alignment Model

端到端的系统基于 attention 来发掘 alignment，但是会产生一些无法预测的 artifacts（skipped or repeated words）。

这里就简单一点，phoneme 和 target 声学特征之间的 alignment 通过一个 duration model 来获得。每个 phoneme 的 duration 为和他对齐的声学帧的数量。

> 训练时，使用 forced alignment 来获得 alignment。

合成时，有一个额外的 duration model，且这个模型就通过最小化预测 的 duration 和 forced alignment 得到的 duration 的 MSE 来训练的。

### Decoder

decoder 也和 Tacotron 用的差不多，区别在于 attention context 替换为 alignment model 得到的 encoder state $e_{1:T}$。

### 多模态合成

语音和面部表情的同步可以通过两种方法得到：
+ 多任务学习，需要 parallel speech and face data
+ 通过 duration model，不需要 parallel speech and face data，可以独立地训练两个模型，且共享相同的 duration model
> 没看懂，面部表情的 label 怎么来的？？

## Fine-grained 风格控制

假设训练的时候有离散的 style labels。在 DurIAN 中，通过 style code 来实现风格控制：
![](image/Pasted%20image%2020230915214839.png)
而 style code 其实就是用于对 style embedding 进行 scaling。

而且由于 duration 和 style 是有关系的，会把  style code 引入到 duration model。另外就是和 encoder  的 hidden state 进行拼接。

图中，训练的时候 style code 都是非0即1，但是推理的时候可以是任意值。

## Multi-band WaveRNN

之前的 multi-band 策略是，对于每个 subband 训练一个独立的 vocoder。其实并没有减少总的计算量（只是可以并行化了）。

提出的 Multi-band WaveRNN 如下：
![](image/Pasted%20image%2020230915215705.png)
简单来说，就是不同的子带公用同一个 WaveRNN，将上一个 step 的 subband samples 作为输入来预测下一个 subband samples。

## 实验
https://tencent-ailab.github.io/durian/