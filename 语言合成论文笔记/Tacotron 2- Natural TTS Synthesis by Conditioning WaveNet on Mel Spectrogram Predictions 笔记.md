> Google，2018，ICASSP

1. 提出 Tacotron 2，直接从文本中合成语音的神经网络
2. 系统由以下部分组成：
	1. recurrent sequence-to-sequence feature prediction network 将 character embeddings 映射为 mel 谱
	2. modified WaveNet model 作为 vocoder 合成波形
3. 可以实现 4.53 的 MOS（专业录音系统 4.58）

## Introduction

1. 之前的系统产生的音频听起来都 muffled and unnatural
2. WaveNet 效果很好，但是输入需要很强的域知识来产生（包含lnguistic features, predicted log fundamental frequency (F0), and phoneme durations 等特征）
3. Tacotron 可以只用一个神经网络就产生这些特征（如 mel 谱），但 Tacotron 用的是  Griffin-Lim 合成波形
4. 本文提出了一个  统一的、全神经网络的 语音合成方案，将  Tacotron-style model 和 WaveNet vocoder 组合起来，可以合成人类无法区分的音频

## 模型架构

包含两个部分：
![](image/Pasted%20image%2020230827215840.png)
1. 一个 recurrent sequence-to-sequence feature prediction network with attention，用于给定输入字符预测 mel 谱
2. 修改的 WaveNet，基于 mel 谱 生成时域波形

### 中间特征表征

本文选择 low-level 的声学特征：即 mel 谱 来串联两个模块。

### Spectrogram 预测网络

网络由 encoder 和 带有 attention 的 decoder 组成，encoder 将字符序列转换为 hidden feature representation，decoder 基于此 representation 输出 mel 谱。

输入字符被表征为 512 维的 embedding，通过三层卷积，每层卷积都包含 512 个 $5\times 1$ 的 filter，即每一层看到 5 个字符，然后接 batch norm 和 relu 激活。最后一层卷积的输出通过一个 BLSTM 得到编码后的特征。

encoder 的输出送入到 attention network 中生成**一个**固定长度的 context vector。
> attention network 采用的是 location-sensitive attention，这可以缓和一些 failure mode

在将输入和 location feature 投影到 128 维的 hidden representation 后计算 Attention probabilities 。
>  Location features 是通过 32 个 1-D 卷积滤波器计算的。

decoder 是一个自回归的 RNN，从编码的输出中每次预测一帧。且前一个 time step 的预测输出会首先通过一个 pre-net。pre-net 作为 information bottleneck，对 attention 的学习很重要。
> pre-net 包含两个 256 维 的全连接层

pre-net 的输出和前面得到的 context vector 进行拼接，然后通过两层单向的 1024 unit 的 LSTM 层。

再把 LSTM 的输出和 context vector  再拼接通过一个线性投影层得到最终预测的 mel 谱 帧。

最后，预测的 mel 谱 帧通过一个 5 层的卷积 post-net 来预测 residual 加入到预测中以提高整体的重构性能，最后是把 post-net 的输入和输出相加来作为最终的 mel 谱。
> post-net 的每一层卷积都包含 512 个 $5\times 1$ 的滤波器+batch norm+tanh 激活。

在 预测 mel 谱 帧的同时，decoder LSTM的输出和 context vector 的拼接还会投影到一个标量中，然后通过 sigmoid 函数来预测输出序列结束的概率，即图中的 ‘stop token’（大于 0.5 认为结束）。此 token 在推理的时候可以动态地决定啥时候结束生成而不是总生成一个固定的长度。

decoder 中的 pre-net 中的卷积层都有 dropout 为 0.5，LSTM 层都有 zoneout 为 0.1。

相比于原始的 Tacotron，模型使用了更简单的模块：简单的 LSTM 和 卷积层而非 CBHG 和 GRU 模块。

相比于 Tacotron，每次只预测一个帧。
> Tacotron 每次预测三个帧

### WaveNet vocoder

采用修改的 WaveNet 将 mel 谱 转换为波形。原始的架构中有 30 个 dilated convolution layers 分为 3 组，每组的 dilation 为 $1,2,4, \ldots, 512$。为了处理 12.5 ms 的帧移，在条件模块中只用了两层的上采样层。

同时没有像原始的 WaveNet 那样预测离散的量化波形点，采用 PixelCNN++ 和  Parallel WaveNet 中的方法，使用 10-component mixture of logistic distributions (MoL) 来生成 24kHz 下的 16 bit 的样本。损失为 NLL。
> 为了计算 logistic mixture distribution, WaveNet stack output 通过 ReLU 激活 + 线性投影层 来预测每个 mixture component 的参数 (mean, log scale, mixture weight) 。

> 此外，根据 李宏毅 老师的课程，Tacotron 在 inference  的时候也需要加 dropout！！！加了效果才会好！！！

## 实验（略）