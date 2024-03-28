> computer science 2020，UC Berkeley

1. 大多现有的 vocoder 很难并行化，因为每个生成的样本都是基于之前的样本
2. 提出 SqueezeWave，基于 WaveGlow 的一系列轻量级 vocoder，可以生成质量相似的音频，但是 MACs 减少了 61-214 倍

> 两个点：分析 WaveGlow 的结构，reshape 了一下输入维度（减少时间维度增大通道维度）；将 1D 卷积替换为 深度可分离卷积。

## Introduction

1. 现有 TTS 包含两个部分：
    1. synthesizer 从文本输入生成声学特征（如 mel-spectrogram）
    2. vocoder 从声学特征生成波形
2. 本文重点在于提高 vocoder 的效率
3. WaveGlow 是并行的，但是还是需要很多的计算，并不适合在边缘设备上实时部署
4. 提出 SqueezeWave，一系列轻量级 vocoder，可以在设备上实时合成语音：
    1. 重新设计 [WaveGlow- a Flow-based Generative Network for Speech Synthesis 笔记](../WaveGlow-%20a%20Flow-based%20Generative%20Network%20for%20Speech%20Synthesis%20笔记.md) 的网络结构，通过重新排列音频张量、采用深度可分离卷积等优化，使得 SqueezeWave 的 MACs 减少了 61-214 倍
    2. 在 Intel i7 CPU 的 Macbook Pro 上，SqueezeWave 的速度为 123K - 303K samples/s，比实时快 5.6-13.8 倍

## WaveGlow 的计算复杂度

WaveGlow 是一个基于 flow 的模型，可以把 mel-spectrogram 作为条件来生成音频波形。WaveGlow 由一系列的双射组成，双射以文本为条件，训练时将数据分布转换为 latent space 中的高斯分布。推理时，从高斯分布中采样，然后将其转换回数据分布。

具体来说，先将相邻的样本分组，形成一个多通道输入 $x \in \mathbb{R}^{L,C_g}$，其中 $L$ 是时间维度的长度，$C_g$ 是每个时间步的音频样本数（波形中的总样本数就是 $L \times C_g$）。然后，这个分组的波形 $x$ 通过一系列的双射进行转换，每个双射以 $x^{(i)}$ 为输入，产生 $x^{(i+1)}$ 作为输出。

在每个双射中，输入信号 $x^{(i)}$ 先通过可逆卷积，然后沿着通道维度分为 $x_a^{(i)}, x_b^{(i)} \in \mathbb{R}^{L,C_g/2}$。$x_a^{(i)}$ 用于计算仿射耦合系数 $(\log s^{(i)}, t^{(i)}) = WN(x_a^{(i)}, m)$，然后把 $s^{(i)}, t^{(i)} \in \mathbb{R}^{L,C_g/2}$ 用于 $x^{(i)}$ 仿射耦合系数。$WN(·,·)$ 是类似 WaveNet 的结构，$m \in \mathbb{R}^{L_m,C_m}$ 是 mel-spectrogram，$L_m$ 是 mel-spectrogram 的时间长度，$C_m$ 是频率数。ACL 计算为：$x_b^{(i+1)} = x_b^{(i)} \otimes s^{(i)} + t^{(i)}, x_a^{(i+1)}=x_a^{(i)}$，$x_a^{(i)}$ 和 $x_b^{(i+1)}$ 沿着通道维度连接。

WaveGlow 的大部分计算在 WN 函数 $WN(·,·)$ 中，如图：
![](image/Pasted%20image%2020240328160956.png)

函数的第一个输入通过 point-wise convolution 处理，标记为 start。这个卷积将 $x_a^{(i)}$ 的通道数从 $C_g/2$ 增加到一个更大的数。在 WaveGlow 中，$C_g=8$，start 的输出通道数是 256。然后输出通过一个 kernel size 为 3 的 dilated 1D convolution 处理，记为 in layer。同时，mel 谱 $m$ 也输入到函数中，其时间长度 $L_m$ 音频波形长度 $L$ 小很多。在 WaveGlow 中，$L_m=63,C_m=80,L=2000,C_g=8$。为了匹配时间维度，WaveGlow 对 $m$ 进行上采样，然后通过一个卷积层 cond layer。in layer 和 cond layer 的输出通过gate 函数进行组合，最后通过 res skip layer。这层的输出在 WaveGlow 中的长度是 $L=2000$，通道数是 512。然后沿着通道维度分为两个分支。

上面的结构重复 8 次，最后一个输出通过一个点卷积 end 处理。此卷积计算 $s^{(i)}, t^{(i)}$，并将通道数从 512 降到 $C_g=8$。

WaveGlow 的计算复杂度为：生成 1 秒 22kHZ 音频需要 229G MACs。其中：
+ in layer 占 47%
+ cond layer 占 39%
+ res skip layer 占 14%

## SqueezeWave

### Reshaping 音频波形

可以发现，冗余主要来自输入音频波形的 shape。在原始 WaveGlow 中，输入波形 reshape 为时间维度大、通道数小（$L=2000,C_g=8$）。这导致计算复杂度高：
1. WaveGlow 是 1D 卷积，计算复杂度与 $L$ 成正比
2. mel-spectrogram 的时间维度比分组后的音频少：$L=2000$ 而 $L_m=63$。为了匹配这两个维度，需要对 mel-spectrogram 上采样。上采样会导致冗余。从而 WaveGlow 中的 cond layers 中的大部分计算是不必要的
3. 在 WN 函数中，8 通道被投影为大的中间通道数，然后在输出的时候又被压缩到 $C_g=8$。这种 drast reduction 会导致网络中的信息瓶颈，中间表征的信息可能丢失

于是 reshape 输入音频 $x$，减少时间长度，增大通道数，同时保持 WN 中的内部通道数不变。给出两种设置：$L=64,C_g=256$ 或 $L=128,C_g=128$。当 $L=64$ 时，时间长度和 mel-spectrogram 一样，不需要上采样。当 $L=128$ 时，我们改变操作的顺序，先对 mel-spectrogram 应用 cond layer，然后应用最近邻上采样，从而可以进一步减少 cond layers 的计算。

### Depthwise 卷积

用 depthwise separable convolutions 替换 in layer 中的 1D convolutions：计算如下图：
![](image/Pasted%20image%2020240328163217.png)

> 其实就是直接替换 1D 卷积。

### 其他改进

其他改进包括：
+ 由于时间长度变小了，WN 不需要用 dilated convolutions 来增加感受野，所以可以用普通卷积替换所有的 dilated convolutions
+ res skip layers 的输出分为两个分支，作者认为这个分支是不必要的，因为两个分支的拓扑结构几乎相同，于是将其合并为一个，并将 res skip layers 的输出通道数减半

最终得到的结构见第一张图的右边。

## 实验（略）

