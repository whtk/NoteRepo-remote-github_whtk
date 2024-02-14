> ICLR2021，UCSD、NVIDIA、BaiDu

1. 提出 DiffWave，一种用于有条件和无条件的普适的波形生成 DPM
2. 模型是非自回归模型，通过马尔可夫链将白噪声转换为语音波形
3. DiffWave 可以实现基于 mel 谱 的条件声码器, 以类别为条件的生成 和 无条件生成
4. 效果和 WaveNet 相匹配，但是速度更快；在无条件生成中超过了基于 GAN 的模型

> 本质为 vocoder，从噪声波形开始进行的迭代去噪，思路就是标准的 DDPM 的思路，但是模型不用 U-Net 而是用的 WaveNet 中的 卷积+gate 的网络，然后生成的时候不是一个点一个点的生成，而是一次生成所有的样本点（因为没有使用因果卷积）。

## Introduction

1. 自回归模型在无条件下会生成虚构的单词的声音或者效果不好的样本，因为没有任何条件去生成长序列是非常困难的
2. 提出 DiffWave，相比于之前的工作优点在于：
	1. 非自回归，从而可以并行合成高质量音频
	2. 灵活，相比于 flow-based 模型没有任何结构的约束
	3. 损失只有 ELBO，没有额外的损失函数（如 spectrogram-based losses）
	4. 普适：可以进行有条件或无条件的生成
3. 本文贡献：
	1. DiffWave 采用类似于 WaveNet 的 feed-forward and bidirectional dilated convolution，在合成质量上和 WaveNet 相匹配，但速度更快
	2. 只有 2.64M 的参数，在 V100 GPU上合成 22.05 kHz 的音频，且比 real time 快5倍，但是比 SOTA 的 flow-based 模型慢
	3. 在无条件和以类别为条件的情况下，可以极大的超过 WaveGAN 和 WaveNet


## DPM（略）



## DiffWave 结构
![](image/Pasted%20image%2020230824110213.png)
基于 bidirectional dilated convolution 结构建立模型 $\epsilon_\theta: \mathbb{R}^L \times \mathbb{N} \rightarrow \mathbb{R}^L$  ，模型是非自回归的，从噪声生成音频 $x_0$ 只需要 $T$ 次迭代，而迭代数显然是少于波形的长度 $L$ 的。
> 为什么结构相似但是却是非自回归呢？
> 因为这里没有使用因果卷积，而是双向的空洞卷积，从而可以看到所有的 time step 的样本，也就是说，模型每次都输出全部的 音频，此时网络层需要同时进行降噪和样本关系建模（自回归则不需要学习降噪），这里的网络结构和 UNet 完全不一样了。

网络由 $N$ 个 residual layer 组成，每个 residual  layer 都有 $C$ 个通道。这些 layer 分成 $m$ 个 block，每个 block 则有 $n=\frac{N}{m}$ 个 layer。每个layer 都采用 kernel size 为 3 的 bidirectional dilated convolution，且 dilation 逐层加倍。然后把所有的 residual layers 的输出加起来。

> 每一个 residual layer 的输出都是下一层的输入，且输入输出的维度相同，都是长为 $L$ 的样本点。说明，这个没有用 UNet 架构，

### DIFFUSION-STEP EMBEDDING 

对于每个 time step，采用 128 维的向量：
$$t_{\text {embedding }}=\left[\sin \left(10^{\frac{0 \times 4}{63}} t\right), \cdots, \sin \left(10^{\frac{63 \times 4}{63}} t\right), \cos \left(10^{\frac{0 \times 4}{63}} t\right), \cdots, \cos \left(10^{\frac{63 \times 4}{63}} t\right)\right]$$
然后采用三个 FC 层，前两个在所有的 layer 中共享，最后一个将向量映射为 $C$ 维的 embedding。

### CONDITIONAL GENERATION

Local conditioner：语音合成中，vocoder 可以将对齐的语言特征、mel 谱 或者 hidden state 作为条件合成波形。本文以 mel 谱（通过 transposed 2-D convolutions 转换为和波形长度相同） 作为条件，通过 Conv1×1 映射到 $2C$ 通道，将此条件作为 dilated convolution 的 bias 项。

Global conditioner：很多情况下，condition 是全局的离散标签（如 speaker ID 或 word ID），假设每个 label 的 embedding 维度都是 $d_\text{label}=128$ ，首先采用 Conv1×1 将 $d_\text{label}$ 映射到通道数为 $2C$ ，然后同样作为 bias 项引入卷积。

### UNCONDITIONAL GENERATION

无条件生成的重点在于，网络输出的感受野的大小 $r$ 要比语音的长度 $L$ 长，即需要 $r \ge 2L$，从而最左和最右的输出可以 cover 整个序列。

对于堆叠的 dilated convolution 层，输出的感受野为 $r=(k-1) \sum_i d_i+1$，其中 $k$ 为 kernel size，$d_i$ 为第 $i$ 层的 didilation。例如，对于 30 层的 dilated convolution，当 $k=3$，$d_i=[1,2,\dots,512]$ 时，$r=6139$，此时只能覆盖16KHz 下的 0.38s 音频。 

虽然可以增加层数来提高感受野的大小，但是网络越深效果越差。

但是 DiffWave 有优势来增加感受野：通过从 $x_T$ 到 $x_0$ 的迭代，感受野增加到 $T\times r$，从而使得 DiffWave 适合无条件的生成。
> 妙啊！

## 相关工作（略）

## 实验