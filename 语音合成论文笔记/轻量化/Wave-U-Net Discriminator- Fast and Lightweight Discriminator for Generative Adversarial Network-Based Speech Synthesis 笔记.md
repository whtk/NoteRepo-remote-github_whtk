> ICASSP 2023，NTT 实验室，日本

1. TTS 中，GAN 可以用于提高语音质量，通常会采用一系列的 discriminator，但是会带来参数和计算量的增加
2. 提出 Wave-U-Net discriminator，采用 Wave-U-Net 结构的 single but expressive discriminator，这种结构可以为 generator 提供足够丰富的信息使得合成的语音接近真实语音
3. 在 HiFi-GAN 和 VITS 中实验，对于 HiFi-GAN，速度可以快 2.31 倍，轻了 14.5 倍；对于 VITS，速度可以快 1.90 倍，轻了 9.62 倍

## Introduction

1. 作者考虑以下问题：能否将集成的 discriminator 替换为一个 expressive discriminator，于是提出 Wave-U-Net discriminator，训练架构如下：

![](image/Pasted%20image%2020240120103827.png)

以 sample-wise 的方式处理波形，同时还采用 encoder 和 decoder 架构提取 multi-level 的特征

2. 实验在 HiF-GAN vocoder 的不同的数据集下、在 VITS TTS 模型中进行验证，结果表明可以实现 comparable 的合成质量但是速度更快，参数更少

## GAN-based 语音合成

语音合成中，GAN 的 generator 通常用于从输入 $s$ 中合成波形 $x$。训练的时候， generator 和 discriminator 用两个 loss 进行训练：adversarial 和 feature-matching losses。

### 损失

对抗损失定义为：
$$\begin{aligned}\mathcal{L}_{Adv}(D)&=\mathbb{E}_{(x,s)}[(D(x)-1)^2+(D(G(s)))^2],\\\mathcal{L}_{Adv}(G)&=\mathbb{E}_s[(D(G(s))-1)^2],\end{aligned}$$
其中 $D$ 用于区分真实和合成的语音。

feature mapping 损失为：
$$\mathcal{L}_{FM}(G)=\mathbb{E}_{(x,s)}\left[\sum_{i=1}^T\frac1{N_i}\|D^i(x)-D^i(G(s))\|_1\right],$$
其中 $T$ 表示 $D$ 中的 layers 数量。generator 通过最小化这个 loss 来合成接近于 GT 的语音。

## Wave-U-Net Discriminator

### 概览

discriminator 必须捕获足够丰富的信息 来传递到 generator 中，于是提出 Wave-U-Net Discriminator，如图：
![](image/Pasted%20image%2020240120110353.png)

通常的 discriminator 只包含 encoder 部分，然后通过下采用提取特征来判断真实性，而 Wave-U-Net discriminator  则有 encoder-decoder 架构，通过上采样和下采样，输出的维度和输入相同，以 sample-wise 的方式判断真假。然后还通过  skip connection 提取 multi-level 的特征，类似于之前的集成 discriminator 来提供足够的信息进行判断真假。

### 稳定 GAN 训练的技巧

GAN 对于架构的设计和敏感，实验表明， 采用传统的 Wave-U-Net 结构会使对抗损失饱和。
> 原因可能在于，discriminator 只关注于某些特定的特征从而很容易判别真假，导致传到 generator 的信息不够。

normalization 和 residual 连接可以缓解这个问题。

已有工作表明，语音合成中的 discriminator 对 normalization 非常敏感，但之前的工作表明，weight normalization 不足以稳定 Wave-U-Net discriminator 的训练，可能是因为 Wave-U-Net discriminator 包含 encoder 和 decoder 结构，比通常的 discriminator 更深，于是提出采用 global normalization，定义为：
$$b=a\left/\sqrt{\frac1N\sum_{i=1}^N(a^i)^2+\epsilon}\right.$$
其中 $\epsilon=10^{-8}$， $N$ 为总的特征数， $a,b$ 分别代表原始的和归一化的特征向量。 这种规划可以防止 discriminator 学习某一特定的特征。

也可以使用 layer normalization， 但是同用的是 Globol normalization。

在训练完成后，adversarial losses 在训练完成之后还是会饱和，可能是因为网络过深的原因。因此引入残差连接来避免梯度消失。残差连接的输出乘以了缩放因子 $0.4$。

整个结构如下：
![](image/Pasted%20image%2020240120155549.png)

## 实验（略）
