> ICLR 2023， Seoul National University，NVIDIA

1. 提出 BigVGAN，一种可以在不需要 fine tune 的情况下泛化到多种 OOD 场景下的通用 vocoder
2. 训练的 vocoder 最大 112M 参数，在 LibriTTS 数据集上训练，可以在多个 zero-shot 的条件下实现 SOTA 的性能

## Introduction

1. 基于 GAN 的 vocoder 有以下优点：
	1. 并行，一次即可生成高维的波形
	2. 对模型架构不做限制（相比于 flow）
2. 提出 BigVGAN：
	1. 在 generator 中引入 periodic activations
	2. 提出 anti-aliased multi-periodicity composition 模块来建模复杂的波形
	3. 拓展到 112M 参数
	4. 14M 参数的模型可以超过相同大小的 SOTA 模块的性能

## 相关工作（略）

## 方法

架构如图：
![](image/Pasted%20image%2020231226112743.png)

采用 [HiFi-GAN- Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis 笔记](../语音合成论文笔记/HiFi-GAN-%20Generative%20Adversarial%20Networks%20for%20Efficient%20and%20High%20Fidelity%20Speech%20Synthesis%20笔记.md) generator 作为 baseline 架构，discriminator 通常多个 sub- discriminator 用于不同 resolution 的窗口，通常包含 MPD、马上到！、MRD。

训练的目标函数用的也是 HiFi-GAN 类似的，但是这里发现用 MRD 替换掉 MSD 效果更好，具体包含：
+ least- square 对抗损失
+ feature mapping loss
+ spectral L1 regression loss

### Periodic Inductive Bias

音频通常有很高的周期性，可以表示为一系列的主要周期成分的叠加（傅立叶级数），所以需要在 GAN generator 中引入这种 inductive bias。

采用别人提出来的一种新的 periodic activation，称为 Snake function 来作为 激活函数，定义为：$\begin{aligned}f_{\alpha}(x)=x+\frac{1}{\alpha}\sin^{2}(\alpha x)\end{aligned}$，其中 $\alpha$ 是一个用于控制频率周期的可训练的参数，越大频率越高。

### Anti- aliased（抗混叠） Representation

Snake activations 会引入一些混叠伪影，可以通过采用低通滤波器来抑制。通过将原始的信号进行 2 倍的上采样然后经过 Snake activations 之后再进行 2 倍的下采样，每次上采样和下采样都伴随采用基于 Kaiser 窗的 sinc 滤波器。

在每个 residual dilated convolution layers 中都采用上述操作，得到的模块称为 anti-aliased multi-periodicity composition (AMP).

### 大规模训练下的 BigVGAN

采用  HiFi-GAN V1 的配置（14M的参数），称为 BigVGAN-base，然后逐步增加上采样模块的数量和卷积通道，最后把通道增加到 1536，得到 112M 参数的 BigVGAN。

后面讲的是一些工程经验。

## 实验（略）