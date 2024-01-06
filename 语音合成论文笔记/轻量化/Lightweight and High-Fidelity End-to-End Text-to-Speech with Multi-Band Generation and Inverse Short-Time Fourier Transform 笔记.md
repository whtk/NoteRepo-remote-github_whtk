> ICASSP 2023，东京大学，LINE Corp

1. 基于 VITS，改了两个点来提高推理效率：
	1. 采用 iSTFT 变换来替换计算最耗时的模块
	2. 采用多频带生成
2. 没有用任何优化或者知识蒸馏的方法，可以实现端到端的优化

## Introduction

1. 发现 VITS 中 decoder 是最耗时的，于是将 decoder 替换为 iSTFT；为了进一步加速推理，将  iSTFT-based sample generation 和 multi-band 组合起来，每个 iSTFT 模块生成 sub-band 信号，然后依次相加来生成全频带的波形
2. 和 Nix-TTS 相比生成的质量更好

## VITS 分析

### VITS 简介

见 [VITS- Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech 笔记](../VITS-%20Conditional%20Variational%20Autoencoder%20with%20Adversarial%20Learning%20for%20End-to-End%20Text-to-Speech%20笔记.md) 。

### 每个模块的推理速度

计算每个模块的 RTF，定义为 合成所需的时间/合成语音的时长，如下表：
![](image/Pasted%20image%2020231227103306.png)

其中的 decoder 消耗了 96% 的推理时间。

## 方法

采用  iSTFTNet  中的思路，采用 iSTFT 来替换  HiFi-GAN  中的一些输出 layer。同时为了进一步提高生产速度，提出一种方法来将 iSTFT 和 多频带 组合起来。

### Multi-band iSTFT VITS

![](image/Pasted%20image%2020231227104126.png)

decoder 以顺序的方式执行以下过程：
+ VAE 的 latent $z$ 首先通过上采样因子 $s$ 通过 Res-Block 模块进行上采样，对于 $N$ 个子带信号，将其投影到幅度和相位变量
+ 采用 iSTFT 根据幅度和相位来生成子带新信号
+ 最后通过一个固定的合成滤波器组得到全频带信号（用的是 pseudo-QMF）

训练的时候，VITS 的重构损失修改为包含一个额外的 multi-resolution STFT loss。

最终得到的模型称为 multi-band iSTFT VITS (MB-iSTFT-VITS)，可以直接以端到端的方式优化。

### Multi-stream iSTFT VITS

前面的方式是一种固定的分解模式，可能存在一些 inflexible  constraint。于是受 multi-stream vocoder 启发，在 multi-band 结构中采用 trainable synthesis filter，从而可以以一种数据驱动的方式来分解波形，称为 multi-stream iSTFT-VITS (MS-iSTFT-VITS)。

## 实验