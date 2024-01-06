> Interspeech 2023，NTT Communication Science Laboratories

1. iSTFTNet 可以实现快速、轻量、高质量的语音合成，但是用的是 1D 的 CNN 通过时域的上采样来降低频域的维度，这样会导致一些东西被压缩
2. 提出 iSTFTNet2，采用 1D 和 2D CNN 来分别建模时域和 spectrogram 的结构

## Introduction

1. iSTFTNet 中的 temporal upsampling 可以缓解 1D CNN 建模高维表征的问题，但是会压缩一些 potential
2. 提出 iSTFTNet2，采用 1D 和 2D CNNs 来分别建模 global temporal 和 local spectrogram 结构
3. 设计了一个 2D CNN 在一定程度的时域转换之后，在频域进行上采样，从而可以利用高维 spectrogram 的建模，同时提出了一个高效的模块来进一步提升速度和模型大小

## iSTFTNet 原理（略）

见 [iSTFTNet- Fast and Lightweight Mel-Spectrogram Vocoder Incorporating Inverse Short-Time Fourier Transform 笔记](轻量化/iSTFTNet-%20Fast%20and%20Lightweight%20Mel-Spectrogram%20Vocoder%20Incorporating%20Inverse%20Short-Time%20Fourier%20Transform%20笔记.md) 。

## iSTFTNet2

![](image/Pasted%20image%2020231229161724.png)

前三个模块和 1 一样，除了在  1D ResBlock 中用的是 channel concatenation 而非 addition 来捕获更多的信息。

然后再用 1D-to-2D 转换，后面接 2D CNN 来捕获 spectrogram 中的 local structure。

为了避免由于 2D CNN 引入的参数的增加，在下采样后的频域维度空间做 2D 卷积，然后再用转置卷积做频域的上采样。

2D 模块如图：

