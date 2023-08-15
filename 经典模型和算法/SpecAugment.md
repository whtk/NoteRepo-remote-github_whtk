> 来自论文 - SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition，Google Brain 团队，InterSpeech 2019

1. 提出了SpecAugment，用于语音识别的数据增强方法
2. 直接应用于声学特征，包括三种增强方法：
	1. Time warping
	2. Frequency masking
	3. Time masking
3. 将此方法用于 LAS 模型，取得了 SOTA 结果

## Introduction
1. 提出 SpecAugment 来增强音频的 log mel spectrogram，简单又方便，不需要额外的数据

## 增强策略

### Time warping

给定 log mel spectrogram，time step 为 $\tau$，将其看成是一张图片，其中 时间为横轴，频率为纵轴。在 time step $(W,\tau-W)$ 内，沿着穿过图像中心的水平线的随机点将向左或向右弯曲距离 $w$，然后 $w$ 是从 $0,W$ 之间的均匀分布中随机选择的。在边界、四个角和垂直边的中点上固定了六个锚点。

### Frequency masking

在 mel 频率轴上，mask $f$ 个 连续的频率 $[f_0,f_0+f)$，其中 $f$ 是从 $0,F$ 中的均匀分布中随机选择的，$f_0$ 从 $[0,\nu-f)$ 中随机选的，$\nu$ 为 mel 频率的通道数。

### Time masking

同理，mask $t$ 个 连续的时间帧 $[t_0,t_0+t)$，其中 $t$ 是从 $0,T$ 中的均匀分布中随机选择的，$t_0$ 从 $[0,\tau-t)$ 中随机选的。

### 示例

下图给出了三种增强的示例：![](./image/Pasted%20image%2020230215194519.png)
由于 log mel spectrogram 被归一化为 0 均值，所以 mask 值设为 0 就相当于设为平均值。

同时还可以将上面的三种增强方法进行组合，如同时 mask 时间和频率。