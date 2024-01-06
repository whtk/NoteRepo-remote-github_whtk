> ICASSP 2022，NTT Communication Science Laboratories

1. mel 谱 vocoder 需要解决三个逆问题：
	1. 回复原始 spectrogram 的幅度
	2. 相位重构
	3. 频域到时域的转换
2. 通常方法都是直接在 black box 中解决问题，不能有效地利用 mel 谱中存在的时频结构
3. 提出 iSTFTNet，将一些输出替换为 iSTFT，从而减少了计算的参数
4. 在三个不同的 HiFi-GAN 变体中进行实验，实现了更快和更轻的模型

## Introduction

1. 通常的 vocoder 采用时域的上采样层直接从 mel 谱 中计算 raw waveform，但是是直接在 black box 中解决问题，不能有效地利用 mel 谱中存在的时频结构
2. 提出 iSTFTNet，用 iSTFT 替换模型中输出的一些卷积层，从而减少了计算的参数

## 相关工作

iSTFTNet 和其他卷积模型的比较：
![](image/Pasted%20image%2020231228163630.png)

## 方法

### 卷积 mel 谱 vocoder

mel 谱的提取如下：
![](image/Pasted%20image%2020231228163721.png)

1. 先提取幅度和相位
2. 丢掉相位
3. 转为 mel 尺度

典型的 vocoder 通过使用 CNN 隐式地逆转上述三个过程。

### iSTFTNet

在减少频域维度之后，显式地利用时频结构 + iSTFT ，如图：
![](image/Pasted%20image%2020231228164042.png)

iSTFT 有以下性质：
$$\text{iSTFT}(f_s,h_s,w_s)=\text{iSTFT}\left(\frac{f_1}s,\frac{h_1}s,\frac{w_1}s\right)$$
其中 $s$ 为缩放因子，$f_s,h_s,w_s$ 分别为 FFT size, hop length 和 window length。

这说明可以通过增加 $s$ 来减少频域的维度。

经验发现，通过两个上采样层即可实现快速且轻量化的效果。

### 实现

可以从任何卷积结构的 mel 谱 vocoder 中实现，但是三个修改点：
+ 输出通道为 2（包含幅度和相位）
+ 输出端应该分别用指数和 sine 激活函数
+ 原始波形通过 iSTFT 得到

## 实验（略）