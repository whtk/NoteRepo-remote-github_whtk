> ICLR，2019，

1. 提出 WaveGAN，第一个把 GAN 用于无监督的波形合成，可以合成 1s 的音频，且有 global coherence，很适合 sound effect generation
2. 实现表明，没有 label 时，WaveGAN 可以产生 intelligible words，而且可以从鼓声、鸟声和钢琴声等合成音频

> 只看模型不看实验的话，WaveGAN 其实就是把 DCGAN 的二维图像输入改成了一维音频，然后做一些改进来提高感受野。同时提出了一个所谓的 phase shuffle 来减轻棋盘效应的影响。

## Introduction

1.  本文研究了两个 GAN：
	1. SpecGAN：首先设计了 spectrogram representation，然后采用 DCGAN 来建模
	2. WaveGAN：将 DCGAN 的结构 flatten 以在一维上进行
2. 论文的关注点其实并不是语音生成任务，而是：
	1. 无监督方法是否可以隐式学习高维音频信号的 global structure（无条件，也就是没有文本）

## GAN（略）

## WaveGAN

### 音频和图像之间的本质差异

可以用 PCA 分析这两种数据类型的差异，如图：
![](image/Pasted%20image%2020230928220204.png)

图像的主成分主要捕获强度、梯度和边缘特征，而音频则具有周期性，可以分解成频率带。

从而较大窗口的一致性在音频中很普遍，表明处理音频需要更大的感受野。

### WaveGAN 结构

基于 DCGAN，generator 采用 transposed convolution 来对 feature maps 进行上采样得到高分辨率的图片。这里修改其卷积来实现更大的感受野（其实就是改采样因子和滤波器大小）：
![](image/Pasted%20image%2020230928220933.png)
左边是用在图像上的，右边是 WaveGAN。
> Discriminator 也以相同的方式进行修改。

最终会生成一个 16384 点的输出（大概就是 1s），到这就已经可以产生 reasonable audio 了。

### Phase Shuffle

由于转置卷积的棋盘效应，在音频中，会有一些  pitched noise。

而且这些 artifact frequencies 总是出现在一个特定的 phase 中，使得判别器很容易就区分出来了。

于是提出 phase shuffle 的操作，也就是随机打乱每一层的 activation 输出的 phase，打乱的数量是从 $-n,n$ 之间的均匀分布，下图显示了 $n=1$ 的情况：
![](image/Pasted%20image%2020230928223611.png)

而且只在 discriminator 中使用这个操作，其实就是变相提高了 discriminator 区分真假的难度。

### SPECGAN：生成 semi- invertible 的 spectrogram

设计 spectrogram representation，是 128x128 维的二维特征：
+ 对输入做 STFT，得到 0-8KHz 的 128 frequency bins
+ 计算对数幅度谱，然后做归一化
+ 然后 clip 到 $3\sigma$ 附近，最后再 scale 到 $[-1,1]$ 之间

然后把得到的特征当作图片，直接送入 DCGAN，最后采用的是  Griffin-Lim algorithm 得到波形的。