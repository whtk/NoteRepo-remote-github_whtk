> 2019 年

1. 通常说话人分类器是采用 softmax + 交叉熵损失进行训练的，但是这种损失并不能实现类间可分和类内一致
2. 本文将三种不同的 margin loss 引入到 DNN 中，不仅可以将不同的类别分开，而且要求类别之间有确定的边界 margin，并且 margin 是获得更有区分性的 embedding 的关键
3. 在 VoxCeleb1 和 SITW 上进行的实验。

## Introduction

1. 大多数的 SR 使用的损失都是 softmax+交叉熵
2. 本文研究了三种 loss：
	1. angular softmax loss，A-Softmax loss
	2. additive margin softmax loss，AM-Softmax loss
	3. additive angular margin loss，AAM-Softmax loss
	4. 发现 margin 对于学习有区分性的特征非常重要，且能带来巨大的性能提升

## DNN 说话人 embedding

使用的系统基于 x-vector，采用 SGD 进行优化。

## 损失

原始的 softmax loss 和 其他三种 loss 的原理见 [[损失函数]]。

## 实验

在 VoxCeleb1 训练， SITW 上测试。

### 基本实验设置

为了增加训练数据的量和多样性，采用数据增强来添加 noise、music、babble 和 reverberation。

30 维的 MFCC 特征，10ms 帧移，25 ms window size，采用长为 3s 的滑动窗口进行均值归一化。同时采用了 VAD 去除静音段。

模型采用下表的：![](./image/Pasted%20image%2020230325165723.png)
训练完成后，segment6 的输出作为最终的 embedding。

在长为 2-4 s 的音频上进行训练，通过 random cutting 得到。8 GeForce GTX 1080Ti GPUs 训练，每张卡 batch size 64，学习率从 0 逐渐增加到 1e-4（在前 65,536 batch，对于每个 GPU 来说就是前 8,192 batch），训练 3 个 epoch，momentum 0.7, weight decay 1e-5，maximum gradient norm 1e3。

采用标准的 PLDA 作为后端评分。

### VoxCeleb1 

在 VoxCeleb2 的所有 + VoxCeleb1 的训练集进行训练，共计 1,277,503 条语音。同时保留 1,000,000 条增强语音，在去除静音段后，最终 2,128,429 条数据。采用 EER 和 minDCF 作为指标（$s=32$）：![](./image/Pasted%20image%2020230325170536.png)
> 注：第一行是用 kaldi 训练的，第二行是 pytorch。

加入 margin 之后效果可以提升进 30%。

如果在 VoxCeleb2 development 上训练，没用数据增强，测试集则在 两个修改的 VoxCeleb1 数据集中进行：![](./image/Pasted%20image%2020230325171120.png)

### SITW（略）

