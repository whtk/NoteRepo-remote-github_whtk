
1. 提出了 lightweight Time Delay Neural Network 架构，使用 focal loss 来处理类不平衡问题，同时强调 hard-to-classify 样本
2. 使用 NN 作为 embedding extractor，基于这些 embedding 提出 one-class 高斯分类器

## Introduction

本文探讨了不同的神经网络结构和数据增强策略的使用，提出了以 focal loss 为目标函数，对具有不同输入特征的分类器进行线性融合的方法

## 数据增强

> 给出了用于 PA 和 LA 的数据增强方法。

LA 任务中的数据增强很重要，能够实现对未知编码和伪影的鲁棒性。

### PA 任务

分别以原始速度0.9和1.1生成了2个速度扰动版本，并且创建了包含原始音频和房间脉冲响应（RIR）的混响版本。因此，得到的训练集的大小是原始训练集的4倍。

### LA 任务

提出多种 transformations 来模拟 编解码器和 信道效应，分为两类：
1. Multimedia transformations：
	1. MP3 encoding
	2. AAC encoding
	3. OGG encoding
2. telephony transformations：首先下采样到 8k，然后进行 transformation 然后上采样到 16k
	1. a-law encoding
	2. u-law encoding
	3. g.729 encoding
	4. energy-based VAD

最终数据集是原始数据集的五倍。

## 模型

### PA

基于 TDNN 架构，但是网络深度和输入特征不同。

同时也开发了 基于 TDNN 的 Complementary systems，细节见论文。

TDNN 的具体原理 见 [[x-vector、i-vector]]。

#### 主系统

主系统使用 MFCC 作为特征。

主系统用的下表（TDNN-L）：![[Pasted image 20221215095552.png]]
使用 focal loss 作为损失函数，适用于类不平衡任务，把关注点放在难以分类的样本上。

#### complementary 系统

主要用于捕获信息辅助主系统，其架构如图：![[Pasted image 20221215095830.png]]
所有的 DNN 使用相同的 TDNN 架构，此外还探索了使用这些系统来作为特征提取器（也就是不分类），然后用其他分类器的方法。

其他都差不多，但是这里训练用的是 交叉熵损失。

实现了以下几种系统：
+ tdnn-MFCC
+ tdnn-logFBE
+ tdnn-CQCC
+ tdnn-SCMC
+ Embedding：使用从 tdnn-MFCC 提取的特征训练下游分类器，如 SVM、高斯线性分类器

### LA

LA 主要关注数据增强。探索了三个模型，RawNet2、LFCC-LCNN 和 Lightweight TDNN。

#### Lightweight TDNN

和 PA 中的 TDNN 差不多，但是参数更少，如图：![[Pasted image 20221215100547.png]]
也使用了 focal loss 作为损失函数。

## 结果

### PA

PA 主系统结果：![[Pasted image 20221215102041.png]]
L-TDNN+focal loss 的效果最好。

baseline 对比：![[Pasted image 20221215102351.png]]
所有的 complementary 系统都有利于系统融合，且是相同权重的融合。

将 TDNN 作为 embedding extractor 的结果：![[Pasted image 20221215102749.png]]
后端是 GLC的小效果最好，GLC 本质是一种 one class 的分类器。这在提供对未知攻击的鲁棒性或避免过拟合特定攻击方面可能具有积极的意义。

### LA

数据增强的效果：![[Pasted image 20221215103043.png]] 
两种增强都有改进，结合起来更强。但是增加训练集并没有改善效果。

和 baseline 比：![[Pasted image 20221215103347.png]]
数据增强有用，效果最好的是 RawNet2。

