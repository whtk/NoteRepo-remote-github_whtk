
> JHU 2019 ASVspoof 的系统

1. 本文使用 SE 和 residual network 进行 反欺骗检测
2. ASSERT 有四个组成部分：特征工程、DNN模型、网络优化和系统组合，其中 DNN 模型是 SE 模块的变体
3. ASSERT 是 2019年ASVspooof 两个子挑战中性能最好的系统之一

## Introduction

反欺诈的研究可以分为三类：
+ 特征学习
+ 统计建模
+ DNN

本文贡献有：
1. 对几种DNN模型在检测 RA、TTS和VC产生的欺骗攻击方面的有效性进行了实验。DNN模型基于 SENet 和 ResNet 的变体。是第一个引入带有 statistical pooling 的SENet和 ResNet 模型来解决反欺诈
2. 进行消融实验，包括 特征工程、网络优化 和 融合方案。相比于 baseline 取得了显著性能提升。

## ASSERT 模型

### 特征工程

声学特征：[[CQCC]] 和 logspec，都没有进行 VAD 和 归一化。

统一特征图（Unified Feature Map）：把所有的 utterances 乘以 $M$ 帧，然后分成长为 $M$ 帧的片段，片段可以有 $L$ 帧的重叠。每个 utterances 有多个 段，最终平均所有段的 CNN 输出，如图：![[Pasted image 20221129165035.png]]

Whole Utterance：还考虑了另一种特征工程方法，即用整个话语（变长输入）训练模型。

### DNN 模型

SE 网络：使用 [[SENet]] 进行欺诈检测。实现了两种变体：具有ResNet34 backbone 的SEnet34和具有 ResNet50 backbone 的SEnet50。使用 Unified Feature Map 进行训练。

Mean-Std ResNet： ResNet with pooling 可以实现和 [[x-vector、i-vector]] 相当的结果。因此，引入了ResNet with pooling。具体而言，采用Mean Std ResNet，在从ResNet34中提取帧级特征之后，在 time step 维度上估计平均值和标准差，以此来表示整个话语。由于 pooling 层考虑了变长的输入，用 Whole Utterance 训练Mean Std ResNet。同时使用了CQCC和logpsec。

Dilated ResNet：Dilated ResNet在每个 res block中包含 dilated convolution 层，如图：
![[Pasted image 20221129170143.png]]
将原始的 dilated residual block扩展到 multiple residual units。由于没有 pooling 层，只能接受固定大小的输入。和 SENet 相同的训练设置。

Attentive-Filtering Network （见 [[Attentive Filtering Networks for Audio Replay Attack Detection 笔记]]）：在 Dilated ResNet 前面用 attention-based feature masking，feature masking 包含四个下采样和四个上采样。下采样单元基于 max pooling 和d ilated convolution layers，而上采样单元基于卷积和bilinear upsampling layers。还通过将双线性上采样方法替换为转置卷积和自注意力机制来扩展原始的 AFN。和 SENet 相同的训练设置。

### 网络优化

训练目标函数：最直接的是进行二分类。但是 [[Replay spoofing detection system for automatic speaker verification using multi-task learning of noise classes 笔记]] 表明，通过对音频重放中的噪声进行分类，可以对网络进行优化，因此进一步使用多分类（输出为多类标签）来训练模型。在推理阶段，将 bonafide 类的对数概率作为 score。

Optimizer：Adam 优化器，$\beta_1=0.9, \beta_2=0.98$，weight decay 为 $10^{-9}$，前 100 个 step lr 线性增加，然后以 step 的平方根倒数成比例地降低（Transformer 的配置）。

### 融合

采用 greedy fusion 方法选择最优组合。使用Bosaris工具包通过 logistic regression 进行融合和校准。


## 实验

### baseline 

LFCC-FMM、CQCC-GMM、i-vector

### 实验设置

数据集：ASVspoof 2019 corpus
评估指标：min t-DCF 和 EER

### 结果：
1. 和 baseline 比：![[Pasted image 20221129172651.png]]结论是：logspec优于CQCC；最佳单系统基于SENet34。
2. 比赛的评估结果：单系统基于 SENet34，主系统基于前面提到的五个 DNN 模型的融合：![[Pasted image 20221129173009.png]]效果很好，不过在 LA 上出现过拟合。