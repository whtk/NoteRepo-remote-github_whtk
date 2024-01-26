1. 提出一种重放检测系统，包括：特征提取、DNN模型和分数融合
2. DNN 模型包括：SENet、DenseNet 和提出的信道一致性模型 DenseNeXt，最后对三个 DNN 模型的分数进行融合，最终在 PA 数据集上实现了 SOTA

## Introduction

1. ASVspoof 2019 的研究主要可以分成两类：
	1. 特征角度：CQCC、LFCC、MFCC、IMFCCs、GD、x-vector
	2. 模型角度：统计模型和 DNN
2. 本文构建一个 DNN 模型，贡献有：
	1. 在欺诈检测中首次引入频谱增强和 DenseNet
	2. 提出了基于 DenseNet 和 ResNeX 的新模型
	3. 相比于 baseline，融合效果很好


## 方法

### 特征
> 这一节介绍了不同的特征（属于是水篇幅了）

### DNN 模型

采用 SEnet34 作为baseline，然后引入 DenseNet，最后提出 信道一致性DenseNet。

#### SEnet

采用了 [[ASSERT- Anti-Spoofing with Squeeze-Excitation and Residual neTworks 笔记]] 论文中的结构，squeeze-and-excitation 的原理见论文 [[SENet]]。

#### DenseNet

采用 CV 中的论文 [[densenet]] 的模型，具体原理见原论文。

DenseNet 模型如图：![[Pasted image 20221121210612.png]]

#### Channel consistency DenseNeXt

DenseNet 的问题在于 bottleneck 层数和特征图的增加，消耗大量的内存和计算，且限制了模型的深度。

提出 DenseNeXt 结构，如图：
![[Pasted image 20221121210827.png]]堆叠的 bottleneck 层被多个顶部和底部模块取代，其结构为：![[Pasted image 20221121211317.png]]

 的结构都是这个，不过 bottom modules 包括 channel reduction。

在 DenseXt 中，包括 4 个 top modules 和 4 个 bottom modules，从上到下 feature map 在增大，但是数量在减小。reduction 层可以减少参数，且每个 DenseXt 模块的输入特征图和输出特征图的数量一样。

总之：相比于 DenseNet，使用了更少的 feature map，然后卷积层也是并行的，最后使用 SE 模块可以提高性能（牺牲参数）。

### 谱增强

从257维的特征中随机选择连续30个置 0 。

### 分数融合

对 SEnet34、DenseNet和信道一致性DenseXt 三个模型的得分进行平均。

## 实验设置

baseline：CQCC-GMM 和 LFCC-GMM

网络参数详见论文。

## 结果

1. 不同模型的性能对比：![[Pasted image 20221121220101.png]]表1 是 logspec 特征，DenseNeXt 效果最后；表2 是模型参数和计算量
2. DenseNeXt 网络，但是三个不同的特征：![[Pasted image 20221121220927.png]]结论：logspec优于其他特征。
