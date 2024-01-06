
1. 本文研究了不同的池化方法和损失函数
2. 研究了随机加权平均的有效性，进一步提高欺骗检测系统的鲁棒性

## Introduction

1. 本文一共训练了三个系统，
	1. 第一个来自于 [[Generalization Of Audio Deepfake Detection 笔记]]，基于  large margin cosine loss 的 ResNet 系统
	2. 第二个系统为第一个的拓展，使用 learnable dictionary encoding 层来替代 均值和标准差池化层
	3. 第三个系统也使用LDE池化层，但在输出层中使用 Softmax 激活，使用 交叉熵损失进行训练

 最终提交的系统是三个系统的融合。

整体框架为：![[Pasted image 20221222203413.png]]

## 数据增强

使用三种数据增强：
1. 混响和背景噪声
2. 模拟音频压缩
3. 添加编解码器的传输效应


## 方法

### 特征 

linear filter bank （LFB），训练的时候进行 frequency masking。

### embedding 提取

ResNet18-L-FM 模型：ResNet-18 的变体，使用均值和标准差池化层来替代全局平均池化层，训练时使用 Large margin cosine loss 提高泛化性，最后进行二分类。

ResNet-L-LDE 模型：使用 LDE 替代 池化层。
LDE 如图：![[Pasted image 20221222204144.png]]
LDE 假设ResNet编码器输出的 frame level 的表征分布在 $C$ 个簇中，以有监督的方式学习编码器参数和 inherent dictionary。

ResNet-S-LDE 与ResNet-L-LDE 模型相同，但使用Softmax输出和交叉熵损失进行训练。

分类器：FC+BN+ReLU+dropout+Softmax，使用 SWA 技术进行训练。

## 实验结果ResNet-L-LDE

LA：![[Pasted image 20221222205441.png]]
结论：ResNet-L-LDE 效果最好；SWA 可以提供更好的泛化性能；

DF ：![[Pasted image 20221222205805.png]]




