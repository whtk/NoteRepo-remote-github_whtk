> TAAI 2018

1. 提出了一种使用噪声类的多任务学习的重放攻击检测系统
2. 将重放攻击导致的噪声定义为重放噪声，同时训练 DNN 用于重放攻击检测和重放噪声分类
3. 多任务学习是指，对包括播放设备、录制环境和录制设备的噪声以及欺骗检测进行分类

## Introduction

1. 本文所指的欺诈检测一律指重放攻击检测
2. 音频信号的信道噪声是由记录环境、记录设备和播放设备引起的，在 ASV 中，信道噪声会降低系统精度
3. 在欺诈检测中，重放攻击中的噪声非常重要，因为在重放过程中，会叠加录制环境、重放设备和录制设备的噪声；将在重放攻击期间附加的信道噪声定义为重放噪声
4. 欺骗检测的传统方法是二分类。考虑到重放噪声，本文了训练一个用于重放噪声分类和欺骗检测DNN以进行多任务学习，以同时在各种任务上训练网络。

## 相关工作

ASVspoof2017 中，[[Audio replay attack detection with deep learning frameworks 笔记]] 的效果最好，本文提出了一个类似于的结构，并修改使其可以用在噪声分类任务中，系统包含前端 DNN 特征提取和后端高斯分类网络。

[[Replay Attack Detection using DNN for Channel Discrimination 笔记]] 使用 channel discrimination，  通过训练DNN用于欺骗检测或 channel discrimination，并选择了仅用于 channel discrimination 的DNN。本文则使用多任务学习同时训练DNN进行欺骗检测和重放噪声分类。

### 前端 DNN

CNN ，尤其是 LCNN很强


### 后端高斯分类模型

二分类 DNN 输出的两种情况：
+ 输出为单个节点（sigmoid 激活），输出的值作为得分
+ 输出为两个节点（softmax 激活），输出中的一个作为得分

但是这样输出的值就不能被用做 reliability 的测度，[[Audio replay attack detection with deep learning frameworks 笔记]] 采用单个高斯建模，把最后一个 hidden layer 的 linear activation 作为输出的 code。
具体来说，训练DNN后，通过分别计算真实信号和欺诈信号 code 的平均值和标准差，对两个 单高斯模型 进行建模。测试阶段使用DNN提取 code。然后，将真实模型的对数概率和欺骗模型的对数概率的差作为分数。

## 方法

下图描述了欺诈语音的过程：![[Pasted image 20221201111539.png]]

将真实信号录制过程中的噪声称为 内部噪声，将重放过程中引入的噪声称为 重放噪声，公式描述为：$$\mathrm{y}_{\text {genuine }}(\mathrm{t})=\mathrm{x}(\mathrm{t}) * \mathrm{n}(\mathrm{t})$$$$\mathrm{y}_{\text {spoofed }}(\mathrm{t})=\mathrm{y}_{\text {genuine }}(\mathrm{t}) * \mathrm{P}(\mathrm{t}) * \mathrm{E}(\mathrm{t}) * \mathrm{R}(\mathrm{t})$$
其中，$*$ 代表卷积，假设内部噪声未知。重放信号和真实信号的区别就在于重放噪声。

提出 采用多任务学习 同时进行欺骗检测和重放噪声分类，系统架构为：![[Pasted image 20221201112328.png]]
对于每个重放噪声类添加一个节点，如果信号是真的，则该节点输出真。而在识别重放设备的任务中，有与重放设备的数量一样多的节点，并且有一个额外的节点指示输入信号是真实信号。我们期望通过为真实信号添加”真“节点（G），可以提高模型泛化能力。

## 实验

数据集：ASVspoo2017 1.0
特征：kaldi 提取，语谱图，固定 4s 的长度

使用的 baseline 配置见论文。

使用四个任务进行训练，训练完成后消除了输出层。
> 这里，输出层的总节点数为 $2+(4+1)+(8+1)+(7+1)=24$

多任务学习的所有损失都是一样的，节点信息如图：![[Pasted image 20221201113325.png]]

## 结果
![[Pasted image 20221201113420.png]]
表五结果表明，重放噪声的使用可以提高欺骗检测系统的性能。

图3显示了使用t-SNE的可视化结果。图中黄点和紫点分别表示真实信号和欺骗信号。可以发现，欺骗信号的分布是由信道差异而引起的多个簇，通过对重放噪类的训练，显著减少了与真实信号的重叠。