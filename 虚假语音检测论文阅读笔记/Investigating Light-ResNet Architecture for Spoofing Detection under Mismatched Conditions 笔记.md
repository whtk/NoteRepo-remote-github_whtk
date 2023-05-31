1. 本文提出了一种新的  Light-ResNet 架构，有更强的泛化性
2. 在 residual 模块中引入 skip connection，可以训练更深的分类器，同时可以泛化到不匹配条件下
3. 效果优于 CQCC-GMM 和 AFN 网络

## Introduction

1. 问题：许多系统无法检测不可见的SS或VC攻击，或无法检测来自不同录制会话的重放攻击；解决：采用融合方法
2. 基于 DNN 的 [[Audio replay attack detection with deep learning frameworks 笔记]] 是2017年最好的重放检测系统，基于 DNN 的 SS/VC 检测 [[Spoofing Detection in Automatic Speaker Verification Systems Using DNN Classifiers and Dynamic Acoustic Features 笔记]] 也表现不错。
3. 同时，许多检测系统也使用了 residual 模块，如 [[End-To-End Audio Replay Attack Detection Using Deep Convolutional Networks with Attention 笔记]] 、[[Attentive Filtering Networks for Audio Replay Attack Detection 笔记]] ，但是纯 ResNet 架构在对抗重放攻击方面还未被证明有效，同时还要 SS/VC 的攻击。
4. 本文提出 Light-ResNet架构，输入为语谱图，在以下三个标准下进行评估：
	1. 相同数据集下的 training, development and evaluation 集
	2. 在一个数据集下训练在另一个进行 evaluation
	3. 跨攻击检测（在 重放、SS、VC 下训练的可以同时检测三个欺诈）


## Light-ResNet 架构

### residual 模块
![[Pasted image 20221114192946.png]]
1. 解决神经网络退化问题
2. 解决梯度消失问题
3. pre-activation 中，Batch Norm 和 RELU 在 CNN 之前，post-activation 中，对所有的卷积层的输入进行归一化，实验发现，pre 相比于 post ，过拟合更少。


### Light-ResNet 架构

1. 图像中的 ResNet 的结构不适用于欺诈检测
2. 提出 Light-ResNet 以减少参数（滤波器数量、层数、滤波器大小等）来避免过拟合
3. 上图中的 18 层 ResNet 中，每个 residual block 包括两个 residual modules，每个 modules 的数量都可变。
4. 输入为 语谱图，因为包含最具有判别性的信息


## 实验

特征：hanning 窗，长度为 25ms，帧移 10ms，dft 512点，使用了 CMVN。语音长度为 5s。

模型：
+ 首先是 7x7 卷积、3x3 池化
+ 四个 residual block（除了第一层卷积外，都使用 stride=2 进行下采样）
+ 然后是 global average pooling 层，对于每个 feature map 只有一个值输出
+ 最后是 全连接+sigmoid 进行分类
所有的卷积层都采用 l2 weight regulariser

优化器：Adam，lr=0.001，15 epoch

evaluation 集：ASVspoof 2015、2017 V2.0和BTAS 2016（replay）

baseline：CQCC+GMM 和 AFN

## 结果

滤波器数量和 EER 的关系：![[Pasted image 20221114195716.png]]整体趋势是，滤波器数量越少，效果越好。参数数量的减少使网络能够检测未知的重放攻击。

层数和 EER 的关系：![[Pasted image 20221114195905.png]]18层的 ResNet 效果最佳。

![[Pasted image 20221114200055.png]]
第一个表是和 baseline 的对比，Light-ResNet-34 效果最好。
第二个表是跨数据集性能对比：
+ Light-ResNet-34 比 ResNet-34 效果好，说明减少网络参数可以获得的性能改进
+ Light-ResNet-34 比 Light-ResNet-18，额外隐藏层使分类器能够更好地区分真假语音
当进行组合训练时，所有系统的性能都会下降，评估重放攻击时，所提出的Light-ResNet-18和34的性能优于AFN，ResNet-34和Light-ResNet-34在SS/VC攻击中表现特别好。

总而言之，Light-ResNet-34 系统在内部、跨语料库和统一条件下都能有最佳性能。