# A Light CNN for Deep Face Representation With Noisy Labels 笔记

1. 提出了LCNN，学习大量噪声标签下的大规模人脸数据上的嵌入
2. 在 CNN 中引入 最大特征图（MFM），不仅可以分分离噪声和信号，还可以在两个特征图之间起到特征选择的效果
3. 提出了一种语义自举的方法，使网路预测和噪声标签更加一致

## Introduction
1. 本文提出了 Light CNN 框架，用于从含有大量噪声标签的大规模数据集中学习人脸的深度表征。
2. 定义了一个MFM（最大特征图）操作，可以获取紧凑的表征，同时进行特征过滤选择
![](./image/Pasted%20image%2020221011201306.png)
3. LCNN 结构包括MFM、小型卷积滤波器和 Network in Network；为了处理有噪声的标记图像，提出了一种语义自举方法，通过预先训练的深度网络自动重新标记训练数据。
4. 本文贡献：
	1. MFM
	2. 根据AlexNet、VGG、ResNet 提出三种LCNN模型，在速度和存储方面都更好
	3. 语义自举
	4. 提出的256维表征的单一模型在各种人脸基准上获得了最先进的结果，参数更少、速度更快

## 相关工作
### 基于CNN的人脸识别（略）
### 带噪标签问题
有三种方法处理带噪标签：
1. 在分类任务中设计鲁棒损失
2. 识别错误标签来提高数据质量
3. 直接对噪声标签的分布进行建模

文中还总结了其他策略。


## LCNN 架构
本节首先提出了CNN的Max Feature Map操作来模拟神经抑制，从而为人脸分析和识别提供了一个新的Light CNN框架。然后，详细讨论了一种噪声标记训练数据集的语义自举方法。

### MFM
数据中的噪声信号会导致CNN输出有偏差的结果。在 ReLU 中通过阈值来分离噪声和信号，但是该阈值会导致一些信息的丢失，为了缓解这个问题，Leaky ReLU、Parametric RELU、Exponential Linear Units（ELU）被提出。

而在神经科学中，侧抑制（LI）能够增加视听觉反应的对比度和清晰度，并帮助大脑感知图像中的对比度。同时考虑 LI 和噪声，本文提出了用于卷积层的一种激活函数，其具有以下特征：
1. 能够分离噪声和信号
2. 当图像中存在水平边或者水平线时，对应于水平信息的神经元被激活，而对应于垂直信息的神经元被抑制
3. 对神经元的抑制是参数无关的，所以它不依赖于训练数据

为了实现上述特性，提出了 MFM，是 Maxout激活函数的拓展。但是MFM和Maxout的基本动机不同。Maxout的目标是通过足够的隐藏神经元来逼近任意的凸函数。使用的神经元越多，得到的近似结果越好。

Maxout网络的规模大于ReLU网络的规模。而MFM利用max函数来抑制少量神经元的激活，从而使基于MFM的CNN模型更轻量鲁棒。本文定义了两种类型的MFM操作来获得竞争特征图。

给定输入卷积层 $x^n \in \mathbb{R}^{H \times W}$，其中 $n=\{1,2,\dots,2N\}$，$W$ 和 $H$ 表示特征图的宽和高，MFM 2/1 操作结合两个特征图，选择元素最大的值作为输出：
$$\hat{x}_{i j}^k=\max \left(x_{i j}^k, x_{i j}^{k+N}\right)$$
其中，$2N$ 代表输入卷积层的通道数，$1\leq k\leq N, 1 \leq i \leq H, 1 \leq j \leq W$，如上图所示，MFM 操作之后的输出 $\hat{x} \in \mathbb{R}^{H\times W\times N}$ ，上式的梯度计算如下：
$$\begin{aligned}
\frac{\partial \hat{x}_{i j}^k}{\partial x_{i j}^k} &= \begin{cases}1, & \text { if } x_{i j}^k \geq x_{i j}^{k+N} \\
0, & \text { otherwise }\end{cases} \\
\frac{\partial \hat{x}_{i j}^k}{\partial x_{i j}^{k+N}} &= \begin{cases}0, & \text { if } x_{i j}^k \geq x_{i j}^{k+N} \\
1, & \text { otherwise }\end{cases}
\end{aligned}$$
通过 MFM2/1，从输入特征图中获取了 50% 的信息神经元。 

如果为了获得更具可比性的特征图，MFM3/2 操作可以定义为：输入三个特征图，删除最小的一个元素：
$$\left\{\begin{array}{l}
\hat{x}_{i j}^{k_1}=\max \left(x_{i j}^k, x_{i j}^{k+N}, x_{i j}^{k+2 N}\right) \\
\hat{x}_{i j}^{k_2}=\operatorname{median}\left(x_{i j}^k, x_{i j}^{k+N}, x_{i j}^{k+2 N}\right)
\end{array}\right.$$
其中，$x^n \in \mathbb{R}^{H\times W}, 1\leq n \leq 3N, 1 \leq k \leq N$，median() 函数为输入特征图的中值。MFM3/2 的梯度类似于上一个，当特征图 $x^k_{ij}$ 激活时 ，梯度为1，反之为0，最终可以从输入特征图中选择并保留2/3的信息。

### LCNN 框架
在CNN的背景下，MFM操作在生物特征识别中扮演着类似于本地特征选择的角色。MFM在每个位置选择最佳特征。在反向传播过程中，它会产生二元梯度（1和0）来激发或抑制一个神经元。

MFM 的两个优点，当 MFM 稀疏时，使用MFM可以或者更为紧凑的表示，同时在反向传播过程中 SGD 只对响应变量的神经元产生影响；另外，MFM 可以通过最多激活两个特征映射，从前一层卷积层中获得更具竞争力的节点。

文中给出了三种 LCNN 架构，具体结构如下图：
![](./image/Pasted%20image%2020221011215810.png)
![](./image/Pasted%20image%2020221011215819.png)
![](./image/Pasted%20image%2020221011215830.png)
### 用于噪声标签的语义自举
自举，也称为“自训练”，为估计样本分布提供了一种简单有效的方法。其基本思想是，通过重采样和从 原始标记样本到重新标记样本的推理，可以对训练样本的正向过程进行建模。它可以估计复杂数据分布的标准误差和置信区间，也可以控制估计的稳定性。

令 $x \in X$ 为数据，$t$ 为标签，基于 softmax 损失函数的CNN 预测条件分布 $p(t \mid f(x)), \sum_i p\left(t_i \mid f(x)\right)=1$，最大概率 $p\left(t_i \mid f(x)\right)$ 表明对应于最有可能的预测标签。基于该理论，作者提出了语义自举法，首先在 CASIA WebFace上训练Light CNN-9模型，并在原始带噪标签的 MS-Celeb-1M 数据集上对模型进行微调，然后使用训练好的模型根据条件概率 $p\left(t_i \mid f(x)\right)$ 重新标记带噪数据，再在重新标记的数据集上重新训练Light CNN-9，最后利用二次训练模型对原始的含噪标记数据集进行了重采样，形成了”降噪“后的MSCeleb-1M数据集。

## 实验（略）