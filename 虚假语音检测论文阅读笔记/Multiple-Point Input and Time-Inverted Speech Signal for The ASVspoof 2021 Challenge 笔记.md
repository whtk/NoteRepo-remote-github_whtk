
1. 本文提出以下两点来构建重放攻击检测系统：
	1. multiple-point input for convolutional neural networks
	2. 使用时间反转语音信号的相位谱
2. 使用了 四种基于 CNN CNN的网络和五种组合方法构建了几个子系统，最终提交时对所有子系统进行融合

##  Introduction

1. 欺骗检测是基于这样一个事实，即真实话语和欺骗话语之间的频率属性存在差异
2. 本文只针对 PA 系统
3. 本文使用了 四种 基于 CNN 的网络，同时使用了两种已经被提出的方法：
	1. 在 CNN 中使用 multiple-point input 来增加更多的信息，来自作者之前的论文 
	2. 使用语音信号及其反转之后的信号的 相位谱，通过生成不可见的类内条件来减少类内方差，也是来自作者之前的论文
4. 本文就是将前面两篇直接结合。。。

> 这就能发论文了吗。。。


## 系统描述

### Multiple-point input for CNN
>多点输入是为了克服使用CNN时处理可变长度特征（例如，来自语音的声学特征）的挑战。

之前用于处理语音变长特征的技术通常是分割和填充，但是这样会导致部分信息丢失，于是提出了 基于双向特征分割的多点输入CNN，如图：![[Pasted image 20221213152956.png]]
通过把特征正向和反向分割从而获得两个 segments，分别是 正向 $\left\{\mathbf{F}_i\right\}_{i=0}^{N-1}$ 和 反向 $\left\{\mathbf{B}_i\right\}_{i=0}^{N-1}$ ，其中每个 segment 的长度都是 $M$，然后把第 $i$ 个特征 $\mathbf{F}_i ,\mathbf{B}_i$ 组成一对。


CNN 的输入都是一对特征对，每对中的两个 segment 通常涵盖不同的时间范围，无形中增加了输入的信息。

### 时间反转语音信号的相位谱

由于相位谱具有时间反转特性（和幅度谱不同）：当信号的时间反转时，其相位谱的值会发生变化。

因此，当时间顺序颠倒时，与类内变化相关的身份（例如，短语、说话人和内容信息）被改变，但与类间变化相关的那些身份（例如播放和记录设备的信息）没有改变。

于是使用 原始信号 $x(n)$ 和反转信号 $\tilde{x}(n)$ 的相位谱如图：![[Pasted image 20221213154223.png]]
最终得到两个相位谱 $\boldsymbol{\tau}$ 和 $\tilde{\boldsymbol{\tau}}$ ，CNN 正好将这两个特征同时作为输入，但是此时包含的时间范围要一致。

### 组合方法

提出了五种组合方式，如图：![[Pasted image 20221213154534.png]]
其中，a 是传统的方法。

b 中，两个特征在输入的 channel level 进行拼接，也就是得到 $2 \times T \times D$ 维度的特征，最后导致 CNN 的第一层的参数翻倍。

c 是 embedding level 的组合，其中的 embedding 是 global average pooling 层的输出。两个特征输入到同一个 CNN 网络中，然后记过 GAP 之后将两个 embedding 进行组合，有三种组合方法：
+ concat
+ vmax（element-wise maximum）
+ vmean（element-wise average）

d 是 map level 的组合，feature map 表示 CNN 最后一层的输出，由于 CNN 模型是共享参数，所有直接进行 fmax 操作来组合特征图。

## 实验

使用了四种基于 CNN 的网络：SE-ResNet34、DenseNet-121、具有0.5x输出信道的ShuffleNetV2和深度乘数为1.0的MNASNet。

建立 2x5x4=40 个系统，使用 Adam 优化器。

结果如图：![[Pasted image 20221213163828.png]]
数值是通过对五个组合的子系统进行融合得到的，最后一行是所有系统的得分融合结果。
说明：
1. Multiple point input 效果好于时间反转的相位谱
2. 幅度谱比相位谱更有效，但是相位谱也包含了一些重放检测的信息

最终提交的结果：![[Pasted image 20221213164200.png]]
结论：效果很好，但是dev和 eval之间的差距很大，且如果是基于时间反转的方法时这个差距更大，最终结论是，仅使用模拟的语音仍然难以为真实环境构建鲁棒系统。