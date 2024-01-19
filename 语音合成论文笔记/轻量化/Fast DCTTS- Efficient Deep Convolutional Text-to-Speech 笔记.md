> ICASSP 2021，Handong Global University，韩国

1.  提出 Fast DCTTS，可以在单个 CPU 线程上实时合成音频，包含了精心设计的轻量化网络
2. 还提出一种新的 group highway activation，可以在计算效率和门控机制的 regularization effect 之间权衡
3. 还引入了一个新的称为 elastic mel-cepstral distortion (EMCD) 来测量输出 mel 谱的保真度
4. 相比于 baseline，MOS 从 2.62 提高到 2.74，且只需 1.76% 计算量和 2.75% 的参数

## Introduction

1. 提出 Fast DCTTS
2. 实验发现，depthwise separable convolution 并不能提高 TTS 的速度，尽管理论上可以减少计算量
3. 同时观察到，将 DCTTS 中的 highway activation 替换为 residual connection 会降低性能

## 相关工作（略）

选择 [DCTTS- Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention 笔记](DCTTS-%20Efficiently%20Trainable%20Text-to-Speech%20System%20Based%20on%20Deep%20Convolutional%20Networks%20with%20Guided%20Attention%20笔记.md) 作为 baseline。

## 优化技术

### 计算优化技术

Depthwise separable convolution 用 2D 和 1D 卷积来近似 3D 卷积，从而可以降低计算复杂度。

DCTTS 采用的 highway activation layer 如下：
$$y=T(x,W_T)H(x,W_H)+C(x,W_C)x$$
其中 $x,y$ 分别为输入输出特征图，$T,C$ 分别是两个 gate，且加起来为 1，这两个 gate 可以帮助神经网络学习，但是增加了复杂度。

于是采用两种其他的 highway activation，其一是 residual connection，此时 $T(x,W_T)=C(x,W_C)=1$，得到的模型称为 Residual DCTTS，但是 Residual DCTTS 会极大地降低语音质量，尤其会极大地增加 skipping 和 repeating 的 error。

于是提出另一种称为 group highway activation 的方法，将特征元素分组，gate vector 的大小降低为 $\frac 1 g$，$g$ 为 group size，通过调整 $g$，可以实现 计算效率和门控机制的 regularization effect 之间权衡。

减少网络的尺寸也可以提高合成速度，主要是减少维度和层数来实现。

网络剪枝也可以减少尺寸和计算量，训练后通过剪枝算法移除重要性比较低的 unit。本文用的是已有的剪枝方法，做了一点小的修改的来适应 group highway activation。

### fidelity 提升技术

positional encoding 可以在特征向量中引入相对位置信息。本文采用 [Transformer-TTS- Neural Speech Synthesis with Transformer Network 笔记](../Transformer-TTS-%20Neural%20Speech%20Synthesis%20with%20Transformer%20Network%20笔记.md) 中的 scaled positional encoding，然后相加引入位置信息 $\begin{aligned}x'_i=x_i+\alpha PE(pos,i)\end{aligned}$。

还采用了 scheduler sampling 技术，训练自回归模型时，teacher forcing 用的是 GT mel 谱而非上一帧的预测输出，随着时间进行逐渐增加采用前一帧的概率。

## elastic mel-cepstral distortion (EMCD)

对于没有被对齐的mel 谱，测量两个mel 谱之间的距离并不简单（例如存在 skipping 或 repeating 问题时），提出一个新的评价指标 EMCD，为考虑对齐时候的 mel 谱 之间的差异：
$$\begin{aligned}D(i,j)&=w_m\times MCD(x_i,y_j)\\&+min\big\{D(i,j-1),D(i-1,j),D(i-1,j-1)\big\}\end{aligned}$$
Mel cepstral distortion(MCD) 可以很有效地测量两个语音信号之间的感知距离，定义为：
$$\begin{aligned}MCD(i,j)=\sqrt{2\sum_{d=1}^D(x_d[i]-y_d[j])^2},\mathrm{~where~}i=\{1,...,T_{syn}\},j~=~\{1,...,T_{gt}\}\end{aligned}$$
其中 $x,y$ 为 MFCC 序列，$T$ 代表对应序列的长度，将其和 dynamic time warping (DTW) 组合来拓展 MCD，得到 EMCD。
> 和 MCD-DTW 很类似，但是对 horizontal, vertical, and diagonal transitions 有不同的惩罚权重，因此更有效。


## 实验