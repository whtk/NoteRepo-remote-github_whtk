> EUSIPCO 2018

1. 声音事件检测（SED）是从给定的音频信号中检测环境声音事件，包括对事件进行分类以及估计其开始和结束时间（类似于SD）
2. 提出采用 CapNet 架构，训练一个可以隐式学习全局一致性的网络，提高模型的泛化性能

## Introduction

1. SED 是对是对音频中的声音事件进行分类和定位的任务，为每个检测到的事件分配一个类别标签以及开始和结束时间
2. 一般都采用监督学习方法，但是存在一些过拟合的问题
3. 本文采用 capsule network 来减轻过拟合，同时 capsule routing 可以看成是一种 attention 机制，当训练数据是 weakly labeled 的时候、或者 开始和结束时间不知道的时候，注意力机制对于 SED 特别有用，而 routing 则通过权衡 low level 和 high level capsule 之间的关系来实现这种 attention
4. 本文就专注于这种 weakly-labeled 事件检测，但是也适用于 strongly-labeled 的场景

## Capsule Routing（略）

## 方法
![](./image/Pasted%20image%2020230203192640.png)
分为两个部分：特征提取和检测

### 特征提取

提取固定维的 logmel 特征（因为 MFCC 相比于 logmel 的信息更少，进行了 DCT变换）。

### 网络架构

上图给出了模型架构。网络的初始层为 gated convolution（可以提高性能），每个 block 有两个这种层，一共三个 block，每个 block 之后进行 max-pooling 使维度减半。

初始层之后是 primary capsule layer，输出为 $T\times \cdot \times U$，其中 $\cdot$ 代表可以从前面进行推断（应该就是特征的维度）。$U=4$ 代表 capsule size，然后把 $1 \times \cdot \times 4$ 看成是一个 time slice，把他们看成是 capsule layer 和temporal attention layer 的输入：
+ capsule layer 包括 $U=8$ 的 $L$ 个 capsule，$L$ 为声音类别的数量，因为输入也是 capsule，所以采用 dynamic routing 进行聚合，最后计算每个输出 capsule 向量的模，得到每个 time slice 的激活 $\mathbf{o}(t) \in \mathbb{R}^L$
+ temporal attention layer 用于实现注意力机制，也是连接了 $L$ 个 unit和 sigmoid 激活，输出为 $\mathbf{z}(t) \in \mathbb{R}^L$，然后在时间维度将这两个向量合并：$$\begin{aligned}
y_l & =\frac{\sum_{t=1}^T o_l(t) z_l(t)}{\sum_{t=1}^T z_l(t)} \\
& =\mathbb{E}_{t \sim q_l(t)}\left[o_l(t)\right],
\end{aligned}$$，此时可以将 $y_l$ 看成是根据 TA 得到概率分布的加权的 capsule 的模。然后选择阈值 $\tau_1$，如果 $y_l>\tau_1$ 则声音事件 $l$ 在时间 $t$ 发生，而为了计算起止时间，针对 $o_l(t)$ 采用另一个阈值 $\tau_2$，同时采用形态学的闭操作减少零星的噪点。

## 实验

数据集：DCASE 2017 weakly-labeled dataset ，Task 4，包含重叠

评估指标：precision, recall, 和 F-scores

标签分类和事SED结果：![](./image/Pasted%20image%2020230203201038.png)
提出的是 GCCaps，和其他两个对比，对于分类任务效果是最好的；
在 SED 中，GCCaps的表现略好于GCRNN。EMSI 其实是最好的，但是这个系统用了很多集成技术。

同时使用 capsule 还可以降低过拟合：![](./image/Pasted%20image%2020230203201517.png)

