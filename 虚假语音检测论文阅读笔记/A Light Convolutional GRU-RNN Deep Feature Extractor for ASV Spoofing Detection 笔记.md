
1. 提出 Light Convolutional Gated Recurrent Neural Network (LC-GRNN) 进行前端特征提取
2. 将 light convolutional 层在帧级提取特征的能力与GRU-RNN 学习 序列深度特征的长期依赖能力相结合
3. 用于 ASVspoof 2019，相比于 baseline 有很大提升


## Introduction

1. 目的是开发一个通用框架，旨在检测不同的欺骗攻击，即TTS、VC和重放攻击
2. [[Robust Deep Feature for Spoofing Detection - The SJTU System for ASVspoof 2015 Challenge 笔记]] 提出深度特征提取，从 DNN 的内层提取 feature embedding。
3. DNN 有强大的非线性建模和辨别能力，不仅可以作为后端分类，而且有利于特征提取，其架构可以决定 反欺骗系统的性能
4. 本文提出混合 LCNN 和 RNN 的架构，将LCNN在帧级提取辨别特征的能力与基于GRU的RNN学习后续深度特征的长期依赖性的能力相结合。本文是首次将 DNN 用于欺骗检测的工作

## 系统描述

[[A Deep Identity Representation for Noise Robust Spoofing Detection 笔记]] 提出了混合 CNN-RNN 的架构来获得话语的 spoofing identity vector，本文更进一步，把 CNN 替换成 LCNN，从而：
+ 在帧级提取鉴别性特征
+ 学习长期依赖
+ 将帧级的特征提取和话语级的身份向量集成到单个网络中

LCNN 已经被用于重放攻击检测 [[Audio replay attack detection with deep learning frameworks 笔记]] 。

LC-GRNN 的框架如图：![[Pasted image 20221124150840.png]]
在每个 timestep，模型处理 $W$ 个连续帧的上下文窗口，每个 timestep 窗口向前移动 $\delta$ ，总的 time step 为 $(T-W) / \delta$ ，$T$ 为帧数。LC-GRNN 有 $N$ 个循环层作为分类器，来判别输入话语的 真 或者 假（假 有很多种），最后一个 time step 的输出和最后的循环层的输出送到具有MFM激活的全连接层，获得 spoofing identity vector，训练时，该向量最终通过另一个全连接的层，该层具有 $K+1$ 神经元的softmax激活，以区分真实类和 $K$  个欺骗类。

和经典的 RNN 不同，LC-GRNN 的 hidden state $\mathbf{h}_t^n(t=1, \ldots,(T-W) / \delta ; n=1, \ldots, N)$ 通过当前特征 $\mathbf{x}_t^n$ 和前一个 state $\mathbf{h}_{t-1}^n$ 进行卷积得到 ，由于能够检测欺骗攻击的大多数线索都可以在某些频带中找到，通过用卷积替换 GRU 中的全连接。优点在于可以在帧级提取更多的鉴别性特征。

模型更细节的结构如图：![[Pasted image 20221124152329.png]]GRU 的三个门都通过 LCNN 来实现（原来是全连接层），每个LCNN块由一个或两个LCNN层组成，然后通过 MFM 激活来将特征图减半。LC-GRNN 的每个 time step 为帧级深度特征提取器，为W个连续帧的每个上下文窗口提供N个状态（特征）向量。

当只用单层的 LCNN 时，更新门和重置门的计算过程为：$$\begin{aligned}
&\mathbf{z}_t^n=\sigma\left(\operatorname{MFM}\left(\mathbf{W}_z^n * \mathbf{x}_t^n+\mathbf{U}_z^n * \mathbf{h}_{t-1}^n\right)\right) \\
&\mathbf{r}_t^n=\sigma\left(\operatorname{MFM}\left(\mathbf{W}_r^n * \mathbf{x}_t^n+\mathbf{U}_r^n * \mathbf{h}_{t-1}^n\right)\right)
\end{aligned}$$
其中，$\text { * }$ 表示卷积操作。这些卷积层可以被解释为 filter banks ，其被训练和优化以检测来自欺骗语音的伪影。类似，$$\tilde{\mathbf{h}}_t^n=\tanh \left(\operatorname{MFM}\left(\mathbf{W}_{\tilde{h}}^n * \mathbf{x}_t^n+\mathbf{U}_{\tilde{h}}^n *\left(\mathbf{r}_t^n \odot \mathbf{h}_{t-1}^n\right)\right)\right)$$
$\odot$ 表示 element-wise 的乘法，模型参数在所有的 time step 共享。

## 实验 

架构参数：![[Pasted image 20221124153423.png]]$N=3$ ，Adam 优化器，$lr=3\times 10^{-4}$ ，dropout=0.6（FC1），输入进行均值和方差归一化，后端使用 SVM、LDA 和 PLDA，输出是得分，有些模型还根据先验进行了得分归一化。

数据集：ASVspoof2019 

特征：window 16ms，帧移 4ms，256 维的 logspec 特征，$W=32, \delta=12$，

## 结果

1. 在 ASVspoof 2015和2017 的结果：![[Pasted image 20221124154825.png]]
2. 在 ASVspoof2019的结果：![[Pasted image 20221124154842.png]]
效果一个字：好