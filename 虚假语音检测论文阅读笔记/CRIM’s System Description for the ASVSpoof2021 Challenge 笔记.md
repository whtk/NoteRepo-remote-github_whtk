
1. 本文基于残差网络和TDNN训练了多个系统，其中训练集通过各种音频编解码器进行数据增强
2. 采用 higher order statistics pooling 方法提取 embedding，
3. 采用激活函数集成同时融合不同系统的分数来提高最终提交的系统的性能

## Introduction

1. 语音反欺骗的最新趋势是在原始信号/手工制作的特征的基础上，以端到端的方式使用深度学习架构，以区分真实和欺骗语音信号
2. [[Generalization Of Audio Deepfake Detection 笔记]] 引入基于 frequency mask 的数据增强和 使用 LMLC 的 ResNet  网络。
3. [[One-class Learning Towards Synthetic Voice Spoofing Detection 笔记]] 提出了具有ResNet18架构的 oc-softmax 损失
4. [[Light Convolutional Neural Network with Feature Genuinization for Detection of Synthetic Speech Attacks 笔记]] 提出了基于 LCNN 的 Feature genuinization
5. [[Audio Spoofing Verification using Deep Convolutional Neural Networks by Transfer Learning 笔记]] 使用迁移学习进行欺诈检测
6. 本文测试多个系统，外加使用多种编解码器和压缩算法进行数据增强，同时实验了 HOSP 技术，以充分考虑语音的分布属性，最后集成激活函数和模型得分加权实现最终的系统

## 系统描述

### 声学特征

使用了四种声学特征：
+ Product spectral cepstral coefficient（PFCC）：功率谱和群延时的乘积
+ LFCC
+ DCT-DFTspec
+ Log-linear filterbank energy（LLFB）

都使用了 delta 和 delta-delta 。

### 编解码器增强

两种编码：
+ 压缩编码：mp3, mp2, m4a, m4r 等
+ 电话编码：PCM µ-law, PCM a-law, speex 等

### 反欺诈系统

使用以下系统：
+ TDNN
+ SE-ResNet-18
+ SE-ResNet-18-ens5：包括两个激活函数，ELU 和 AReLU
+ SE-ResNet-18-ens6：包括六个激活函数：ReLU, leakyReLU，ELU，PReLU，AReLU

为了聚合网络 frame level 的输出，采用 attention pooling layer，在时间维度进行加权汇聚。

最后送到分类器中进行分类。

整个模型结构如图：![[Pasted image 20221223105552.png]]

### Higher Order Statistics（HOS）

pooling 层通常计算一阶和二阶特征，本文还实验了高阶矩 HOS，三阶和四阶，即：$$\begin{gathered}
\mu=\frac{1}{D} \sum_{d=1}^D h_d \\
\sigma=\frac{1}{D} \sum_{d=1}^D\left(h_d-\mu\right)^2 \\
s=\frac{1}{D} \sum_{d=1}^D\left(h_d-\mu\right)^3 \\
k=\frac{1}{D} \sum_{d=1}^D\left(h_d-\mu\right)^4
\end{gathered}$$
通过向系统提供关于帧级表示的分布的详细信息，将HOS用于池化可以提高欺骗检测性能。

### one-class softmax

$$L_{O C S}=-\frac{1}{N} \sum_{i=1}^N \log \left(1+e^{u\left(m_{y_i}-\hat{W}_0 \hat{\omega}_i\right)(-1)^{y_i}}\right)$$


## 结果

编解码器数据增强的效果：![[Pasted image 20221223110327.png]]

不同特征：![[Pasted image 20221223110738.png]]

融合系统的比较：略

> 反正就是挑效果最好的那个提交，各种系统融合就完事。