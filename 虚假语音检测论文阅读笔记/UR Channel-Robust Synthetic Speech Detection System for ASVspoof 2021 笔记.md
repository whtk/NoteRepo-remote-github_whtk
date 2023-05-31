> 有代码

1. 提出了一个信道鲁棒的合成语音检测系统，专注于 LA 和 DF
2. 在传输编解码器、压缩编解码器和卷积脉冲响应使用声学模拟器来扩充原始数据集
3. 采用 Emphasized Channel Attention Propagation and Aggregation Time Delay Neural Networks （ECAPA-TDNN）作为 backbone
4. 将 one class 学习与 通道鲁棒的训练策略结合，进一步学习通道不变的语音表征

## Introduction

1. 本文主要针对合成语音检测（SSD）
2. 本文开发一个检测系统，用于解决 one-class 学习的泛化能力和通道鲁棒策略的鲁棒性问题，主要贡献有：
	1. 设计了几种数据增强方法
	2. 采用 ECAPA-TDNN 及其变体作为 DNN 架构（多说话人识别中最好的模型）
	3. 采取通道鲁棒训练策略来配合前面的数据增强方法

## 系统描述

### 数据增强

在 [[An Empirical Study on Channel Effects for Synthetic Voice Spoofing Countermeasure Systems 笔记]] 中，验证了 通道效应 是跨 数据集 性能下降的重要原因，而数据增强方法被证明可以有效地提高反欺骗系统的跨数据集性能。

本文为了解决 DF 中的压缩可变性和 LA 中的传输可变性，相应设计了不同的数据增强方法，pipeline 为：![[Pasted image 20221215203444.png]]
即 四种不同的编解码器+设备脉冲响应。

具体来说，对DF，使用 压缩增强，即对输入音频进行 mp3 和 aac 编解码；对 LA ，使用 传输增强，即模拟不同的传输信道（固话、蜂窝、VoIP等场景），使用对应的编解码方法。

然后和设备脉冲响应进行卷积得到最终增强后的数据。

### ECAPA-TDNN 模型

见论文 [[ECAPA-TDNN- Emphasized channel attention, propagation and aggregation in tdnn based speaker verification 笔记]]

本文通过改变卷积层的滤波器数量以及是否插入多层特征聚合（MFA）块和 通道与上下文相关的统计池（CCSP）修改原始模型，最终得到6个变体模型。

### 训练策略

采用 OC-softmax 作为损失函数，使用 one-class 学习的思想（[[One-class Learning Towards Synthetic Voice Spoofing Detection 笔记]]），同时采用了之前论文中提到的信道鲁棒性策略进行数据增强。

数据的平均幅度谱如图：![[Pasted image 20221215214734.png]]两个数据集的差距还是很大的，尤其是在高低频部分，因此有理由认为，使用信道鲁棒性策略可以提高性能。

为了适应前面使用的数据增强，设计了两个额外的分类器来针对编解码器和设备增强：$$\begin{aligned}
\left(\hat{\theta}_e, \hat{\theta}_s\right) & =\underset{\theta_e, \theta_s}{\arg \min } \mathcal{L}_s\left(\theta_e, \theta_s\right)-\lambda_1 \mathcal{L}_c\left(\theta_e, \hat{\theta}_c\right)-\lambda_2 \mathcal{L}_d\left(\theta_e, \hat{\theta}_d\right) \\
\left(\hat{\theta}_c\right) & =\underset{\theta_c}{\arg \min } \mathcal{L}_c\left(\hat{\theta}_e, \theta_c\right) \\
\left(\hat{\theta}_d\right) & =\underset{\theta_d}{\arg \min } \mathcal{L}_d\left(\hat{\theta}_e, \theta_d\right) .
\end{aligned}$$
其中，$\theta_c,\theta_d$ 分别为分类器的参数。

整个模型结构如图：![[Pasted image 20221215215324.png]]

## 实验 & 结果

特征：60 维 LFCC 特征，固定帧长 750

在 evaluation phase 的结果（详细的测试结果见论文图 4 和 6）：![[Pasted image 20221215221315.png]]

在基于 softmax 的系统中，输入质量越低，embedding norm 越小：![[Pasted image 20221215223545.png]]
可以看到，两个数据集之间分布明显不匹配，所以性能才差。

和 baseline 比：![[Pasted image 20221215223900.png]]
DF 中，大部分都较优，LA 则全部超过了 baseline。

不同的 backbone 模型：![[Pasted image 20221215224125.png]]
小的模型性能反而优于大的，说明大模型发生过拟合。

最后提交的是融合模型，优于单系统。
