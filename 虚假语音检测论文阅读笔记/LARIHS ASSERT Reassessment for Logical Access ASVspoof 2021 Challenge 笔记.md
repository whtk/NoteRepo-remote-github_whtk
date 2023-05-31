> JHU 2021 ASVspoof 的系统

1. 使用 ASSERT 系统
2. 研究 LFCC 和 log-linear filter banks 在 SENet 上的性能，同时使用了 angular margin-based softmax 和 large margin cosine loss
3. 提出使用 MFM 激活函数的 SENet

## Introduction

1. 已有研究表明，使用 A-softmax 和 LMLC 损失可以获得较高的性能
2. 本文基于先前的研究 [[ASSERT Anti-Spoofing with Squeeze-Excitation and Residual neTworks 笔记]]，提出不同的 版本：
	1. 不同的输入特征
	2. 引入 MFM、A-softmax、LMLC，修改 SENet backbone 模型
3. 最终提交的模型是 MFM-ASSERT18

##  ASSERT 变体
> 灵感来自于 LCNN

### 输入特征相关

1. 将LFCC 和 LFB 替换为 logLFB
2. 对 logLFB 应用 2D DCT，进行 CMVN 生成 GM-LFB 特征

如图：![[Pasted image 20221222212459.png]]

也探索了基于 logLFB 使用 local binary pattern（LBP）生成 logLFB textrogram ：![[Pasted image 20221222212823.png]]

### SENet 相关

提出了将 SENet34 作为 backbone 的ASSERT的修改，同时使用前面说的两种损失函数进行训练，定义为：
$$B_{\text {Asoftmax }}= \begin{cases}C_1: & \cos \left(m \theta_1\right) \geq \cos \left(\theta_2\right) \\ C_2: & \cos \left(m \theta_2\right) \geq \cos \left(\theta_1\right)\end{cases}$$
$$B_{L M C L}= \begin{cases}C_1: & \cos \left(\theta_1\right) \geq \cos \left(\theta_2\right)+m \\ C_2: & \cos \left(\theta_2\right) \geq \cos \left(\theta_1\right)+m\end{cases}$$
其中，$C_1,C_2$ 表示真实和虚假类。

同时考虑了使用 BCE 损失 + SENet18 + MFM 的 thin-ResNet34 backbone。

## 实验和结果

特征提取的具体配置见论文。

不同特征在 LA 上的效果：![[Pasted image 20221222214924.png]]
不同的特征相比于 logspec 都有改进。

损失函数的比较：![[Pasted image 20221223095047.png]]
A-softmax 在2019的 eval 效果是最好的。LMLC 在 2021 的效果更好。


不同 backbone 和 最终提交的效果：![[Pasted image 20221223095349.png]]


progress step 和  post-eval step 之间差距很大，主要原因是编解码器的影响，导致模型的泛化性不一致。



