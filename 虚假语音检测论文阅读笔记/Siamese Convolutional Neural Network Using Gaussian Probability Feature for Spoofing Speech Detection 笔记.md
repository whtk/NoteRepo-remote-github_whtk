1. 提出 1-D CNN ，输入为 GMM 的对数概率，不仅考虑了 GMM 的得分分布，还考虑了帧之间的局部关系
2. pooling 用于提取全局特征
3. 孪生网络基于两个GMM分别在真实和虚假语音中进行训练
4. ASVspoof 2019 中，相比于 baseline 有提升

## Introduction

在分类器方面：
1. [[Audio replay attack detection with deep learning frameworks 笔记]] 采用 LCNN，是 ASVspoof 2017 的最好性能
2. [[A Light Convolutional GRU-RNN Deep Feature Extractor for ASV Spoofing Detection 笔记]] 提出一种混合LCNN+RNN架构，LCNN 用于帧级别的特征提取和鉴别，GRU 用于学习长期依赖。
3. [[Deep Residual Neural Networks for Audio Spoofing Detection 笔记]] 提出用于音频欺骗检测的深度残差神经网络，分别处理MFCC、CQCC和频谱图输入特征
4. [[Attentive Filtering Networks for Audio Replay Attack Detection 笔记]] 提出注意力过滤网络，由基于注意力的过滤机制和基于ResNet的分类器组成，这种机制增强了频域和时域的特征表示

在孪生网络方面：
1. 孪生网络最初在 [[siamese]] 中提出
2. 各种孪生网络：
	1. Siamese CNN
	2. Pseudo-Siamese
	3. regularized Siamese deep network
	4. Siamese style CNNs
3. [[Deep Siamese Architecture Based Replay Detection for Secure Voice Biometric 笔记]] 提出深度孪生网络，将同一个类别的语音样本识别为相似的，不同类别的判为不同

经典 GMM 中，分数在所有的特征帧上累计，且高斯分量信息被丢弃；同时沿时间轴相邻帧之间的关系被忽略，本文提出了使用高斯概率特征进行欺诈语音检测的 1-d CNN 和 孪生网络。

## 高斯概率特征

### 高斯混合模型

1. GMM 的优点在于可以光滑近似任意形状的分布，对于 d 维 向量 $x$，使用似然函数的混合密度为：$$\mathrm{p}(\mathrm{x})=\sum_{i=1}^M w_i p_i(x)$$通过 $M$ 个 unimodal Gaussian densities 进行线性组合得到 densities $p_i{(x)}$ ，且满足多维高斯分布：$$\mathrm{p}_i(\mathrm{x})=\frac{1}{(2 \pi)^{D / 2}\left|\Sigma_i\right|^{1 / 2}} \exp \left\{-\frac{1}{2}\left(x-\mu_i\right)^{\prime} \Sigma_i^{-1}\left(x-\mu_i\right)\right\}$$ 通过EM算法来估计 GMM 的参数。

2. baseline 系统包含两个 GMM：分别用于估计建模真实和虚假语音，给定测试语句：$X=\left\{x_1, x_2, \ldots, x_N\right\}$ ，对数似然比定义为：$$\text { score }_{\text {baseline }}=\log p\left(X \mid \lambda_h\right)-\log p\left(X \mid \lambda_s\right)$$其中，$\lambda_h,\lambda_s$ 分别指真实的和虚假的 GMM 模型。

### 高斯概率特征

对于语音特征序列，GMM独立地累积所有帧上的分数，并且不考虑每个高斯分量对最终分数的贡献。此外，相邻帧之间的关系也被忽略。

提出的高斯概率特征对每个GMM分量上的分数分布进行建模。

对于第 $i$ 帧特征 $x_i$ ，对于GMM模型的分量 $j$ 的新特征为：$$\mathrm{f}_{i j}=\log \left(w_j \cdot p_j\left(x_i\right)\right)$$
然后进行平均和归一化。


## 孪生网络

### 1-d CNN

输入：GMM 分量上语音帧的对数概率
输出：二分类结果

结构：![[Pasted image 20221111211729.png]]
对于每个语音帧，都有 512 个对数概率。

同时卷积操作使得模型考虑了相邻帧的关系。

max-over-time pooling 操作作用于特征图中，可以捕获不同长度的输入语音。

使用不同窗口的多个过滤器获得多尺度特征，在倒数第二层进行拼接。


### 孪生网络

本文提出 孪生 CNN 如图：
![[Pasted image 20221111212329.png]]

包含两个相同的 CNN，输入分别是 真实和虚假的 GMM 模型的对数概率，在倒数第二层将两个 CNN 的结果进行拼接，最终输出二分类结果。

## 实验

数据集：ASVspoof 2019

特征：CQCC 和 LFCC

baseline：两个独立的 GMM（真实、虚假）

损失函数：交叉熵

优化器：Adam，lr=0.0001

batch_size：32

结果：
1. LA 数据集![[Pasted image 20221111213253.png]]

LFCC 的效果貌似更好。

2. PA 数据集![[Pasted image 20221111213349.png]]

效果就一个字：好！


## 未来研究

1. CNN+LSTM
2. 结合ResNet