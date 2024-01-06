1. 提出 SimCLR，用于视觉表征下的对比学习
2. 表明：
	1. 在定义高效的预测任务时，数据增强很重要
	2. 在表征和 loss 之间引入可学习的非线性变换可以提高学到的表征的质量
	3. 相比于半监督学习，对比学习受益于更大的 batch size 和 更多的 step
3. 最终学到的表征是要超过半监督和自监督学习的

## Introduction

1. 表征学习分两个方法：
	1. 生成式，用于生成或者建模
	2. 判别式，用和有监督相似的目标函数学习表征，但是训练网络执行 pretext 任务，输入和标签都来自 un-labeled 数据
2. 本文提出 SimCLR，不仅性能更好，结构也更简单

## 方法

### 对比学习框架

SimCLR 通过 maximizing agreement between differently augmented views of the same data example via a contrastive loss in the latent space 来学习表征。

如图：
![](image/Pasted%20image%2020231116215716.png)
包含：
+ 随机数据增强：将同一个给定的数据样本随机转换成两个相关的 views，记为 $\tilde{\boldsymbol{x}}_i$ 和 $\tilde{\boldsymbol{x}}_j$，成为 正样本对，本文包含三种增强：随机裁剪、随机 color distortions、随机高斯模糊
+ 基于神经网络的 encoder $f(\cdot)$，从增强后的样本中提取表征，这个网络可以是任何的架构，本文选的是 ResNet，得到 $\boldsymbol{h}_i=f(\tilde{\boldsymbol{x}}_i)=\text{ResNet}(\widetilde{\boldsymbol{x}}_i)\mathrm{~where~}\boldsymbol{h}_i\in\mathbb{R}^d$。
+ 一个简单的基于神经网络的 projection head $g(\cdot)$，采用一个带有隐藏层的 MLP 得到 $\boldsymbol{z}_i=g(\boldsymbol{h}_i)=W^{(2)}\boldsymbol{\sigma}(W^{(1)}\boldsymbol{h}_i)$，其中 $\sigma$ 为 ReLU 非线性激活。
+ 对比损失函数

随机采样 batch 为 $N$ 的样本，基于增强后的数据定义对比预测任务，得到 $2N$ 个数据样本。没有显式地采样任何负样本，而是给定一个正样本对，剩下的 $2(N-1)$ 个样本都作为负样本。同时定义 $\operatorname{sim}(\boldsymbol{u},\boldsymbol{v})=\boldsymbol{u}^\top\boldsymbol{v}/\|\boldsymbol{u}\|\|\boldsymbol{v}\|$ 为余弦相似度，此时给定正样本对 $(i,j)$，损失函数为：
$$\ell_{i, j}=-\log \frac{\exp \left(\operatorname{sim}\left(\boldsymbol{z}_i, \boldsymbol{z}_j\right) / \tau\right)}{\sum_{k=1}^{2 N} \mathbb{1}_{[k \neq i]} \exp \left(\operatorname{sim}\left(\boldsymbol{z}_i, \boldsymbol{z}_k\right) / \tau\right)}$$
其中，$\mathbb{1}_{[k \neq i]} \in\{0,1\}$ 为 indicator function（当 $k\neq i$ 时为 1，其他情况为 0），$\tau$ 为温度系数。最终的损失在 mini-batch 中的所有正样本对之间计算。

### 大 batch size 下的训练（略）

### 评估方案（略）

## 用于对比表征学习的数据增强（略）