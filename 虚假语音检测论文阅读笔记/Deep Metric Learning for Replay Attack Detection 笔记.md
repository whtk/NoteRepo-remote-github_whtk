
1. 使用 通过度量学习方法 训练 的 SE-residual 网络来学习具有 小的类内距离 和 大的类间距离 的表征
2. 同时研究了包括线性变换和模板匹配在内的后处理方法

## Introduction

1. 深度学习方案对重放攻击的泛化性能较差
2. [[Speech Replay Detection with x-Vector Attack Embeddings and Spectral Features 笔记]] 使用深度神经网络训练 x-vector 提取器学习 embedding，通过分类损失表示攻击类型和声学环境类型
3. 基于度量学习的训练方法直接学习 embedding，这是分类方法的替代方案。已经为深度度量学习开发了各种损失函数，如对比损失、triplet loss
4. 受深度度量学习启发，本文的模型旨在学习类内相似性和类间差异，贡献如下：
	1. 提出了一种受度量学习启发的方法，提取用于重放攻击的表征
	2. 探索了不同的后处理方法来检测重放攻击
	3. 实验了不同 embedding 大小的性能的影响

## 方法

### 训练方法

常规方法使用分类损失（就是 BCE）来训练神经网络，但是这个损失的目标不是优化 embedding 的相似度，所以训练的模型的区分性不强。

度量学习损失直接学习 embedding。

对于虚假语音，考虑了不同的虚假类别（9类），也就是一共 10 类。

#### 基于分类的目标函数

选择 $N_s$ 个语音来构建一个 batch，embedding 为 $x_i$，label 为 $y_i$，$1\le y \le 10$ ，$W_j,b_j$
为最后一层网络的权重，损失函数定义为：$$L=-\frac{1}{N_S} \sum_{i=1}^{N_S} \log \frac{e^{W_{y_i} x_i+b_{y_i}}}{\sum_{j=1}^{10} e^{W_j x_i+b_j}}$$这个损失仅仅用于惩罚分类错误，但是不关注 类内相似性和类间差异。

#### 基于度量学习的目标函数

使用 generalized end-to-end 损失来训练网络，使用 $N_C \times N_S$ 个语音作为一个 batch，$N_C$ 小于类的数量，包含 $1\times N_S$ 条真实语音和 $(N_C-1)\times N_S$ 条虚假语音，每条语音的输出都是 embedding，$E_{i, j}\left(1 \leq i \leq N_C, 1 \leq j \leq N_S\right)$，所有第 $k_{th}$ 类的 $N_S$ 条语音的 embedding 的中心为 $C_k$，计算为：$$C_k=\frac{1}{N_S} \sum_{j=1}^{N_S} E_{k, j}$$
相似矩阵是所有的 embedding 和 所有的中心 $C_k$ 的余弦值：$$S_{i j, k}=\cos \left(E_{i, j}, C_k\right)$$
训练时，目的是最小化相同类之间的相似度，最大化不同类之间的相似度，使用交叉熵损失来计算 相似度矩阵的 softmax 和 target label 之间的损失：$$L=-\frac{1}{N_C \times N_S} \sum_{i=1}^{N_C} \sum_{j=1}^{N_S} \log \frac{e^{s_{i j, i}}}{\sum_{k=1}^{N_C} e^{S_{i j, k}}}$$

### 后处理方法

由于实际的攻击范围比训练集要多，通过增加类的数量来更好的学习表征（类越多，表现越好）。但是任然存在一些难以区分的类，于是提出了两个后处理方法。

### 模板匹配

在完成网络训练后，提取真实语音的 embedding 求平均得到 template embedding，在测试的时候，计算测试样本的 embedding 和 template embedding 的相似度，由于假的语音有很大的多样性，检测是否为真语音更适合实用和实际条件。

### 线性转换

由于很难计算虚假语音 embedding 的中心，于是实现一个额外的线性模型（不是同时训练，是单独训练），在测试的时候，使用线性模型的输出做测试。

> 简单来说，就是将特征维度进行转换使其更加容易区分。（参考神经网络解决异或问题）

## 实验

数据集：2019 的 PA  （train 、dev）上训练，2019 的 dev、evel 和2021的 eval 上测试。

baseline：官方给的四个 baseline

特征：logspec

模型结构：SE-ResNet34（[[ASSERT- Anti-Spoofing with Squeeze-Excitation and Residual neTworks 笔记]]），具体结构如图：![[Pasted image 20221213223406.png]]

## 结果

1. 在 2019 dev 和 eval 中的结果：![[Pasted image 20221213223543.png]]结论：基于 metric 的方法（M-）好于基于 classification 的方法（C-）；大一点的 embedding size 可以提高性能，但是太大了就没用了
2. 最终提交的结果：![[Pasted image 20221213224144.png]]一个字，好！




