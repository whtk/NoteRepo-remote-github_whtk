> interspeech 2019

1. 本文提出的系统将 x-vector attack embeddings 和 信号处理特征相结合，其中使用了 TDNN 网络从 MFCC 中提取特征
2. embeddings 联合建模了27种不同的环境和9种类型的攻击，同时使用了 SCMC 作为特征
3. 训练过程中加入了加性高斯噪声层来增强未知攻击的鲁棒性

## Introduction

1. [[Attentive Filtering Networks for Audio Replay Attack Detection 笔记]] 表明，重放线索可以在时域和频域中找到
2. 声音信号中的高频子带包含重放线索，如 IMFCC （[[Speech Replay Detection with x-Vector Attack Embeddings and Spectral Features 笔记]]）
3. 有工作表明，SCMC 是最好的和最一致的 特征，CQCC 也很有前途
4. 本文的贡献有：
	1. 引入新的 x-vector attack embeddings
	2. 分析 x-vector attack embeddings 在不同记录条件下的变化
	3. 展示了 x-vector embeddings 和 信号特征的组合优于所有的 baseline

## 特征

### 语音信号特征

本文提取了以下特征：
+ MFCC
+ IMFCC
+ RFCC
+ LFCC
+ SCMC
+ CQCC
且只是用了静态特征（没有用delta）。

使用IDIAP Bob.ap信号处理库提取这些特征，最终的特征是 $N\times M$ 维的矩阵，$M$ 为帧数，$N$ 为特征维度。使用下采样固定帧长为 10。

特征进行了 $[-1,1]$ 的归一化。

### X-vector Embedding 生成

目标是提取固定长度的，utterance-level 的向量，可以表征环境和攻击类型。

使用了 kaldi 进行提取 x-vector，这个 vector 可以表征 环境和攻击类型。其输入是 40 维的 MFCC。采用的是原论文中的 TDNN 网络，x-vector 是从第六层提取的，**唯一的区别是，不是训练进行说话人分类而是进行分类声学环境和攻击类型**。

> 包括 10 个攻击类型（9假1真）和 27 个声学环境，总共 270 类。

最后分类的准确率是 85%，说明这样改还不错。

然后使用 LDA 将 x-vector 从 512 降低到 10 维。

### X-vector Embedding 分析

相比于联合分类，单独分类的效果更好。环境分类比攻击分类更容易（原因可能是数据分布不平衡）。

分析 confusion matrix 发现，重放设备质量分类效果不错，但是 攻击到讲话者的距离 的分类不好。而对于环境分类，说话人与ASV的距离和房间大小似乎没有很好地捕捉，混响时间则很好地区分了类别。

其实最好的就是有一种 embedding 对各种条件的效果都差不多，这样泛化性最好。

作者还是不相信 x-vector 的能力，于是将信号处理的特征作为补充特征。

### 特征组合

将x-vector 和信号处理特征拼接之前，对 x-vecotr 使用 0.1 的因子进行了缩放（经验发现效果很好）。

最终提交的系统基于 SCMC+前面的 x-vector，包括 $40\times 10+10=410$ 维。

## 系统架构

采用 keras 实现，系统架构如图：![[Pasted image 20221231205755.png]]
将分类标签转换为数值，本质是一个回归问题，-1 为假，1 为真，所以用 tanh 激活函数。

系统的第一层是 加性高斯噪声层，可以看成是数据增强的一种方式。

训练时使用 adam 优化器，具体的参数见论文。

## 结果

![[Pasted image 20221231212753.png]]

1. 虽然单独的x-vector不能很好地区分欺骗和真实语音，但与信号特征相结合时有一定的改进。（其实单独 SCMC或者单独 IMFCC 效果也挺好）
2. 使用高斯噪声层可以提高性能（？？？不对啊，表中的效果不是更差了吗）
