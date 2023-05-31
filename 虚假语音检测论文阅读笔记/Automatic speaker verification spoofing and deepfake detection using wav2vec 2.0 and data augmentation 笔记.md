>The Speaker and Language Recognition Workshop 2022

1. 使用自监督 wav2vec 2.0 作为前端然后进行微调
2. 表征仅基于真实语音进行训练，当进行数据增强时，相比于 baseline 提高了 90%

## Introduction

1. 现有的 ASV 反欺诈系统泛化性不行
2. 开放使用外部数据可能会提高性能，但是合成算法的种类太多了，根本不可能代表所有的攻击算法
3. 本文使用自监督学习提高泛化能力，贡献包括：
	1. 使用预训练的、带有微调的自监督语音模型来改进泛化和域鲁棒性
	2. 使用额外的数据增强补充自监督学习
	3. 提出新的  self-attention based aggregation layer 

## 相关工作

1. [[Investigating self-supervised front ends for speech spoofing countermeasures 笔记]] 比较了不同的基于SSL的前端和后端架构，并表明了微调SSL模型对欺骗检测的重要性，且发现，与HuBERT模型相比，wav2vec 2.0前端提供了更好的通用欺骗检测性能
2. 本文探索了 wav2vec 2.0 XLS-R 模型，使用 AASIST 作为后端

## AASIST baseline 系统

原理见论文 [[AASIST- Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks 笔记]]。

AASIST 直接从原始波形提取表征。

## 自监督前端

使用 wav2vec 2.0模型替换sinc层前端，如图 b：![[Pasted image 20221229222408.png]]

wav2vec 2.0 原理见论文 [[../语音自监督模型论文阅读笔记/wav2vec 2.0- A Framework for Self-Supervised Learning of Speech Representations 笔记]]。

### 预训练

预训练和 fine-tune 过程如图：![[Pasted image 20221229222611.png]]

潜在表征 $z_{1: N}$ 量化成 $q_{1: N}$，把 $z_{1: N}$ mask 掉一部分然后送到 transformer 中从而生成上下文表征 $c_{1: N}$，然后计算对比损失。

本文所有的工作都基于 wav2vec 2.0 XLS-R 模型，采用 Fairseq project toolkit 来提取模型中的特征。

### fine-tune

预训练仅仅使用真实数据，而根据之前的论文，使用域内真实数据和欺骗数据进行微调可以提高检测性能。

作者认为 fine-tune 可以避免过拟合。

实际中，使用了 ASVspouf 2019 LA train 进行 fine-tune，训练时 wav2vec 2.0 XLS-R 和 AASIST 联合优化。

fine-tune 的时候不会进行 input mask，同时添加了一个全连接层来减少表征的维度。

## 基于自注意力的聚合层

Attention based pooling layers 如  self-attentive pooling 或 attentive statistical pooling 已证明有利于帧级特征的聚合和说话人识别和验证任务嵌入的提取。

作者还发现，在前端和后端之间引入基于2D自注意力的聚合层有助于提高欺骗检测性能。

提出的层用于提取更相关的时频表征，使用 conv2d 来生成二维注意力图（注意力权重矩阵），公式表述如下：$$W=\operatorname{Softmax}(\operatorname{conv2d}(\operatorname{BN}(\operatorname{SeLU}(\operatorname{conv2d}(\mathbf{S})))))$$
时域表征计算为：$$\mathbf{t}=\sum_F \mathbf{S} \odot W$$
频域表征计算为：$$\mathbf{f}=\sum_T \mathbf{S} \odot W$$
其中，$\odot$ 表示 element-wise 乘法。

## 实验

### 数据增强

传统的 DA 使用额外的、人工生成的方法来扩增数据集，本文则使用了 RawBoost 工具，在已有数据中实时添加 nuisance variability，包括以下三个方面：
+ 线性和非线性卷积噪声
+ 脉冲信号相关的附加噪声
+ 静止信号相关附加噪声

具体见论文 [[Rawboost- A Raw Data Boosting and Augmentation Method Applied to Automatic Speaker Verification Anti-Spoofing 笔记]]

### 实现细节详见论文

代码即将开源。

## 结果

LA 数据集中：不同前端、自注意力聚合层（SA）比较：![[Pasted image 20221229225659.png]]使用 wav2vec 2.0前端 的EER 最低，使用 SA 进一步降低，使用 DA 大幅降低（且对 wav2vec 2.0前端 的改进最大）。最终的结果是 ASVspoof 2021 LA 最低的 EER。

同理，DF 数据集中：![[Pasted image 20221229230113.png]]
带 * 表示用的是 LA 那一套的 DA，不带表示用的是 专门针对 DF 的 DA。

如果不用 AASIST，效果好像也挺好的：![[Pasted image 20221229230407.png]]