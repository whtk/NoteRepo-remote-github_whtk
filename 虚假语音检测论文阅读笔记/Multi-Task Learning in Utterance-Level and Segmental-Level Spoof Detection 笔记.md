
1. 本文提出多任务 benchmark，用于 segmental and utterance levels 的欺诈检测
2. 提出 SELCNN，将 SE 模块插入到 LCNN 中，来增强隐藏特征选择能力
3. 以 SELCNN 和 Bi-LSTM 为基本模型，实现多任务学习框架，相比于单任务性能更好
4. 使用 binary-branch 的 MTL架构，fine tune 一个模型比从零开始训练效果要好

## Introduction

1. 基于 ASV spoof 2019 LA 设计了 PartialSpoof 数据库，一句话可能同时包含真实和虚假的语音段，为数据库构建了 所谓的 utterance-level 和 segmental-level 模型，结果表明，检测语音中的欺骗片段是一项很有挑战的任务（意义何在？？）
2. 同时由于 segmental 和 utterance level 的信息是相关的，所以简单的使用两个独立模型进行检测是不够的，于是考虑采用 多任务学习 MTL，利用多个任务中包含的有用信息来帮助提高所有任务的泛化性
3. 本文分两部分讨论 MTL 结构：
	1. 从结构角度，单分支 or 二分支
	2. 从训练角度，从零开始训练或者fine tune

## 基本模型

LCNN 关键是 MFM，但是 Max-Out 的操作对于 channel selection 过于粗糙，因此引入 SE 模块（[[SENet]]）来辅助学习 channel 的注意力权重，从而形成所谓的 SELCNN，模型的具体结构见论文中的表1。

把 SELCNN 的输出送到 Bi-LSTM 中，使用 residual 结构来稳定训练过程。然后使用average pooling、全连接层 在 utterance level 的模型中，使用 MSE for P2SGrad 损失函数。

## 单任务 or 多任务 学习

### 单任务学习

基于前面的结构使用单任务训练模型。单独训练时，两个level 模型主要的区别是，
+ utterance level 的 pooling 层用于将 segmental feature matrix 转换成 vector
+ 两个模型的 label 不同
两个模型独立训练。

关于两个模型的损失函数和 score 计算如下：![[Pasted image 20221209105547.png]]
具体的模型细节见论文。

### 多任务学习

传统的多任务学习有多个输出分支（每个任务对应一个分支），由于 PartialSpoof 中的utterance 和 segmental 的目标相关（分段标签序列可以唯一地确定话语标签）在推理过程中，两个预测分数可以相互导出，因此可以用单分支也可以用多分支的模型。
![[Pasted image 20221209143629.png]]

#### 单分支模型

上图中，a b 分别代表 utterance 和 segmental level 的单分支模型，每个架构都基于 单任务的结构，但是模型同时产生 score 并同时计算 loss。

但是相同的部分是共享模型和参数的（不共享其实就是单任务结构了）。

#### 二分支模型

上面由于共享（依次计算更新），两个 level 的梯度可能产生冲突，而在二分支结构中，模型并行预测两个独立分支的 score（上图中 c）。

#### 优缺点

单分支架构更加简单明了；可以在不修改基本架构的情况下从中产生损失。但这也加剧了MTL中冲突的梯度问题。

本文则通过为每种类型的标签构建了一个具有独立分支的二分支模型来解决这个问题。

## 训练策略
> 探索了从零开始训练模型和 fine tune warm up 模型两种训练方法

#### 从零训练

两个模型通过共享的层进行联合训练，见图 c。

#### Warm-up

使用一个 level 的预训练模型，设计了两种 warm up 模型，如图 d和e，这时的训练包括两个步骤：
1. single level 的 初始 warm up
2. 联合 fine tune 整个模型
> 这里好像两个模型只会 fine tune 一个

## 实验

数据库：PartialSpoof

特征：LFCC

评估指标：EER

## 结果

1. 和 baseline 比：![[Pasted image 20221209164700.png]]
2. 单任务和多任务比：![[Pasted image 20221209164751.png]]
3. 训练策略比：![[Pasted image 20221209164806.png]]注意、：BW 代表 binary warm-up，BS 代表 binary scratch