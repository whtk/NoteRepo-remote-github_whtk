> X-Vectors: Robust DNN Embeddings for Speaker Recognition 论文
> Deep Neural Network Embeddings for Text-Independent Speaker Verification 论文


# X-Vectors: Robust DNN Embeddings for Speaker Recognition
1. 使用数据增强来改善 DNN embedding 用于说话人识别的性能
2. 将可变长度语音映射到固定大小的向量 x-vector，使用 附加噪声和混响进行数据增强来增加训练数据

## Introduction（略）


## 说话人识别系统
> 包含两个 i-vector baseline 和 DNN x-vector，建立在 kaldi 的语音识别工具包中。

### 声学 i-vector

传统的 i-vector 基于 GMM-UBM，是本文的 baseline。使用 600 维的 i-vector 和 PLDA 计算分数。

### Phonetic bottleneck i-vector

采用 phonetic bottleneck features 特征结合 i-vector，具体原理略。

### v-vector

提取 x-vector 的结构如图：![](./image/Pasted%20image%2020221215113558.png)
假设输入帧数为 $T$，前五层的使用第 $t$ 帧的上下文进行计算，例如 layer frame3 的输入是 frame2 在帧 $t,t-3,t+3$ 处输出的拼接。因此 frame3 的每帧能看到输入的 15 帧。

statistics pooling layer 聚合frame5所有的 $T$ 帧输出，计算其均值和标准差，然后拼接起来送到 segment level layer，最终经过 softmax 得到输出。

DNN 用于训练分类数据集中的 $N$ 个说话人。训练完成后，从 segment6 中提取 embedding 得到所谓的 x-vector。

### PLDA 分类器

PLDA分类器用于 x-vector 和 i-vector 系统。表示居中，并使用LDA进行投影。

LDA维度调整为 i-vector 为200，x-vector 为150。降维后，对表示进行长度归一化，并通过PLDA进行建模。

## 实验（略）



#  Deep Neural Network Embeddings for Text-Independent Speaker Verification

1. 使用 DNN 提取 embedding 来替代说话人验证中的 i-vector。
2. 将可变长度的语音映射到固定维的 speaker embedding，使用基于 PLDA 的后端对embedding进行评分
3. 在短语音中效果优于 i-vector，在长语音中也有一定的竞争力

## Introduction

1. 传统的 i-vector 说话人识别：使用数据训练 UBM ，提取 i-vector 的投影矩阵，然后使用 PLDA 计算 i-vector 之间的相似度得分。
2. 结合 ASR DNN 可以提高 UBM 建模能力，但是代价是增加计算量，且这个方法只对英语有用，多语言环境中没用。
3. 基于 DNN 的ASV 中，有研究使用 DNN 在帧级进行训练对说话人进行分类，然后丢弃 softmax 层，然后通过平均 hidden layer 的输出得到说话人表征，即所谓的 d-vector；
4. 也有研究表明，联合学习 embedding 和相似性度量的端到端系统可以胜过传统的 i-vector 。
5. 本文将端到端的方法分为两个部分：
	1. 生成  embedding 的 DNN（目的是说话人分类）
	2. 比较 embedding 的后端

## DNN embedding 系统

### 概述
提出的系统如图：![](./image/Pasted%20image%2020221216091426.png)
端到端方法需要大量域内数据，本文用多类交叉熵目标代替端到端损失。 此外，单独训练的 PLDA 后端用于比较 pairs of embeddings 。 这使得 DNN 和相似性度量能够在不同的数据集上进行训练。

### 特征

使用 MFCC 特征，应用均值归一化和基于能量的 VAD，通过 TDNN 网络处理 short-term temporal context。

### 网络架构

网络包括：
+ 对语音帧进行操作的层
+ 聚合 frame level 表征的统计池层（statistics pooling layer）
+  segment-level 的附加层
+ softmax 输出层

网络前五层是 frame-level 的，使用了 TDNN 架构。假设 $t$ 为当前的 time step，输入端（第一层）把 $t-2,t-1,t,t+1，t+2$ 这几帧拼起来，后面两层则分别把前一层的输出的 $t-2,t+2$ 和 $t-3,t+3$ 拼起来。剩下两层差不多但是没有任何上下文。

statistics pooling layer 层 的输入为 frame level 最后一层的输出，然后分段进行聚合，计算平均值和标准差，得到所谓的 segment level 的信息。然后将这些向量拼接在一起送到两个 hidden layer 中，最后通过 softmax 进行输出。

使用 x-vector 的时候不包括 softmax。用的是 hidden layer 的输出。

### 训练（略）

### PLDA 后端（略）

 