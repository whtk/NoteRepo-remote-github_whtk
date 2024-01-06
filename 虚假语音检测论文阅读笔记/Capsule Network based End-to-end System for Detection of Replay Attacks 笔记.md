> [ISCSLP](https://ieeexplore.ieee.org/xpl/conhome/9362048/proceeding) 2021 的文章
> 

1. capsule networks 使用向量来同时记录 spatial information 和 the probability of presence，对检测伪造图像和视频非常有效
2. 本文研究了用于重放攻击检测的 capsule networks，考虑了不同的输入特征
3. 结果表明可以和最先进的单系统相媲美

## Introduction

1. Capsule networks （[[CapsuleNet]]）可 通过同时使用 向量 和 存在概率 捕获 空间信息
2. Capsule networks 已被用于各种语音处理，也被用于检测伪造的图像和视频，其识别此类攻击方面优于其他最先进的系统
3. 本文认为，capsule networks 捕获的空间信息可能有助于以与伪造图像和视频类似的方式描述用于检测重放语音的伪影

## 用于 重放检测的 Capsule Network

### Capsule Network 原理（略）

### 架构

capsule networks 的输入来自卷积层，输出通过 dense layer 处理后作为最终的分类层。本文研究了两种 capsule network。如图：![[Pasted image 20221227214402.png]]
1. CapsNet，没额外的 FC 层
2. CapsNetFC，有 FC 层

其中的 capsule layers 细节如下：![[Pasted image 20221227214629.png]]
输入为 STFT特征，维度为 $120\times 1025$，卷积之后维度为 $15\times 129 \times 256$，然后 transform 到  $16\times 8320$，即 8320 个 capsule，每个向量的维度为 16，通过 capsule network 之后，capsule 的数量变成 2 个，每个维度为 32，且每个 capsule 的长度作为最终的 score。

如果加上 FC 层则形成另一个模型，两个 capsule 输入 到两个连续的 FC 层（不是一对一的），每层 dropout 0.5，进行 ReLU 激活后通过 softmax 得到最终的结果。

损失函数为 capsule network 中的：$$L_c=T_c \max \left(0, m^{+}-\left\|v_c\right\|\right)^2+\lambda\left(1-T_c\right) \max \left(0,\left\|v_c\right\|-m^{-}\right)^2$$

## 实验 & 结果

数据集：ASVspoof 2019 PA

实验设置见论文。

不同输入特征：![[Pasted image 20221227215615.png]]
基于 STFT 的效果更好；CapsNetFC优于基于CapsNet的系统

和 baseline 比：![[Pasted image 20221227215658.png]]
比 baseline 要好很多。

实验也表明，重放设备的质量对识别重放攻击的影响最大；与CQCC baseline 相比，基于CapsNetFC的系统对重放设备的质量以及距离范围不太敏感。