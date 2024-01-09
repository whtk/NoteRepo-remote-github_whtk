> CVPR 2023，港中文，微软

1. ViT 计算消耗很高，不适用于实时
2. 提出一组高速的 ViT，称为 EfﬁcientViT，现有 transformer 的速度主要来自于内存低效的操作，包括 MHSA 中的 reshape 和 element-wise 计算
	1. 设计 sandwich layout 结构，在  FFN 之间用单个 memory-bound MHSA 来提高效率
	2. 不同 attention head 之间的相似度很高，提出 cascaded group attention 模块
3. 性能超过现有的 efﬁcient model

## Introduction

1. 现有的多数 light 和 efficient ViT 都关注减参或减少 Flops，但是并不能影响实际的推理吞吐量
2. 分析了三个影响模型推理速度的因子，发现主要原因在于第一点：
	1. 内存读取
	2. 计算冗余
	3. 参数使用
3. 大部分的 memory-inefﬁcient 是频繁地 tensor reshape 和 element-wise 计算
4. 基于前面的，提出一种新的 memory efﬁcient transformer，称为 EfﬁcientViT

## ViT 效率分析

从 内存读取、计算冗余 和 参数使用 三个方面分析。

### 内存效率

transformer 中许多内存读取都是低效的，如图：
![](image/Pasted%20image%2020240109102231.png)

通过减少 memory-inefﬁcient 操作来节约内存读取花销。具体来说，将 Swin-T 和 DeiT-T 中设置不同的的 MHSA layers 占比，然后分析性能，发现 20%-40% 的 MHSA layers 的效果最好，而通常的 ViT 中的 MHSA 有 50%。同时发现，在 20% 的 MHSA 中，memory-bound 操作减少到原来时间的 44.26%。

这说明，减少 MHSA layer 占比可以显著增强内存效率和提高性能。

### 计算效率

MHSA 将输入分为多个 head 然后分别计算 attention map，这个过程非常耗时，且有研究表明，大部分都不那么重要。于是可以减少一冗余的 attention。

于是提出，对每个 head 只输入 feature 的一部分，想法类似于 group attention。实验表明，在不同的 head 中采用 不同的 channel-wise splits of the feature，相比于采用 full feature 可以减少计算冗余。

### 参数效率

ViT 通常继承 NLP 中的 Transformer 的配置，如采用 等宽 QKV 矩阵，随着 stage 增加 head 数，将 FFN 中的 expansion ratio 设为 4 等，但是 lightweight 则需要好好设计。

本文采用 Taylor structured pruning 来自动查找 Swin-T 和 DeiT-T 中的重要组分。此剪枝方法可以移除不重要的 channel，保留最重要的 channel。

实验观测到：
+ 前两个 stage 保留更多的维度，最后一个 stage 更少的维度
+ QK 和 FFN 的维度大幅修剪，V 的维度几乎被保留

这些现象表明：
+ 之前的那种每个 stage 都 double channel 数量（或保持不变）的配置存在大量的冗余
+ QK 的冗余比 V 更大

## EfficientViT

EfﬁcientViT 架构如下：
![](image/Pasted%20image%2020240109145532.png)

### EfﬁcientViT Building Blocks

EfﬁcientViT Building Blocks 如图 b，包含：
+ memory-efﬁcient sandwich layout
+ cascaded group attention 模块
+ parameter reallocation 策略

sandwich layout 采用更少的 self-attention 层，更多的 memory-efficient FFN 层，即用多个 FFN 层 $\Phi_{i}^{\mathrm{F}}$ 夹住单个 self-attention 层 $\Phi_{i}^{\mathrm{A}}$，此时 transformer 计算如下：
$$X_{i+1}=\prod^{\mathcal{N}}\Phi_i^{\mathcal{F}}(\Phi_i^{\mathcal{A}}(\prod^{\mathcal{N}}\Phi_i^{\mathcal{F}}(X_i)))$$
其中 $X_i$ 为第 $i$ 个 block 的输入特征，前后各有 $\mathcal{N}$ 层 FFN。在每个 FFN 之前添加了一个额外的 token interaction layer，引入 局部信息 的  inductive bias 来提高模型建模能力。

Cascaded Group Attention 对每个 head 输入不同的 splits of the full features，从而减少 attention 的计算：
$$\begin{aligned}
\tilde{X}_{ij}& =Attn(X_{ij}W_{ij}^\mathrm{Q},X_{ij}W_{ij}^\mathrm{K},X_{ij}W_{ij}^\mathrm{V}) \\
\tilde{X}_{i+1}& =Concat[\widetilde{X}_{ij}]_{j=1:h}W_{i}^{\mathrm{P}}
\end{aligned}$$
也就是，第 $j$ 个 head 计算 attention 的时候输入是 $X_i$ 的第 $j$ 个 split，即 $[X_{i1},X_{i2},\ldots,X_{ih}]\mathrm{~and~}1\leq j\leq h$。

然后进一步以级连方式计算 attention map，如图 c，将每个 head 的输出加入到下一个 head 中来逐步 refine 特征：
$$X_{ij}^{'}=X_{ij}+\widetilde{X}_{i(j-1)},\quad1<j\leq h$$
从而 self attention 可以联合捕获 local 和 global relation。

这种级连结构有两个好处：
+ 可以提高 attention map 的多样性，也可以减少参数
+ 允许增加网络深度，在不引入额外参数的情况下增加建模能力

Parameter Reallocation：通过将一些关键模块的 channel width 拓展，而不重要的 channel 的 width 进行压缩。具体而言，将 QK 的投影矩阵的 channel 减小，对于 V 则可以和输入维度一致。同时也将 FFN 中的 expansion ratio 从 4 降到 2。

###  EfﬁcientViT 网络架构

整体架构如图 a，引入 overlapping patch embedding 将 16x16 的 patch 编码为 $C_1$ 维度的 token，整个架构包含三个 stage，每个 stage 包含几层的 EfﬁcientViT Building Blocks，然后接 subsampling layer 将 token 数减少为 1/4。

使用的 subsample block 也有 sandwich layout，除了将 self-attenion 替换为 inverted residual block 来减少上采样过程中的信息损失。

## 实验（略）