
1. 已知用于区分真实语音与欺骗或深度伪造语音的伪影存在于特定子带和时间段中
2. 本文表明，在模型内部进行融合、从原始波形学习表征 可以获得更好的性能
3. 本文提出 spectro-temporal GAT 网络，学习跨子带和时间间隔的欺诈线索之间的关系


## Introduction

1. 欺骗攻击的线索存在于特定子带或时间段中， [[Subband modeling for spoofing detection in automatic speaker verification 笔记]]、[[Spoofing Attack Detection using the Non-linear Fusion of Sub-band Classifiers 笔记]]、[[An explainability study of the constant Q cepstral coefficient spoofing countermeasure for automatic speaker verification 笔记]]、[[End-to-End Anti-Spoofing with RawNet2 笔记]] 等工作表明，可以使用具有谱或时间注意力的模型来学习
2. [[Graph Attention Networks for Anti-Spoofing 笔记]] 显示了 GAT 可以学习不同子带或不同时间间隔的欺诈线索之间的关系，同时观察到，跨越频域或时域的不同注意力机制或多或少地适用于不同的欺骗攻击，并且可以通过它们在分数级别的融合来利用两者的优势
3. 本文探索了模型本身的融合，将 使用单独的模型和注意力机制对不同子带或不同时间段之间的关系进行建模，扩展到 使用具有组合频谱时间注意力的GAT（GAT-ST）对跨越不同频谱时间间隔的关系进行建模
4. 受 [[End-to-End Anti-Spoofing with RawNet2 笔记]] 启发提出 RawGAT-ST 使用 1D sinc 卷积层提取原始波形，本文贡献为：
	1. 端到端的特征表征学习和GAT模型
	2. 提出新的 spectro-temporal GAT 学习不同子带和时间间隔的线索关系
	3. 提出新的 graph pooling 策略来降低图的维度
	4. 探讨不同模型层次的图融合策略

## GAT 反欺诈

见论文 [[Graph Attention Networks for Anti-Spoofing 笔记]]。

## RawGAT-ST 模型

RawGAT-ST 包含四个策略：
1. 以完全端到端的方式从原始波形中学习高层次的语义特征
2. 提出基于 spectro-temporal attention 的新型的图卷积模块
3. 提出新的 graph pooling 进行节点选择
4. 模型级的融合

RawGAT-ST  结构如图：
![[Pasted image 20221208203244.png]]
### 前端特征表征

RawGAT-ST 使用 sinc 滤波器进行前端特征提取（原理见 [[../经典模型和算法/Speaker Recognition from Raw Waveform with SincNet 笔记]]），但是不学习滤波器的截止频率，而是使用固定的截止频率来避免过拟合（？？？，那不就和传统的特征提取差不多了）。

sinc 层的输出加上一层 channel 维度然后进行转换到二维的时频表征，然后送到二维的 residual 网络中学习高层特征表征 $\mathbf{S} \in \mathbb{R}^{C \times F \times T}$ ，每个 residual 网络都包括 BN层、SeLU 激活、2D 卷积和 max-pooling（用于下采样） 。

### Spectro-temporal 注意力

模型将 spectro-temporal 图注意力 融合到一个模型中。RawGAT-ST 模型包含三个模块，每个都包含一个 GAT 层：
+ spectral attention 模块
+ temporal attention 模块
+ spectro-temporal attention 模块

S-attention 和 T-attention模块主要用于识别时间和频率线索，而 ST-attention 模块 用于建模跨越两个 domain 的关系。

这三个模块都包括 graph pooling 层，更详细的原理如下：![[Pasted image 20221208212427.png]]
GAT 的原理：对于输入图 $\mathcal{G}(N, \mathcal{E}, \mathrm{h})$，通过使用自注意力聚合节点信息得到权重来产生输出图 $\mathcal{G}^{\prime}(N, \mathcal{E}, \mathrm{o})$，并将节点向量的维度从 $d$  降到 $d^\prime$。

将上面说的 GAT 分别用在时间和频率轴中来建模 不同 频率子带 和 时间间隔 的伪影。在输入到 GAT 之间通过 max-pooling 消掉一个维度，对于 spectral attention消掉时间维度，计算为：$$\mathbf{f}=\max _T(\operatorname{abs}(\mathbf{S}))$$
其中，$\mathbf{f} \in \mathbb{R}^{C \times F}$ ，而对于 temporal attention 则消掉频率维度，计算为：$$\mathbf{t}=\max _{\mathrm{F}}(\operatorname{abs}(\mathbf{S}))$$
其中，$\mathbf{t} \in \mathbb{R}^{C \times T}$，因为 $\mathbf{S}$ 有正有负，所以使用了 $abs$ 操作 。

然后分别构造图 $\mathcal{G}_{\mathbf{f}} \in \mathbb{R}^{N_{\mathbf{f}} \times d}$ 和 $\mathcal{G}_{\mathbf{t}} \in \mathbb{R}^{N_{\mathbf{t}} \times d}$ 并在图中使用 GAT 得到输出图 $\mathcal{G}_{\mathbf{f}}^{\prime} \in \mathbb{R}^{N_{\mathbf{f}} \times d^{\prime}}$ 和 $\mathcal{G}_{\mathbf{t}}^{\prime} \in \mathbb{R}^{N_{\mathrm{t}} \times d^{\prime}}$ 。

### Graph pooling

Graph pooling 通过选择图中信息量较大的节点来生成输入图的子图，从而减少节点个数。

方法基于 Graph U-Net 结构，原理如图：![[Pasted image 20221208215753.png]]
使用一个可学习的 projection vector $q \in \mathbb{R}^{d \times 1}$ ，计算每个节点和它的 dot product：$$y_n=X_n \cdot q$$
其中，$X_n \in \mathbb{R}^{1 \times d}$，然后选择 $y$ 的 top k 对应的索引，并计算 element-wise 的乘法：$$\mathbf{X}^{\prime}=\mathbf{X}_{n_{i d x}} \odot \operatorname{sigmoid}\left(y_{n_{i d x}}\right)$$
其他没被选中的节点就丢弃。

### 模型级的融合

用于获取 S-attention 和 T-attention 提取的完整的信息。探索了三种不同的融合策略：$$\mathcal{G}_{\mathbf{f t}}= \begin{cases}\mathcal{G}_{\mathbf{f}_{\text {pooled }}}^{\prime} \oplus & \mathcal{G}^{\prime} \mathbf{t}_{\text {pooled }} \\ \mathcal{G}_{\mathbf{f}_{\text {pooled }}}^{\prime} \odot & \mathcal{G}^{\prime}{ }_{\mathbf{t}_{\text {poled }}} \\ \mathcal{G}_{\mathbf{f}_{\text {pooled }}}^{\prime} \| & \mathcal{G}^{\prime}{ }_{\mathbf{p}_{\text {pooled }}}\end{cases}$$
其中，$\mathcal{G}_{\mathbf{f t}} \in \mathbb{R}^{N_{\mathbf{f t}} \times d_{\mathbf{f t}}}$，下标 pooled 表示经过 graph pooling 之后的图。

简单来说，三种策略分别是 element-wise 的 加法、乘法和 拼接。然后对 $\mathcal{G}_{\mathbf{f t}}$ 再用一层 GAT 得到输出图 $\mathcal{G}^{\prime}{ }_{\mathbf{f t}} \in \mathbb{R}^{N_{\mathrm{ft}} \times d_{\mathbf{f t}}^{\prime}}$，最后进行 Graph pooling 得到 $\mathcal{G}^{\prime} \mathbf{f t}_{\text {pooled }}$，使用 projection 和 output layer 得到二分类结果。

## 实验

数据集：ASV spoof 2019 LA

baseline：RawNet2  [[End-to-End Anti-Spoofing with RawNet2 笔记]] 

模型参数见论文。

## 结果

1. 和 baseline 比：![[Pasted image 20221208221341.png]]结论：效果很好
2. 消融实验：![[Pasted image 20221208221536.png]]没有 S-attention，性能相对下降34%。没有 T-attention（0.0385）的下降不那么严重（13%），表明 S 比 T 更重要。没有池化层，性能的相对下降58%，说明使用 pooling 可以集中于信息最丰富的节点特征。
3. 和其他模型的性能比较：![[Pasted image 20221208221925.png]]结论：优于其他，是性能最好的单系统，其他优点诸如不复杂、完全端到端。






