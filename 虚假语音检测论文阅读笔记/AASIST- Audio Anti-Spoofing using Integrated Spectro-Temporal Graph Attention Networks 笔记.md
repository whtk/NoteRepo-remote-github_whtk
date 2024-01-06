> ICASSP 2022

1. 本文提出了一种新的异构 stacking graph attention layer，该层利用异构注意力机制和 stack node 来建模异构的跨时域和频域的伪影
2. 通过新的 max graph 操作，提出的方法可以超过 SOAT 20%，即使是 light 版本的都超过其他系统。

## Introduction

1. 本文重点是 LA
2. 欺骗伪影存在于时频域中，自适应机制具有集中于伪影所在区域的灵活性，对可靠检测至关重要
3. 作者最近的工作 [[Graph Attention Networks for Anti-Spoofing 笔记]]、[[End-to-End Spectro-Temporal Graph Attention Networks for Speaker Verification Anti-Spoofing and Speech Deepfake Detection 笔记]] 使用 rawnet2 like 的encoder 和 图注意力网络，但是仍然有改进的空间，可以使用异构技术将其时域和频域网络进行集成
4. 本文对之前的 [[End-to-End Spectro-Temporal Graph Attention Networks for Speaker Verification Anti-Spoofing and Speech Deepfake Detection 笔记]] RawGAT-ST 模型进行拓展，提出的 AASIST 
	1. 提出 heterogeneous stacking graph attention layer
	2. 提出 max graph operation 机制，用来模拟 最大特征图，使用不同分支学习不同的伪影
	3. 提出一种使用 stack node 的新的 readout scheme


## 预备知识

### RawNet2-based encoder

使用了 [[End-to-End Anti-Spoofing with RawNet2 笔记]] 中的 RawNet2 结构，具体原理见笔记。

本文用的 RawNet2 和 图模块 与 [[End-to-End Spectro-Temporal Graph Attention Networks for Speaker Verification Anti-Spoofing and Speech Deepfake Detection 笔记]] 一样。

同时也使用了 Graph pooling 的操作。

## AASIST

之前的方法将两个模型（频域图和时域图）进行融合。提出的方法则旨在实现进一步高效的融合。

AASIST 框架如图：![[Pasted image 20221212172611.png]]


### Graph combination

 将频域和时域图组合成异质图 $\mathcal{G}_{s t}$ ，设 $\mathcal{G}_t \in \mathbb{R}^{N_t \times D_t},\mathcal{G}_s \in \mathbb{R}^{N_s \times D_s}$ 分别为时域和频域图，其计算为：$$\begin{aligned}
& \mathcal{G}_t=\text { graph\_module }\left(\max _s(\operatorname{abs}(F))\right) \\
& \mathcal{G}_s=\text { graph\_module }\left(\max _t(\operatorname{abs}(F))\right)
\end{aligned}$$其中，$F \in \mathbb{R}^{C \times S \times T}$ 为 encoder 输出的 feature map，graph module 表示 graph attention and graph pooling layers。组合图 $\mathcal{G}_{s t}$ 有 $N_t+N_s$ 个节点，并且在两个子图的节点之间互相添加一条边（上图中的虚线），从而可以估计异质节点之间的时空关系。

尽管如此，$\mathcal{G}_{s t}$ 本质还是异质图，其子图的节点特征还是处于不同的 latent space 中的。

### HS-GAL

本文的贡献在于 heterogeneous stacking graph attention layer，包含
+ heterogeneous attention
+ stack node

HS-GAL 的输入首先 project 到一个相同的特征维度 $D_{st}$ 。

#### Heterogeneous attention

使用三个不同的 projection vector 来计算异质图的 attention weight：
+ $\mathcal{G}_{t}$ 到 $\mathcal{G}_{t}$，图中蓝线
+ $\mathcal{G}_{s}$ 到 $\mathcal{G}_{s}$，图中橙线
+ $\mathcal{G}_{t}$ 到 $\mathcal{G}_{s}$（以及反过来），图中虚线

#### Stack node

引入一种新的 Stack node，用来累计异质图信息（或者说时频域之间的信息和关系），stack node 同时连接两个子图的 主干，而且是单项边。

第一层的 stack node 输出用于初始化下一层。

其作用有点类似于 classification token，只不过是单向的。

#### Max graph operation 和 readout

MGO 如图中灰色部分所示，其本质就是 element-wise 的 maximization。

MGO 使用两个并行的分支，每个分支都是用了 element-wise maximum。

特别的，每个分支都包含两个顺序的 HS-GAL 层（输出使用了 graph pooling）。即 MGO 包含四个 HS-GAL 和 四个 pooling 层。同一个分支共享 stack node，前一层的 HS-GAL 中的 stack node 会传递到第二层。最终每个 HS-GAL都得到一个 maximum 的结果。

read scheme 是图中最右边的灰色框。采用 node-wise maximum and average 得到四个 node，然后再用一个 stack node 把他们 拼接起来做二分类。

还提出了 light 版本的 AASIST-L。


## 实验和结果

数据集：ASVspoof 2019 LA

指标：EER、min t-DCF

不同的 seed 导致的结果相差很大，因此使用不同 seed 的平均值作为结果。

结果：
1. 和 sota 比![[Pasted image 20221212222442.png]]
2. 和 作者之前的模型（baseline）比：![[Pasted image 20221212222613.png]]只能说各有千秋吧。
3. 消融实验：![[Pasted image 20221212222653.png]]表明提出的三个技巧都很有效，且 heterogeneous attention 的影响最大。

