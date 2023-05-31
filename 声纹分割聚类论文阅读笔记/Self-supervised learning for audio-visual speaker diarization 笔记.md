
1. 提出一种 自监督音频-视频 同步学习的方法
2. 引入两个新的损失函数：the dynamic triplet loss 和 multinomial loss
3. 介绍了一个新的大规模中文音视频语料库

## Introduction

许多研究侧重于纯视频或纯音频方法：
+ Everingham 等人使用嘴唇运动（即，仅视频）来定义活动说话者
+ 大多则仅使用音频，如 [[Speaker Diarization with LSTM 笔记]] 和 [[Speaker diarization using deep neural network embeddings 笔记]] 等
+ 也有使用多模态的：[[Putting a Face to the Voice- Fusing Audio and Visual Signals Across a Video to Determine Speakers 笔记]]、[[Audio-Visual Speaker Diarization Based on Spatiotemporal Bayesian Fusion 笔记]]
+ 也有论文使用了 无监督来进行 SD（2013和2014的论文）

有方法使用 课程学习和负样本选择策略，其中使用了对比损失。但是对比损失的缺点是将所有未同步的对都被视为负样本对，导致正样本和负样本之间的严重失衡。同时，将一些稍微不同步的音频-视频对与大量移位的对或异源音频/视频对同等对待会损害模型性能。有人尝试分类（即交叉熵）损失来解决这个问题，但是无法收敛。

在本文中，我们提出了三种用于训练的音频-视频对：
+ synchronized pairs
+ shifted pairs（其中音频和视频移位了 $j$ 个视频帧）
+ heterologous pairs（其中，音频和视频属于不同的源）

然后提出了两个新的损失函数：
+ dynamic triplet loss：正对和负对在每次迭代中都是动态确定的
+ multinomial loss：同时考虑所有的 shifting combinations 和 heterologous combinations


## 方法

> 使用 two stream 分别提取音频和视频特征

## Two-stream 网络结构

分别处理音视频：
+ 对于音频，将输入转换为 MFCC，然后送到 2D 卷积以产生语音特征；
+ 对于视频，采用3D卷积模块来提取连续视频帧之间的时间信息和每个视频帧中的空间信息
![](./image/Pasted%20image%2020221125100017.png)$f_a,f_v$ 分别代表音频和视频特征。

### 采样策略
![](./image/Pasted%20image%2020221125100623.png)
视觉段 $V_n,n\in {1,2,\dots,N}$，其中 $N$ 为总段数：
+ 同步音频段记为 $A_n^S$
+ $j$ 偏移的音频段记为 $A_n^j, \text { where } j \in\{-T, \ldots-1,1 \ldots T-1, T\}$，$T$ 为偏移范围
+ heterologous（异源）音频段为 $A_n^H$
所有的音视频段长度都固定相同。

### Dynamic triplet loss

在对比损失中，非同步音频-视频对之间的距离比同步音频视频对更大。损失如下：$$\begin{array}{r}
L_{c o n}=\frac{1}{2 N} \sum_{n=1}^N\left(y_n\right) d_n^2+\left(1-y_n\right) \max \left(\alpha-d_n, 0\right)^2 \\
d_n=\left\|f_v\left(V_n\right)-f_a\left(A_n\right)\right\|_2
\end{array}$$
其中，$y\in[0,1]$ 为二元相似性测度，表明音视频是否同步（$y=1$ 表示同步）。

对比损失的一个问题是，它平等地对待所有非同步对。因为通过偏移音频/视频段或替换来自另一视频的音频/视频剪辑来对负对进行采样。正负对的数量之间存在严重的不平衡。换句话说，负样本的移位和异源对比正样本的同步对多得多。

此外，模型可能被 easy negatives 而非 hard negatives 所主导。

提出 dynamic triplet loss ，在训练时，正对和负对根据它们的相对距离动态定义：![](./image/Pasted%20image%2020221125102126.png)也就是不明确区分正负样本对，而是根据两者的难易程度进行相对区分。此时，损失为：$$\begin{aligned}
L_{D_{-t r i}}=& \sum_{n=1}^N\left[\left\|f_v\left(V_n\right)-f_a\left(A_n^{\prime}\right)\right\|_2^2\right.\\
&\left.-\left\|f_v\left(V_n\right)-f_a\left(A_n^{\prime \prime}\right)\right\|_2^2+\alpha\right]_{+}
\end{aligned}$$
其中，$\left(V, A^{\prime}\right)$ 为正对，$\left(V, A^{\prime \prime}\right)$ 为负对，$\alpha$ 为 margin。

### Multinomial loss

在 dynamic triplet loss 中，每次迭代中仅对两对进行采样。然而，对于每个音视频对，都有 $\mathrm{C}_{2 T}^2$ 移位可能和更多异源的可能。为了更好利用这些组合，采用 multinomial loss 按簇优化正对和负对，而每个簇都有自己的 margin。这样，具有不同特性（例如移位范围）的负对将被不同地对待。损失计算为：$$\begin{aligned}
L_{m u l} &=\sum_{n=1}^N\left(D\left(f_v\left(V_n\right), f_a\left(A_n^S\right)\right)\right.\\
&+\log \left(\sum_{j^{\prime}}^{0<\left|j^{\prime}\right| \leq m_1} \exp \left(\alpha_1-D\left(f_v\left(V_n\right), f_a\left(A_n^{j \prime}\right)\right)\right)\right.\\
&+\log \left(\sum_{j^{\prime \prime}}^{m_1<\left|j^{\prime \prime}\right| \leq m_2} \exp \left(\alpha_2-D\left(f_v\left(V_n\right), f_a\left(A_n^{j \prime \prime}\right)\right)\right)\right.\\
&+\log \left(\sum_H^{H \neq n} \exp \left(\alpha_3-D\left(f_v\left(V_n\right), f_a\left(A_n^H\right)\right)\right)\right)
\end{aligned}$$第一个是最小化同步对距离，第二部分表示在 $m_1$ 范围内时移位对的损失，第三个表示移位在 $m_1$ 到 $m_2$ 之间的（也就是来自于同一个音视频源，但是时间上完全不重叠），最后一个来源于不同视频和音频段。

三种损失的对比：![](./image/Pasted%20image%2020221125104917.png)

## 实验

数据集：youtube 中的

对比模型：SyncNet 和 UIS-RNN

结果：![](./image/Pasted%20image%2020221125104734.png)
使用 DER 和 F1 得分进行评估。

![](./image/Pasted%20image%2020221125105102.png)提出的模型的区分性更强。