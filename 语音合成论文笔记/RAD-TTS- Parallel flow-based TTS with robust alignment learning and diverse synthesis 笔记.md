> ICML workshop 2021，NVIDIA

1. 提出一种基于 normalizing flow 的并行、端到端 TTS 模型
	1. 用一个分布来额外建模语言韵律分布，从而实现可变的 duration
2. 提出了一种用于在线提取对齐的框架
3. 实现表明， 相比于 baseline，有更好的对齐和更多样化的输出

## Introduction

1. 提出 RAD-TTS，可以实现鲁棒的对齐和多样地合成
2. 提出一种稳定的、无监督的对齐学习框架
3. 和 [Glow-TTS- A Generative Flow for Text-to-Speech via Monotonic Alignment Search 笔记](Glow-TTS-%20A%20Generative%20Flow%20for%20Text-to-Speech%20via%20Monotonic%20Alignment%20Search%20笔记.md) 类似，也是在 normalizing flow 框架下提出一种对齐机制，但是本文的对齐可以用于任意的 TTS
4. 提出采用一个独立的生成模型用于 duration 的生成

## 方法

考虑以 mel 谱 表征的 音频 $X\in\mathbb{R}^{C_{mel}\times T}$，$T$ 为 mel 谱帧数，$\Phi\in\mathbb{R}^{C_{txt}\times N}$ 为 长 $N$ 的 text embedding，$\mathcal{A}\in\mathbb{R}^{N\times T}$ 为对齐矩阵。$\xi$ 为说话人特征向量。建模条件分布：
$$P(X,\mathcal{A}|\Phi,\xi)=P_{mel}(X|\Phi,\xi,\mathcal{A})P_{dur}(\mathcal{A}|\Phi,\xi)$$
从而可以在推理时同时从 mel 谱 帧 和 duration 中采样，框架如图：
![](image/Pasted%20image%2020231201171946.png)

### Normalizing Flows

### mel decoder

mel-decoder 定义为 $g$，建模分布 $P_{mel}()$，在每个 time step 从 latent $z$ 中采样，然后映射到 mel 帧 $x$，这个过程需要基于 文本、说话人特征 和 对齐：
$$\begin{aligned}X&=g(Z;\Phi,\xi,\mathcal{A})=g_1\circ g_2\ldots g_{K-1}\circ g_K(Z;\Phi,\xi,\mathcal{A})
\\
Z&=g_K^{-1}\circ g_{K-1}^{-1}\ldots g_2^{-1}\circ g_1^{-1}(X;\Phi,\xi,\mathcal{A}) \end{aligned}$$
其中对于每个 $z\in Z,z\sim\mathcal{N}(0,\mathbf{I})$。

用的是 Glow-base bipartite-flow 架构。
> 1x1 的可逆卷积+affine coupling layer

#### 无监督对齐学习

将 HMM 中的 维特比算法 和 前向后向算法 组合起来，用于学习 hard 和 soft alignment。令 $\mathcal{A}_{soft}\in\mathbb{R}^{N\times T}$ 为对齐矩阵。目标是提取单调的、二值化的对齐矩阵 $\mathcal{A}_{hard}$，使得对于每个帧，概率质量都以单个 符号 为中心，从而 $\sum_{j=1}^T\mathcal{A}_{hard,i,j}$ 可以得到每个符号的 duration。

和 Glow-TTS 一样，soft alignment 的分布基于文本 token 和 mel 帧 之间的相似度，然后在用 softmax 在 文本 维度 进行归一化：
$$\begin{aligned}
&D_{i,j}=dist_{L2}(\phi_i^{\boldsymbol{enc}},x_j^{\boldsymbol{enc}}), \\
&\begin{aligned}\mathcal{A}_{soft}&=\text{softmax}(-D,\text{dim}=0).\end{aligned}
\end{aligned}$$
这里的 $x^{enc}$ 和 $\phi^{enc}$ 为编码后的 variants。用的是 2-3 层的 1D CNN，这种简单的变换可以产生更好的效果。

前向后向算法给定观测值来最大化隐藏层的似然，文本定义为隐藏层状态，mel 谱定义为观测值，则该算法最大化 $P(S(\Phi)|X)$。考虑所有的单调对齐，其对应的序列为 $\textbf{s}:\:\{s_1=\phi_1,s_2=\phi_2,\ldots s_T=\phi_N\}$，其必须满足：
+ 起点和终点分别为第一个和最后一个 text token
+ 每个 text token $\phi_n$ 至少要使用一次
+ 对应的要么是 0 要么是 1

所有有效的对齐的似然为：
$$P(S(\Phi)\mid X)=\sum_{\mathbf{s}\in S(\Phi)}\prod_{t=1}^TP(s_t\mid x_t)$$
从而可以使用 pytorch 的 CTC loss 来实现。

然后采用 cigar-shaped diagonal prior 来加速对齐。用 Beta-Binomial distribution 来促使元素在对角线上线性分布。

采用维特比算法来选择最可能的那条单调对齐路径，可以得到 $\mathcal{A}_{hard}$，但是由于这个算法不可微，于是强迫 $\mathcal{A}_{soft}$ 来尽可能地接近 $\mathcal{A}_{hard}$，这个过程通过最小化他们之间的 KL 散度实现：
$$\mathcal{L}_{bin}=\mathcal{A}_{hard}\odot\log\mathcal{A}_{soft}$$

###  Generative Duration Modeling

之前的 TTS 采用确定的回归模型来预测 duration，从而限制了其多样性。于是这里采用一个独立的 normalizing flow 来单独建模 $P_{dur}()$。

至于这个模型，可以用另一个 bi-partite flow 来实现，也可用 autoregressive flow 来实现。

### Training Schedule

训练的时候，从下面的 loss 开始：
$$\mathcal{L}=\mathcal{L}_{mel}+\lambda_{1}\mathcal{L}_{align}$$

然后随着训练的 step 变化，改变 loss 的结构：
+ 0-6k：用 $\mathcal{A}_{soft}$ 作为对齐矩阵
+ 6-18k：使用维特比 $\mathcal{A}_{hard}$ 替换前面的对齐矩阵
+ 18k-最后：添加 $\lambda_2\mathcal{L}_{bin}$ 项到损失函数中

## 实验（略）

