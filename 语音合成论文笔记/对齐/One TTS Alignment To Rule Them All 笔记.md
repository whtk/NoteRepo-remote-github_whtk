> ICASSP 2022，NVIDIA

1. 自回归的 TTS 使用 attention 机制来学习对齐，但是对于长序列或者 OOD 文本很容易出问题
2. 非自回归的 TTS 则采用从外部源中提取的 duration
3. 本文把 RAD-TTS 中提出的对齐机制看成是一种通用的对齐学习框架，包含：
	1. forward-sum algorithm
	2. Viterbi algorithm
	3. 一个简单但有效的 static prior
4. 提出的框架可以提升所有测试的 TTS 的性能（自回归/非自回归 都可以）

> 主要贡献是在，attention-based 的 alignment，对于并行模型，其思想和 Glow-TTS 极为相似，都是先有一个概率矩阵，即所谓的 soft alignment，然后通过维特比算法（动态规划）得到 hard alignment。区别在于，Glow-TTS 中的概率矩阵是通过直接带入高斯分布求的，而这里的概率矩阵则是通过 attention 来求解（也就是，最后 attention 得到一个 $N\times T$ 维的矩阵（接 softmax），而且是通过 CTC loss 来对这个 soft attention 做反向传播。

## Introduction

1. 靠外部的 aligner 得到的对齐极不稳定，且无法端到端训练
2. 本文采用 [RAD-TTS- Parallel flow-based TTS with robust alignment learning and diverse synthesis 笔记](../RAD-TTS-%20Parallel%20flow-based%20TTS%20with%20robust%20alignment%20learning%20and%20diverse%20synthesis%20笔记.md) 中提出的框架，来简化一些 TTS 模型中的对齐学习
3. 同时在自回归的 TTS 中加了一个约束来直接最大化给定 mel 谱 后的 文本似然；进一步分析一个简单的 static alignment prior 来引导对齐学习

## 对齐学习框架

如图：
![](image/Pasted%20image%2020231201150619.png)

输入为编码后的文本 $\Phi\in\mathbb{R}^{C_{txt}\times N}$，目标是将其对齐到 mel 谱 $X\in\mathbb{R}^{C_{mel}\times T}$，其中 $T$ 是 mel 帧的长度，$N$ 为文本的长度。下面给出学习目标和其在自回归和并行 TTS 模型中的应用。

### 无监督对齐学习目标

目标是最大化给定 mel 谱 之后文本的似然，用的是 HMM 中的 forward-sum 算法。然后将文本和语音的对齐约束为单调的来避免 missing 或 repeating 的问题。

下面的公式将似然求和：
$$P\left(S(\Phi)\mid X;\theta\right)=\sum_{\mathbf{s}\in S(\Phi)}\prod_{t=1}^TP(s_t\mid x_t;\theta)$$
其中 $s$ 就代表某种特定的对齐方案（如 $\begin{aligned}s1=\phi_1,s2=\phi_1,s3=\phi_2,\ldots,sT=\phi_N\end{aligned}$），然后 $S\left(\Phi\right)$ 为所有可能的单调对齐的集合。$P(s_t|x_t)$ 为某个特定的文本 $s_t=\phi_i$ 对齐到时刻 $t$ 的 mel 谱帧 $x_t$ 的似然。

然后就可以用现有的 CTC 来计算这个函数。

### 自回归 TTS

自回归 TTS 通常采用 attention 来学习在线对齐，但是对于长序列和 OOD 文本会出现一些很大的问题。

本文采用 [Flowtron- an Autoregressive Flow-based Generative Network for Text-to-Speech Synthesis 笔记](../表达性合成与韵律合成/Flowtron-%20an%20Autoregressive%20Flow-based%20Generative%20Network%20for%20Text-to-Speech%20Synthesis%20笔记.md) 中的 stateful content based attention 和 hybrid attention 配置，采用 Tacotron2 encoder 获取编码后的表征 $(\phi_i^{\boldsymbol{e}nc})_{i=1}^N$，然后用 attention RNN 来产生 $h_t$，整个 attention 的计算如下：
$$\begin{gathered}
\begin{aligned}(h_t)_{t=1}^T=\text{RNN}(h_{t-1},x_{t-1},c_{t-1})\end{aligned} \\
\begin{aligned}c_t=\sum\alpha_{t,i}\phi_i^{\boldsymbol{enc}}\end{aligned} \\
\begin{aligned}f_t=F(\alpha_{t-1})\end{aligned} \\
e_{t,i}=-v^T\tanh(Wh_t+V\phi_i^{enc}+Uf_{t,i}) \\
P(s_t=\phi_i|x_t)=\alpha_{t,i}=Softmax(-e_t)_i, 
\end{gathered}$$
得到的注意力权重建模了分布 $P(s_{t}=\phi_{i}|x_{t})$。

### 并行 TTS

并行 TTS 的对齐学习可以解耦为一个单独的 aligner。

这里用的是 Glow-TTS 和 RAD-TTS 中的 soft alignment 分布：
$$\begin{aligned}
&D_{i,j}=dist_{L2}(\phi_i^{enc},x_j^{enc}), \\
&\mathcal{A}_{soft}=\text{soft}\max(-D,\text{dim}=0).
\end{aligned}$$

然后通过维特比算法，从 soft alignment map 中选择最可能的那条单调对齐路径，来将 soft alignment 转为 hard alignment。同时通过最小化两个对齐之间的 KL 散度来使得他们之间尽可能匹配：
$$\begin{aligned}\mathcal{L}_{bin}&=\mathcal{A}_{hard}\odot\log\mathcal{A}_{soft},\\\mathcal{L}_{align}&=\mathcal{L}_{ForwardSum}+\mathcal{L}_{bin}.\end{aligned}$$

### 对齐加速

训练的时候，由于 mel 谱 的长度是已知的，采用 static 2D prior 来加速对齐。主要通过：
The 2D prior substantially accelerates the alignment learning by making
far-off-diagonal elements less probable

在对齐 $P(s\mid X=x_t)$ 上采用这个先验 $f_{B}$ 来得到下面的后验：
$$\begin{aligned}
f_B(k,\alpha,\beta)& =\quad\begin{pmatrix}N\\k\end{pmatrix}\frac{B(k+\alpha)B(N-k+\beta)}{B(\alpha,\beta)}  \\
P_{posterior}(\Phi=\phi_{k}\mid X{=}x_{t})&=P(\Phi=\phi_k\mid X=x_t)\odot f_B(k,\omega t,\omega(T-t+1))
\end{aligned}$$
其中，$k=\{0,\ldots,N\}$，$\alpha,\beta$ 为 $B$ 函数的超参数，$\omega$ 为控制先验分布的缩放因子。

## 实验