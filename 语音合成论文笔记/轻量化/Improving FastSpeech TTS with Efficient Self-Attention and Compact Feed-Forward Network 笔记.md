> ICASSP 2022，微软中国

1. FastSpeech 作为一个非自回归模型，其推理过程中的延迟和计算负载从 vocoder 转移到 transformer，其效率受限于时间的二次幂
2. 提出两个模型，ProbSparseFS 和 LinearizedFS 采用高效的 self-attention 来提高推理速度和内存复杂度
3. 在 LinearizedFS 上采用 lightweight FFN 可以进一步加速推理

## Introduction

1. FastSpeech + MelGAN 的配置下，2/3 的时间都来自 FastSpeech 的transformer
2. self-attention 计算和内存消耗都很大，很多论文都提出改进：
	1. Sparse Transformer 采用 ﬁxed local and stride attention pattern
	2. Reformer 基于 locality sensitive hashing，且引入 reversible Transformer layers 来节约内存
	3. Linformer 采用 low-rank matrix 来近似 self-attention 的随机矩阵
3. 本文基于 FastSpeech 调研了两种不同的 self-attention 模块：
	1. ProbSprase 基于 Informer
	2.  Linearized Attention 基于 Linear Transformer 
	3. 同时简化了 Feed-Forward Network (FFN) 模块来进一步提高效率

## FastSpeech

下图中，b 是 encoder 和 decoder 结构，c 是 FFT 模块，d 是 FFN 模块，e 是提出的两种修改：
![](image/Pasted%20image%2020240121153549.png)

a 中还引入了 speaker 和 style 信息，这两个 embedding 和 encoder 的输出进行拼接作为 decoder 的输入。

## 高效 transformer 模块

标准的 attention 计算 attention 矩阵如下：
$$\mathcal{A}(Q,K,V)=Softmax(QK^T/\sqrt{d})V$$
其中 $Q,K,V\in\mathcal{R}^{N\times d}$，$N$ 为序列长度，$d$ 为特征维度。其内存和计算复杂度为 $O(N^2)$。

### ProbSparse Self-Attention

self-attention 是稀疏的，ProbSparse 通过忽略一些不重要的点积对来提高效率。第 $i$ 个 query 的 attention 对于第 $j$ 个 key 计算为：
$$p(K_j|Q_i)=\frac{exp(Q_iK_j^T/\sqrt{d})}{\sum_{n=1}^Nexp(Q_iK_n^T/\sqrt{d})}$$
定义第 $i$ 个 query 的稀疏性测度为：
$$M(Q_i,K)=ln\sum_{j=1}^Nexp(\frac{Q_iK_j^T}{\sqrt{d}})-\frac1N\sum_{j=1}^N\frac{Q_iK_j^T}{\sqrt{d}}$$
其实就是两个分布之间的 KL 散度。
> 用于判断这个 $Q_i$ 和所有的 Key 的相关性，值越大表明和这些 key 的关联越大，稀疏性越小。

但是，为了比较不同的 query 的稀疏性，需要计算所有的 点积对，为了避免大量的计算，采用下述近似测度：
$$\bar{M}(Q_i,K)=max_j\{\frac{Q_iK_j^T}{\sqrt{d}}\}-\frac1N\sum_{j=1}^N\frac{Q_iK_j^T}{\sqrt{d}}$$
从而只会选择 $U=NlnN$ 个点积对参与计算。此时内存和计算复杂度优化为 $O(NlnN)$。然后将 top-u 个 query 来参与 attention 的计算。

### Linearized Self-Attention

self-attention 可以看成是一种相似度的评估，基于这个思路，通用的 attention 计算的公式可以写为：
$$\mathcal{A}(Q_i,K,V)=\frac{\sum_{j=1}^Nsim(Q_i,K_j)V_j}{\sum_{j=1}^Nsim(Q_i,K_j)}$$
而标准的 attention 有 $sim(Q_i,K_j)=exp(Q_iK_j^T/\sqrt{d})$。

相似性函数可以任意替换，唯一的限制就是非负。给定映射 $\phi(x)$， 上式可以重写为：
$$\mathcal{A}(Q_i,K,V)=\frac{\sum_{j=1}^N\phi(Q_i)^T\phi(K_j)V_j}{\sum_{j=1}^N\phi(Q_i)^T\phi(K_j)}$$
采用矩阵乘法结合律，可以进一步简化为：
$$\mathcal{A}(Q_i,K,V)=\frac{\phi(Q_i)^T\sum_{j=1}^N\phi(K_j)V_j^T}{\phi(Q_i)^T\sum_{j=1}^N\phi(K_j)}$$
其中 $\sum_{j=1}^{N}\phi(K_j)V_j,\sum_{j=1}^{N}\phi(K_j)$ 这两项可以对每个 query 提前计算并重用，从而将计算和内存复杂度减小为 ${O(N)}$。 为了确保相似度为正且避免梯度为 0，映射函数选为 $\phi(x)=elu(x)+1$。

### Lightweight FFN

其实就是将 FFN 内层的线性层的维度 $ffn_{dim}$ 降低。

## 实验（略）