
> interspeech 2021，名古屋大学
1. 基于 multiband WaveRNN data-driven linear prediction，实现离散的波形建模
2. 提出了一种新的用于离散波形建模的数据驱动线性预测方法，LP 系数以 data-driven 的方式估计
3. 提出了一种基于短时傅立叶变换（STFT）的新型损失函数，用于Gumbel 近似下的离散波形建模

> 可不可以理解为，LP 系数不是和 LPCNet 一样用 Levinson-Durbin 算法计算的，而是通过了神经网络（GRU）来预测。
> 而所谓的离散建模就是输出的是一个 5-bit coarse/fine 1-hot vectors。

## Introduction
1. neural vocoder  可以分为两类：自回归的和非自回归的
2. 本文采用大型的稀疏 WaveRNN + multiband，生成10bit mu律波形
3. 为了提高处理多说话人的能力和提高合成质量，提出两个方法进行离散波形建模：
	1. 提出采用 Gumbel approximation 的基于 STFT 的损失函数
	2. 提出采用数据驱动的线性预测技术来生成离散波形
5. 可以生成高品质音频（seen和unseen的speaker下），并且每个speaker只有60条训练语句，同时还能实现实时性。

## 方法

令 $\boldsymbol{s}=\left[s_{1}, \ldots, s_{t_{s}}, \ldots, s_{T_{s}}\right]^{\top}$ 为离散波形样本，$\boldsymbol{c}=\left[\boldsymbol{c}_{1}^{\top}, \ldots, \boldsymbol{c}_{t_{f}}^{\top}, \ldots, \boldsymbol{c}_{T_{f}}^{\top}\right]^{\top}$ 为 conditioning 特征向量，其中 $\boldsymbol{c}_{{t}_f}$ 为 $d$ 维输入特征。sample level 的序列长度定义为 $T_s$，frame level 的长度定义为 $T_f$。

考虑 bands 的数量为 $M$，则第 $m$ 个 band 的波形序列为 $\boldsymbol{s}^{(m)} = \left[s_{1}^{(m)}, \ldots, s_{t}^{(m)}, \ldots, s_{T_{m}}^{(m)}\right]^{\top}$，显然有，$T_m = T_s/M$，对应的上采样conditioning 特征向量为 $\boldsymbol{c}^{(u)}=\left[\boldsymbol{c}_{1}^{(u)^{\top}}, \ldots, \boldsymbol{c}_{t}^{(u)^{\top}}, \ldots, \boldsymbol{c}_{T_{m}}^{(u)^{\top}}\right]^{\top}$。目标是建模离散波形的PMF：
$$
p(\boldsymbol{s})=\prod_{m=1}^{M} \prod_{t=1}^{T_{m}} p\left(s_{t}^{(m)} \mid \boldsymbol{c}_{t}^{(u)}, \boldsymbol{s}_{t-1}^{(M)}\right)=\prod_{m=1}^{M} \prod_{t=1}^{T_{m}} \boldsymbol{p}_{t}^{(m)^{\top} \boldsymbol{v}_{t}^{(m)}}
$$
其中 $\quad \boldsymbol{s}_{t-1}^{(M)}=\left[s_{t-1}^{(1)}, \ldots, s_{t-1}^{(m)}, \ldots, s_{t-1}^{(M)}\right]^{\top}$ 表示同一时刻下所有的 子带样本， $\boldsymbol{p}_{t}^{(m)}=\left[p_{t}^{(m)}[1], \ldots, p_{t}^{(m)}[b], \ldots, p_{t}^{(m)}[B]\right], \quad \boldsymbol{v}_{t}^{(m)}=$ $\left[v_{t}^{(m)}[1], \ldots, v_{t}^{(m)}[b], \ldots, v_{t}^{(m)}[B]\right]^{\top}, \quad \sum_{b=1}^{B} v_{t}^{(m)}[b]=1$, $v_{t}^{(m)}[b] \in\{0,1\}, B$ 是 sample bins 的数量， $\boldsymbol{v}_{t}^{(m)}$ 是 1-hot 向量， $\boldsymbol{p}_{t}^{(m)}$ 是概率向量 (即网络输出)。
> 非 0 即 1 和$\left[v_{t}^{(m)}[1], \ldots, v_{t}^{(m)}[b], \ldots, v_{t}^{(m)}[B]\right]^{\top}, \quad \sum_{b=1}^{B} v_{t}^{(m)}[b]=1$ 和 $v_{t}^{(m)}[b] \in\{0,1\}$ 就表明这是 ont-hot vector。

### 用于离散建模的数据驱动线性预测（DLP）
提出采用数据驱动的线性预测来计算 $\boldsymbol{p}_{t}^{(m)}$。其中，每个 bin $p_t^{(m)}(b)$ 通过 softmax 函数计算如下：
$$
p_{t}^{(m)}[b]=\frac{\exp \left(\hat{o}_{t}^{(m)}[b]\right)}{\sum_{j=1}^{B} \exp \left(\hat{o}_{t}^{(m)}[j]\right)}
$$
定义包含所有sample bins的logits向量为 $\hat{\boldsymbol{o}}_t^m = \left[\hat{o}_{t}^{(m)}[1], \ldots, \hat{o}_{t}^{(m)}[b], \ldots, \hat{o}_{t}^{(m)}[B]\right]^{\top}$，则数据驱动的线性预测计算如下：
$$
\hat{\boldsymbol{o}}_{t}^{(m)}=\sum_{k=1}^{K} a_{t}^{(m)}[k] \boldsymbol{v}_{t-k}^{(m)}+\boldsymbol{o}_{t}^{(m)}
$$
$a_t^{(m)}[k]$ 代表时间 $t$ 下第 $m$ 个 band 的第 $k$ 个系数，$k$ 为 LPC 的索引总系数个数为 $K$。

### 网络结构

![](image/Pasted%20image%2020230916165908.png)

Conditioning  输入特征进入分段卷积层（ segmental convolution）产生高维的特征向量，再进入全连接层和RELU。

coarse 和 fine embedding层用来对 5 bit 的 one-hot 向量进行编码，并且在所有的band之间进行共享。

稀疏 GRU的 hidden state unit 为1184，密集 GRU 则为 32。

dual fully-connected 用于产生两个通道的输出，然后通过可训练的加权向量进行加权，和 [LPCNet- Improving Neural Speech Synthesis Through Linear Prediction 笔记](../语音合成论文笔记/LPCNet-%20Improving%20Neural%20Speech%20Synthesis%20Through%20Linear%20Prediction%20笔记.md) 一样。

DualFC 的每个输出通道包含对应于数据驱动的 LP vector $\boldsymbol{a}_t^{(M)}=\left.[\boldsymbol{a}_t^{(1)^\top},\ldots,\boldsymbol{a}_t^{(m)^\top},\ldots,\boldsymbol{a}_t^{(M)^\top}]^\top\right.$ 和 logit vector $\begin{aligned}o_t^{(M)}&=[o_t^{(1)^\top},\ldots,o_t^{(m)^\top},\ldots,o_t^{(M)^\top}]^\top\end{aligned}$。
> 这里输出的 $\boldsymbol{a}_t^{(M)}$ 就是用神经网络预测的 LP 系数。

### 基于 STFT 的损失函数（用于离散建模）

Gumbel sampling 用于获取第 $m$ 个子带的第 $t$ 时刻的采样概率 $\hat{\boldsymbol{p}}_t^{(m)}=[\hat{\boldsymbol{p}}_t^{(m)}[1],\ldots,\hat{\boldsymbol{p}}_t^{(m)}[b],\ldots,\hat{\boldsymbol{p}}_t^{(m)}[B]]^{\top}$，而其中的每个 bin 的概率为：
$$\hat{p}_t^{(m)}[b]=\frac{\exp(\hat{\gamma}_t^{(m)}[b])}{\sum_{j=1}^B\exp(\hat{\gamma}_t^{(m)}[j])},$$
其中：
$$\hat{\boldsymbol{\gamma}}_t^{(m)}=\hat{\boldsymbol{o}}_t^{(m)}-\log(-\log(\boldsymbol{u})),\mathrm{s.~t.~}\boldsymbol{u}\sim(0,1),$$
且：
$$\hat{\gamma}_{t}^{(m)}=[\hat{\gamma}_{t}^{(m)}[1],\ldots,\hat{\gamma}_{t}^{(m)}[b],\ldots,\hat{\gamma}_{t}^{(m)}[B]]^{\top}$$
其中，$\boldsymbol{u}$ 是均匀分布的 $B$ 维向量。
> 这一段阐述了 gumbel sampling 的思想。

后面实在看不懂了。。。。