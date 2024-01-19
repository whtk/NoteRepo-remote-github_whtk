> ICASSP 2018，株式会社

1. 提出一种新的基于 CNN 的 TTS，不需要任何循环单元，比基于 RNN 的方法快很多
2. 在两张普通的游戏电脑 GPU 下训练 15 小时

## Introduction

1. Tacotron 采用了很多循环单元，所以很难训练
2. 提出  Deep Convolutional TTS (DCTTS)，全卷积的 TTS，结构和 Tacotron 很像，但是基于全卷积的 seq2seq 模型
3. 还提出了一种快速训练 attention 模块的方法，称之为 guided attention

## 预备知识

波形和复数域频谱 $Z\:=\:\{Z_{ft}\}\:\in \mathbb{C}^{F^{\prime}\times T^{\prime}}$ 可以通过 STFT 和 iSTFT 得到，$F^{\prime}\times T^{\prime}$ 分别表示 frequency bins 的数量 和 temporal bins 的数量。通过对时域的频谱 $Z$ 取幅度，得到幅度谱 $|Z|\:=\:\{|Z_{ft}|\}\:\in \mathbb{R}^{F^{\prime}\times T^{\prime}}$，然后在此基础上应用 mel ﬁlter-bank 得到 mel 谱 $S\in\mathbb{R}^{F\times T^{\prime}},(F\ll F^{\prime})$。

本文同时每 4 帧采样一次来对时间进行降维 $\lceil T^{\prime}/4\rceil=:T$，然后还进行了归一化 $S\leftarrow(S/\max(S))^\gamma$。

本文将 1D 卷积记为 $\mathsf{C}_{k\star\delta}^{o\leftarrow i}(X)$，1D 解卷积为 $\mathsf{D}_{k\star\delta}^{o\leftarrow i}(X)$。

卷积层通常会接一个 Highway Network-like gated activation，记为 $\mathsf{Highway}(X;\mathcal{L})=\sigma(H_1)\odot\mathsf{ReLU}(H_2)+(1-\sigma(H_1))\odot X$，其中 $[H_1,H_2]=\mathsf{L}(X)$ ，再定义 $\mathsf{HC}_{k\star\delta}^{d\leftarrow d}(X):=\mathsf{Highway}(X;\mathsf{C}_{k\star\delta}^{2d\leftarrow d})$。

## 提出的网络

采用下面两个网络还合成 spectrogram：
+ Text2Mel，从输入文本中合成 mel 谱
+ Spectrogram Super-resolution Network (SSRN)，从粗粒度的 mel 谱合成完整的 STFT spectrogram

![](image/Pasted%20image%2020240117155732.png)

### Text2Mel

包含四个子模块：Text Encoder, Audio Encoder, Attention, 和
Audio Decoder。

TextEnc 先将输入序列 $L=[l_1,\ldots,l_N]\in\mathsf{Char}^N$ 编码到两个矩阵 $K\text{ (key)},V\text{ (value)}\in\mathbb{R}^{d\times N}$，然后 AudioEnc 编码  coarse mel spectrogram $\begin{aligned}S=S_{1:F,1:T}\in\mathbb{R}^{F\times T}\end{aligned}$ 到 $Q\text{ (query)}\in\mathbb{R}^{d\times T}$：
$$(K,V)=\mathsf{Text}\mathsf{Enc}(L),~Q=\mathsf{Audio}\mathsf{Enc}(S_{1:F,1:T})$$
然后得到 attention 矩阵 $A\in\mathbb{R}^{N\times T}$：
$$A=\mathsf{softmax}_{n\text{-axis}} ( K ^ { \mathsf{T}}Q/\sqrt{d})$$
attention 后的结果为 $R\in\mathbb{R}^{d\times T}$：
$$R=\mathsf{Att}(Q,K,V):=VA.\quad\text{(Note: matrix product.)}$$
然后 $R$ 和 $Q$ 进行拼接，$R^{\prime}=[R,Q]$，AudioDec 基于此来估计 mel 谱：
$$Y_{1:F,2:T+1}=\mathsf{AudioDec}(R^{\prime})$$
然后需要和 $S_{1:F,2:T+1}$ 尽可能相似，损失函数为 $\mathcal{L}_{\mathrm{spec}}(Y_{1:F,2:T+1}|S_{1:F,2:T+1})$，包含 L1 loss 和 binary divergence $\mathcal{D}_\mathrm{bin}$：
$$\begin{aligned}
\mathcal{D}_{\mathrm{bin}}(Y|S) \begin{aligned}:=\mathbb{E}_{ft}\left[-S_{ft}\log\frac{Y_{ft}}{S_{ft}}-(1-S_{ft})\log\frac{1-Y_{ft}}{1-S_{ft}}\right]\end{aligned}  \\
=\mathbb{E}_{ft}[-S_{ft}\hat{Y}_{ft}+\log(1+\exp\hat{Y}_{ft})]+\mathrm{const.},
\end{aligned}$$
其中 $\hat{Y}_{ft}\:=\:\operatorname{logit}(Y_{ft})$。

总损失为：
$$\mathcal{L}_{\mathrm{spec}}(Y|S)=\mathcal{D}_{\mathrm{bin}}(Y|S)+\mathbb{E}[|Y_{ft}-S_{ft}|]\geq0$$

### TextEnc，AudioEnc 和 AudioDec 细节

网络是纯卷积的，不依赖任何循环单元，且用的是 dilated convolution。具体结构如上图。

AudioEnc 和 AudioDec 用的卷积是 1D causal convolution。

### Spectrogram Super-resolution Network（SSRN）

由于最终的目的是从 coarse mel 谱 $Y\in\mathbb{R}^{F\times T}$ 中合成完整的 spectrogram $|Z|\in\mathbb{R}^{F^{\prime}\times4T}$，于是需要 SSRN。将频率 channel 从 $F$ 上采样 $F^\prime$ 可以简单地通过改变卷积通道数来实现，而时间维度的增加则需要使用 两层 stride 为 2 的解卷积实现。

细节如上图，SSRN 中所有的卷积都是 non-causal 的，损失函数和 Text2Mel 一样，但是是 $S$ 和 $|Z|$ 之间的。

## Guided Attention

### Guided Attention Loss

通常 attention 的训练成本很高，于是可以引入一些先验来减轻负担。

TTS 中，attention matrix 位于 $\mathbb{R}^{N\times{T}}$ 空间下的一个小的子空间，因为文本和语音是随着时间近乎线性的关系（和机器翻译不一样）。

于是引入所谓的 guided attention loss，是的 attention 矩阵接近对角：
$$\mathcal{L}_{\mathrm{att}}(A)=\mathbb{E}_{nt}[A_{nt}W_{nt}],\text{ where }W_{nt}=1-\exp\{-(n/N-t/T)^2/2g^2\}$$
本文设置 $g=0.2$，当 $A$ 远离对角时，损失项很大。这个损失和前面的 $\mathcal{L}_{\mathrm{spec}}$ 一起优化。

引入这个之后，可以提高训练效率，5K 左右的迭代之后 attention 就大概对上了。

### 合成阶段的 Forcibly Incremental Attention 

合成的时候，attention 矩阵 $A$ 有时候不能关注于正确的 character，一些 error 包括：
+ 跳过一些字母
+ 重复相同的单词

通过以下规则修改 $A$ 使其接近对角：
+ 设 $n_t$ 为时刻 $t$ 下被读的单词的位置，即 $\begin{aligned}n_t=\text{argmax}_nA_{nt}\end{aligned}$
+ 考虑其和前一个位置的差，如果 $-1\leq n_t-n_{t-1}\leq3$
+ 将当前的 attention 强制设为 $A_{nt}=\delta_{n,n_{t-1}+1},(\text{Kronecker delta})$
+ 从而 $n_t=n_{t-1}+1$

## 实验（略）