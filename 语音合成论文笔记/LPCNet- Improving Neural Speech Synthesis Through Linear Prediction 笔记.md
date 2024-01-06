> Mozilla，Google，ICASSP 2019
> https://zhuanlan.zhihu.com/p/54952637

1. 提出 LPCNet，WaveRNN 的变体，将 线性预测 和 RNN 组合起来提高语音合成效率

## Introduction

1.  提出 LPCNet，不建模 spectral envelope，而是建模 spectrally flat excitation，从而匹配 SOTA 的合成质量

## WaveRNN

公式为：
$$\begin{aligned}
\mathbf{x}_t& =[s_{t-1};\mathbf{f}]  \\
\mathbf{u}_t& =\sigma\left(\mathbf{W}^{(u)}\mathbf{h}_{t-1}+\mathbf{U}^{(u)}\mathbf{x}_t\right)  \\
\mathbf{r}_t& =\sigma\left(\mathbf{W}^{(r)}\mathbf{h}_{t-1}+\mathbf{U}^{(r)}\mathbf{x}_t\right)  \\
\widetilde{\mathbf{h}}_t& =\tanh\left(\mathbf{r}_t\circ\left(\mathbf{W}^{(h)}\mathbf{h}_{t-1}\right)+\mathbf{U}^{(h)}\mathbf{x}_t\right)  \\
\mathbf{h}_t& =\mathbf{u}_t\circ\mathbf{h}_{t-1}+(1-\mathbf{u}_t)\circ\widetilde{\mathbf{h}}_t  \\
P\left(s_{t}\right)& =\text{softmax}\left(\mathbf{W}_2\text{ relu }(\mathbf{W}_1\mathbf{h}_t)\right), 
\end{aligned}$$
其中，$s_{t-1}$ 为上一个 time step 的样本，这里忽略了 dual softmax 和 coarse/fine split，都统一用 $s$ 表示。

详见 [WaveRNN- Efficient Neural Audio Synthesis 笔记](WaveRNN-%20Efficient%20Neural%20Audio%20Synthesis%20笔记.md)。

## LPCNet

如图：
![](image/Pasted%20image%2020230916112850.png)

包含：
+ sample rate network 
+ frame rate network

网络的输入是 20 维的特征，18 Bark-scale cepstral coefficients, and 2 pitch parameters (period, correlation)。
> 在 TTS 中，后两个值可以通过独立的网络计算

### Conditioning Parameters

对应图中左边的网络。

20 维的特征通过两个 filter size 为 $3\times 1$ 的卷积得到感受野为 5 帧的特征，residual connect 后通过两个 FC 层，最后输出 128 维的 conditioning vector $\mathbf{f}$。

### 预加重和量化

mu 律量化在高频产生的噪声是可感知的。
于是采用一阶预加重滤波器 $\begin{aligned}E(z)&=1-\alpha z^{-1}\end{aligned}$，合成后的输出再通过反向滤波器得到：
$$D(z)=\frac1{1-\alpha z^{-1}},$$
从可以可以减少感知噪声，是的 8-bit mu 律可以用于高质量的合成。

### 线性预测

神经网络通畅需要建模整个语音生成过程，包含 glottal pulses（声门脉冲）、noise excitation（噪声激励）和 response of the vocal tract（声道响应）。

但是其实声道响应可以通过一个全极点滤波器来建模。令 $s_t$ 为时刻 $t$ 的信号，基于前一时刻的线性预测表示为：
$$p_t=\sum_{k=1}^Ma_ks_{t-k}\mathrm{~,}$$
其中，$a_k$ 为当前时刻的第 $M$ 阶线性预测系数（LPC）。

$a_k$ 计算如下：
+ 首先将 18-band Bark-frequency cepstrum 转换为 线性频率功率谱密度（PSD）
+ 然后使用 inverse FFT 将 PSD 转换为 auto-correlation
+ 采用 Levinson-Durbin 算法从 auto-correlation 计算 predictor

从而可以让网络预测 excitation 而非直接预测样本点。网络输入前一个 excitation $e_{t-1}$、前一个时刻的信号 $s_{t-1}$ 和当前的预测 $p_t$。
最后的样本点满足 $s_t=e_t+p_t$。

### 输出层

将两个 FC 层组合起来做 element-wise 加权：
$$\mathrm{dual\_fc}(\mathbf{x})=\mathbf{a}_1\circ\tanh\left(\mathbf{W}_1\mathbf{x}\right)+\mathbf{a}_2\circ\tanh\left(\mathbf{W}_2\mathbf{x}\right)$$
DualFC 的输出通过 softmax 激活计算概率 $P(e_t)$。

### 稀疏矩阵

### Embedding 和 代数简化

采用 mu 律的离散特性来学习 embedding matrix $\mathbf{E}$，将 mu 律的每个值 map 到一个 embedding 中。

经过推理，中间存在一些公式的简化，最终的 运算如下：
$$\begin{aligned}
\mathbf{u}_{t}=& \sigma\left(\mathbf{W}_u\mathbf{h}_t+\mathbf{v}_{s_{t-1}}^{(u,s)}+\mathbf{v}_{p_{t-1}}^{(u,p)}+\mathbf{v}_{e_{t-1}}^{(u,e)}+\mathbf{g}^{(u)}\right)  \\
\mathbf{r}_{t}=& \sigma\left(\mathbf{W}_r\mathbf{h}_t+\mathbf{v}_{s_{t-1}}^{(r,s)}+\mathbf{v}_{p_{t-1}}^{(r,p)}+\mathbf{v}_{e_{t-1}}^{(r,e)}+\mathbf{g}^{(r)}\right)  \\
\tilde{\mathbf{h}}_t=& \tanh\left(\mathbf{r}_t\circ(\mathbf{W}_h\mathbf{h}_t)+\mathbf{v}_{s_{t-1}}^{(h,s)}+\mathbf{v}_{p_{t-1}}^{(h,p)}+\mathbf{v}_{e_{t-1}}^{(h,e)}+\mathbf{g}^{(h)}\right)  \\
\mathbf{h}_t=& \mathbf{u}_t\circ\mathbf{h}_{t-1}+(1-\mathbf{u}_t)\circ\widetilde{\mathbf{h}}_t  \\
P\left(e_{t}\right)& =\mathrm{softmax}\left(\mathrm{dual}\_\mathrm{fc}\left(\mathrm{GRU}_\mathrm{B}\left(\mathbf{h}_t\right)\right)\right),
\end{aligned}$$

### 从概率分布中采样

直接从分布中采样会引入很多噪声，通过将 logits 乘以一个常数 $c$，$c$ 定义为：
$$c=1+\max{(0,1.5g_p-0.5)}$$
其中 $g_p$ 为 pitch correlation，然后从分布中减去一个常数来确保低于阈值 $T$ 的概率为 0，从而避免一些低概率造成的脉冲噪声：
$$P^{\prime}\left(e_t\right)=\mathcal{R}\left(\max\left[\mathcal{R}\left(\left[P\left(e_t\right)\right]^c\right)-T,0\right]\right)$$
其中 $\mathcal{R}$ 用于将概率值归一化。

### 训练噪声引入

因为合成的时候样本噪声很大，为了处理这种训练和合成之间的 mismatch，在训练时加入噪声。

![](image/Pasted%20image%2020230916153017.png)

按上面这种加法，可以极大地减少合成伪影。

## 评估（略）

