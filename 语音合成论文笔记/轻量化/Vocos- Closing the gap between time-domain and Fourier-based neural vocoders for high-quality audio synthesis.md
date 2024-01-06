> ICLR 2024，Hubert Siuzdak，charactr AI

1. 最近基于 GAN 的 vocoder 主要是在时域进行操作
2. 但是忽略了一些 inductive bias，从而导致一些冗余和计算复杂的上采样操作
3. 基于傅立叶的时频表征更符合人类的听觉感知且可以实现快速的计算
4. 本文提出 Vocos，一种可以直接生成 Fourier spectral coefficients 的模型，不仅可以提高合成质量，而且可以极大地改进计算效率

## Introduction

相位通常是一个周期的结构，会导致在 $(-\pi,\pi]$ 上的 wrapping 问题，如图：
![](image/Pasted%20image%2020231229205354.png)
调频信号的相位随时间在 $(-\pi,\pi]$ 上变化。

本文提出 Vocos，基于 GAN 的 vocoder 来产生 STFT 系数。之前的方法通过转置卷积实现上采样，本文则在所有的 depths 下都能保持特征的 resolution 不变，然后通过傅立叶逆变换来得到波形。

### 相关工作（略）

##  Vocos

### 概览

模型基于 GAN，采用基于傅立叶的时频域表征作为 generator 的目标数据分布，不需要任何上采样的卷积操作，直接采用 STFT 的逆变换来操作（在整个模型中时间维度保持不变），如图：
![](image/Pasted%20image%2020240102101612.png)

Vocos 采用 STFT 来表征时域的音频信号为：
$$\text{STFT}_x[m,k]=\sum_{n=0}^{N-1}x[n]w[n-m]e^{-j2\pi kn/N}$$

## 模型

采用 ConvNeXt 作为 backbone，首先将输入特征 embed 到 hidden representation 维度，然后采用一系列的卷积模块。

每个模块由 large-kernel-sized depthwise convolution 和 inverted bottleneck 组成，bottleneck 里面用了 GELU 激活。
、
时值信号的傅立叶变换是共轭对称的，所以只需要使用单边带 spectrum，，即每帧 $n_{fft}/2+1$ 个系数点，然后前面的神经网络得到最终的 tensor $\mathbf{h}$ 的 channel 大小为 $n_{fft}/2+2$，然后分成两个：
$$\mathbf{m},\mathbf{p}=\mathbf{h}[1:(n_{fft}/2+1)],\mathbf{h}[(n_{fft}/2+2):n]$$
其中，$\mathbf{p}$ 为相位，$\mathbf{M}=\operatorname{exp}(\mathbf{m})$ 为幅度，然后通过分别计算正余弦得到复数的实部和虚部：
$$\begin{gathered}
\mathbf{x}=\cos(\mathbf{p}) \\
\mathbf{y}=\operatorname{sin}(\mathbf{p}) 
\end{gathered}$$
最后得到的复数域的系数可以表示为：
$$\mathrm{STFT}=\mathbf{M}\cdot(\mathbf{x}+j\mathbf{y})$$
然后采用 MPD 和 MRD 作为 discriminator。

### 损失

损失包含重构损失、对抗损失和 feature mapping loss，然后这里采用的是 hinge loss 而非 least squares GAN loss，即：
$$\begin{gathered}
\begin{aligned}\ell_G(\hat{\boldsymbol{x}})=\frac{1}{K}\sum_k\max\left(0,1-D_k(\hat{\boldsymbol{x}})\right)\end{aligned} \\
\ell_D(\boldsymbol{x},\hat{\boldsymbol{x}})=\frac1K\sum_k\max\left(0,1-D_k(\boldsymbol{x})\right)+\max\left(0,1+D_k(\hat{\boldsymbol{x}})\right) 
\end{gathered}$$
重构损失为 $L_{mel}$，定义为：
$$\hat{\boldsymbol{x}}{:}L_{mel}=\left\|\mathcal{M}(\boldsymbol{x})-\mathcal{M}(\hat{\boldsymbol{x}})\right\|_1$$
feature mapping loss 为：
$$L_{feat}=\frac{1}{KL}\sum_{k}\sum_{l}\left\|D_{k}^{l}(\boldsymbol{x})-D_{k}^{l}(\hat{\boldsymbol{x}})\right\|_{1}$$

## 结果（略）
