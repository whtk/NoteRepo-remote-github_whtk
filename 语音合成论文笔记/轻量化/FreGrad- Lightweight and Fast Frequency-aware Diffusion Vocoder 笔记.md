>  ICASSP 2024，韩国，KAIST

1. 提出 FreGrad，采用轻量化且快速的基于 diffusion 的 vocoder 生成真实音频，包含三部分：
	1. 采用离散小波变换将复杂的波形解耦为子波，从而可以在简单的特征空间上操作
    2. 设计了 frequency-aware dilated convolution，提高了频率感知度，从而生成有准确频率的语音
    3. 引入了一些技巧，提高了模型的生成质量
2. FreGrad 相比 baseline，训练时间快 3.7 倍，推理速度快 2.2 倍，模型大小减少 0.6 倍（只有 1.78M 参数），而且不影响输出质量

> 基于 DiffWave，输入变为高低频的小波，提出一个 frequency-aware dilated convolution 来提高频率建模能力，和一些其他的技巧来提高 diffusion 的生成质量。

## Introduction

1. 之前采用非 AR 的方法，包括 flow、GAN 和信号处理，加速了推理速度，但是质量不如 AR 方法
2. 最近基于 diffusion 的 vocoder 生成质量很好，但是训练收敛慢、推理效率低、计算成本高
3. 本文提出 FreGrad，采用离散小波变换将复杂波形解耦为两个简单的频率子波，从而避免了大量计算：
    1. 使用 DWT 将复杂波形转换为两个频率稀疏、维度较小的小波特征，且不丢失信息
    2. 可以大幅减少模型参数和去噪处理时间
    3. 引入 frequency-aware dilated convolution（Freq-DConv）模块，提高输出质量
    4. 将 DWT 用于 dilated convolutional layer，提供频率信息的 inductive bias，使得模型可以学习准确的频谱分布，从而实现真实音频合成
    5. 为每个小波特征设计先验分布，使用 multi-resolution 幅度损失函数
4. 实验结果表明，FreGrad 在提高模型效率的同时保持生成质量：
    1. 推理速度提高 2.2 倍，模型大小减少 0.6 倍，MOS 与现有工作相当

## 背景（略）

主要介绍 diffusion 原理。

## FreGrad

FreGrad 网络架构基于 [DiffWave- A Versatile Diffusion Model for Audio Synthesis 笔记](../DiffWave-%20A%20Versatile%20Diffusion%20Model%20for%20Audio%20Synthesis%20笔记.md)，但是在简单的小波特征空间上操作，并用 Freq-DConv 替换了现有的 dilated convolution。

### Wavelet 特征去噪

在进行前向过程之前使用 DWT。DWT 将目标维度的音频 $x_0 \in \mathbb{R}^L$ 下采样为两个小波特征 $x_0^l,x_0^h \subset \mathbb{R}^\frac{L}{2}$，分别表示低频和高频成分。
> DWT 可以将非平稳信号分解为两个小波特征，且不会丢失信息

FreGrad 则在小波特征上工作。在每个训练 step 中，小波特征 $x_0^l$ 和 $x_0^h$ 通过不同的噪声 $\epsilon^l$ 和 $\epsilon^h$ 得到带噪特征，两个噪声用神经网络 $\epsilon_\theta(·)$ 同时近似。在反向过程中，FreGrad 生成去噪的小波特征，最后通过逆 DWT（iDWT）转换为目标维度的波形 $\hat{x}_0 \in \mathbb{R}^L$：
$${\hat{x}}_0=\Phi^{-1}({\hat{x}}_0^l,{\hat{x}}_0^h),$$
其中 $\Phi^{-1}$ 表示逆 DWT。

FreGrad 通过解耦复杂波形生成音频，计算量更小。此外，iDWT 保证了从小波特征到波形的无损重构。实验中采用 Haar 小波。

### Frequency-aware Dilated Convolution

准确地重建频率分布对于音频合成很重要。提出 Freq-DConv，引导模型关注频率信息。如图：
![](image/Pasted%20image%2020240329102337.png)

用 DWT 将 hidden signal $y \in \mathbb{R}^{\frac{L}{2} \times D}$ 分解为两个 sub-band $y_l,y_h \subset \mathbb{R}^{\frac{L}{4}\times D}$，隐藏维度为 $D$。然后 sub-band 进行 channel-wise 拼接，再通过 dilated convolution $f(·)$ 提取频率感知特征 $y_{\text{hidden}} \in \mathbb{R}^{\frac{L}{4}\times 2D}$：
$${y}_{hidden}=\mathbf{f}(\mathsf{cat}({y}_l,{y}_h)),$$
其中 $\mathsf{cat}$ 表示拼接操作。提取的特征 $y_{\text{hidden}}$ 沿着通道维度分为 $y_l^\prime,y_h^\prime \subset \mathbb{R}^{\frac{L}{4}\times D}$，最后 iDWT 将这两个特征转换为单个表征，其长度与输入特征 $y$ 相同：
$$y^{\prime}=\Phi^{-1}({y}_l^{\prime},{y}_h^{\prime}),$$

其中 $y^\prime \in \mathbb{R}^{\frac{L}{2}\times D}$ 为 Freq-DConv 的输出。
> Freq-DConv 会嵌入到每个 ResBlock 中。

在 dilated convolution 之前解耦 hidden signal 是为了增加 time level 上的感受野。DWT 使得每个小波特征的时间维度减小，但是可以保留时间相关性。从而卷积层可以在 time level 上有更大的感受野。而且 hidden feature 的低频和高频子波可以分开计算，这在模型中提供了关于频率信息的 inductive bias，有助于生成频率一致的波形。

### 提高生成质量的 tricks

**先验分布**：spectrogram-based 的先验分布可以提高波形去噪性能。于是这里为每个小波序列设计一个先验分布。由于每个 sub-band 包含特定的低频或高频信息，对每个小波特征使用独立的先验分布。具体来说，沿频率维度将 mel-spectrogram 分为两个部分，并采用 [PriorGrad- Improving Conditional Denoising Diffusion Models with Data-Dependent Adaptive Prior 笔记](../../其他文章/PriorGrad-%20Improving%20Conditional%20Denoising%20Diffusion%20Models%20with%20Data-Dependent%20Adaptive%20Prior%20笔记.md) 中的方法来从每个部分获得先验分布 $\{\sigma_l,\sigma_h\}$。

**Noise schedule transformation**：理想情况下，前向过程的最后一个 time step 的信噪比（SNR）应该为零。但是之前的工作中的 noise schedule 无法到零。于是采用已有的算法，可以表示为：
$$\sqrt{\gamma}_{new}=\frac{\sqrt{\gamma}_0}{\sqrt{\gamma}_0-\sqrt{\gamma}_T+\tau}(\sqrt{\gamma}-\sqrt{\gamma}_T+\tau),$$

其中 $\tau$ 用于避免在采样过程中除以零。

**损失函数**：diffusion vocoder 的常见训练目标是最小化预测和 GT 噪声之间的 L2 范数，缺乏频率方面的显式反馈。为了给模型提供频率感知的反馈，添加 multi-resolution 短时傅里叶变换（STFT）幅度损失 $L_{\text{mag}}$。但是 FreGrad 只用幅度部分（实验发现，引入 spectral convergence loss 会降低输出质量）。设 $M$ 为 STFT 损失的数量，则 $L_{\text{mag}}$ 可以表示为：
$$\mathcal{L}_{mag}=\frac1M\sum_{i=1}^M\mathcal{L}_{mag}^{(i)},$$
其中 $\mathcal{L}_{mag}^{(i)}$ 是来自第 $i$ 个 setting 的 STFT 损失。

将低频和高频子波分别应用到 diffusion loss，最终的训练目标为：
$$\mathcal{L}_{final}=\sum_{i\in\{l,h\}}\left[\mathcal{L}_{diff}(\boldsymbol{\epsilon}^i,\boldsymbol{\hat{\epsilon}}^i)+\lambda\mathcal{L}_{mag}(\boldsymbol{\epsilon}^i,\boldsymbol{\hat{\epsilon}}^i)\right],$$

其中 $\hat{\boldsymbol{\epsilon}}$ 是估计的噪声。

## 实验（略）
