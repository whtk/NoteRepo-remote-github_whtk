> ICASSP 2023，索尼

1. 提出用于歌声语音 vocoder 的 hierarchical diffusion 模型
	1. 包含在不同采样率下的多个 diffusion 模型
	2. 最低采样率的模型专注于生成低频成分，如音高
    3. 其他模型则基于低采样率下的特征逐渐生成高采样率下的波形
2. 效果超过了 SOTA

## Introduction

1. diffusion 的推理速度慢，[PriorGrad- Improving Conditional Denoising Diffusion Models with Data-Dependent Adaptive Prior 笔记](../../其他文章/PriorGrad-%20Improving%20Conditional%20Denoising%20Diffusion%20Models%20with%20Data-Dependent%20Adaptive%20Prior%20笔记.md) 通过引入数据相关的先验解决了这个问题
2. 现有的 vocoder 大多关注语音信号，但是应用到歌声时效果不佳，可能是因为缺乏大规模的干净歌声数据集，以及音高、音量和发音的多样性更难建模
3. 提出 hierarchical diffusion model，学习多个不同采样率的 diffusion 模型
    1. 推理时，逐渐生成从低到高采样率的数据
    2. 最低采样率的模型专注于生成低频成分，使得音高恢复更准确；而高采样率的模型则更关注高频细节
    3. 效果优于 SOTA 的 PriorGrad 和 Parallel WaveGAN
4. 是第一个使用不同采样率的多个 diffusion 模型进行 hierarchical vocoding 的工作

## Prior Work

DDPM 略。

对于 PriorGrad，DDPM 中的标准高斯先验需要很多 step 才能获得高质量的数据。PriorGrad 使用了自适应先验 $\mathcal{N}(\mathbf{0},\boldsymbol{\Sigma}_\mathbf{c})$，其中对角方差 $\boldsymbol{\Sigma}_\mathbf{c}$ 从 mel-spectrogram $c$ 计算得到 $\boldsymbol{\Sigma}_\mathbf{c}=diag[(\sigma_0^2,\cdots,\sigma_L^2)]$，$\sigma_i^2$ 是 mel-spectrogram 第 $i$ 个 frame 的归一化能量。损失函数相应地修改为：
$$\begin{aligned}L=\mathbb{E}_{x_0,\epsilon,t}[||\epsilon-\epsilon_\theta(x_t,c,t)||_{\Sigma^{-1}}^2],\end{aligned}$$
其中 $||x||_{\Sigma^{-1}}^2=x^T\Sigma^{-1}x$。自适应先验的功率包络更接近目标信号的功率包络，因此扩散模型可以需要更少的时间步骤来收敛并且更有效。

## 方法

### hierarchical diffusion 模型

PriorGrad 用于歌声语音的效果不好，于是提出采用 diffusion 来建模不同的 resolutions，如图：
![](image/Pasted%20image%2020240213113011.png)

给定多个采样率 $f_{s}^1>f_{s}^2>\cdots>f_{s}^N$，在每个采样率都有一个独立的 diffusion 模型。每个采样率上的反向过程（生成过程）$p_\theta^i(x_{t-1}^i|x_t^i,c,x_0^{i+1})$ 都基于公共的 acoustic features $c$ 和低采样率 $f_s^{i+1}$ 的数据 $x_i$ 上进行的。
> 最低采样率的模型只基于 $c$ 。

训练时，使用 GT 数据 $x_0^{i+1}=D^i(H^i(x_0^i))$ 作为 condition noise estimation models $\epsilon_i(x_i,c,x_{i+1},t)$ 的条件，其中 $H^i(\cdot)$ 表示抗混叠滤波器，$D^i(\cdot)$ 表示下采样。

由于噪声 $\epsilon$ 是线性添加到原始数据 $x_0$ 上的，模型可以看到 GT 低采样率数据 $x^{i+1}$，因此可以避免复杂的 acoustic feature-to-waveform 转换，更简单地去从 $x^i_t$ 和 $x^{i+1}_0$ 预测低频成分的噪声。这使得模型可以更多地关注高频成分的转换。

推理时，从最低采样率 $\hat{x}^N_0$ 开始，逐步基于生成的 $\hat{x}^{i+1}_0$ 得到更高采样率的数据 $\hat{x}^{i}_0$。
> 实验发现，直接使用 $\hat{x}^{i+1}_0$ 作为条件通常会产生 Nyquist 频率附近的噪声，如图 a：
> ![](image/Pasted%20image%2020240213114317.png)
> 原因可以在于训练和推理之间的 gap，训练时使用的 GT 数据 $x^{i+1}=D^i(H^i(x^i))$ 不包含 Nyquist 频率附近的信号，而模型可以学习直接使用 Nyquist 频率附近的信号，而推理时生成的 $\hat{x}^{i+1}$ 可能包含一些信号，导致高采样率的预测被污染。

于是提出对生成的低采样率信号应用抗混叠滤波器：
$$\begin{aligned}\hat{\epsilon}=\epsilon_\theta^i(x_t^i,c,H(\hat{x}_0^{i+1}),t).\end{aligned}$$
如上图 b，这样可以去除 Nyquist 频率附近的噪声并提高质量。

训练和推理算法如图：
![](image/Pasted%20image%2020240213114617.png)

> 具体用哪种 diffusion 模型其实都可以。

### 网络架构

网络架构基于 [DiffWave- A Versatile Diffusion Model for Audio Synthesis 笔记](../DiffWave-%20A%20Versatile%20Diffusion%20Model%20for%20Audio%20Synthesis%20笔记.md)，包含 $L$ 个 residual layer，每个 block 包含 $l$ 个 layer，每个 layer 的 dilation factors 为 $[1,2,\cdots,2^{l-1}]$。DiffWave 和 PriorGrad 使用 $L=30,l=10$，这里使用 $L=24,l=8$ 来减小计算量。

所有采样率使用相同的网络架构（但感受野不同），如图：
![](image/Pasted%20image%2020240213115034.png)

低采样率的模型覆盖了更长的时间段，关注低频成分；高采样率的模型覆盖了更短的时间段，关注高频成分。

## 实验（略）
