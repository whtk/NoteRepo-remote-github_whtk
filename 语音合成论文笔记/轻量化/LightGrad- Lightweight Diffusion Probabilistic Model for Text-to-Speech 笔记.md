> ICASSP 2023，清华深研院、地平线、WeNet、港中文

1. 在边缘设备中运行 DPM 存在两个问题：
	1. 当前的 DPM 并不足够轻量化
	2. DPM 推理时需要很多 step，从而增加 latency
2. 提出 LightGrad：
	1. 采用 lightweight U-Net diffusion decoder + training-free 的快速采样，同时减少参数和 latency
3. LightGrad 也可以实现流式推理
4. 相比于 Grad-TTS，减少了 62.2% 的参数，减少了 65.7% 的 latency，在 4 个 step 下实现 comparable 的语音质量

> 毫无创新点，DPM-solver-1 + 深度可分离卷积 直接套上去。

## Introduction

1. 没有人提出 用于 TTS 的 lightweight DPM ，DPM 在边缘设备部署存在上述的两个问题
2. 提出 LightGrad，模型更小，推理速度更快：
	1. 提出 lightweight U-Net，将卷积替换为 depthwise separable convolutions
	2. 采用 training-free fast sampling technique 来加速推理
	3. 还实现了流式推理

## 方法

基于 [Grad-TTS- A Diffusion Probabilistic Model for Text-to-Speech 笔记](../Grad-TTS-%20A%20Diffusion%20Probabilistic%20Model%20for%20Text-to-Speech%20笔记.md)，包含 encoder、duration predictor 和 lightweight U-Net decoder。

### DPM 原理（略）

### 快速采样技术

从 DPM 中采样可以看成是求解对应的 反向 SDE 或 ODE 方程，本文选择 ODE。采用的是 [DPM-Solver- A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps 笔记](../../其他文章/DPM-Solver-%20A%20Fast%20ODE%20Solver%20for%20Diffusion%20Probabilistic%20Model%20Sampling%20in%20Around%2010%20Steps%20笔记.md) 中的方法。

设神经网络 $s_{\theta}$ 训练用于估计 $\nabla\operatorname{log}p_t$。采样过程从 $X_T\sim\mathcal{N}(\mu,I)$ 开始，求解下式：
$$dX_t=\frac12\Big((\mu-X_t)-s_\theta(X_t,\mu,t)\Big)\beta_tdt$$
令 $\begin{aligned}Y_t=X_t-\mu,\text{where }Y_T\sim\mathcal{N}(0,I)\end{aligned}$，此时：
$$dY_t=-\frac12\beta_tY_tdt-\frac12\beta_ts_\theta(Y_t+\mu,\mu,t)dt$$
DPM-solver 表明，上式存在线性和非线性部分：$-\frac12\beta_tY_tdt$ 为线性部分，$-\frac12\beta_ts_\theta(Y_t+\mu,\mu,t)dt$ 为非线性部分。

对于 $s\in(0,T),t\in[0,s]$，上式的解为：
$$Y_t=\frac{\alpha_t}{\alpha_s}Y_s+\alpha_t\int_{\lambda_s}^{\lambda_t}e^{-\lambda}\sqrt{\Sigma_{\tau_\lambda}}s_\theta(Y_{\tau_\lambda}+\mu,\mu,\tau_\lambda)d\lambda $$
其中，$\alpha_t=e^{\frac12\rho_t},\sigma_t=\sqrt{\Sigma_t},\lambda_t=\lambda(t)=\log\frac{\alpha_t}{\sigma_t},\tau_\lambda=\lambda^{-1}(\lambda)$。

从而给定 $Y_s$，其近似解 $Y_t$ 等效于近似上述积分，从而避免了线性项的误差。将 $s_{\theta}$ 用一阶泰勒展开，从而 DPM-Solver-1 求解为：
$$Y_t=\frac{\alpha_t}{\alpha_s}Y_s+\sigma_t(e^{\lambda_t-\lambda_s}-1)\sqrt{\Sigma_s}s_\theta(Y_s+\mu,\mu,s)$$

### Lightweight U-Net

Lightweight U-Net 和一般的 diffusion decoder 之间的差异就是替换普通卷积为 depthwise separable convolutions。结构包含  三个下采样模块，一个  middle block 两个上采样 blocks 和一个 final convolution block，结构如图：
![](image/Pasted%20image%2020240109211550.png)

且上下采样层都是用 separable resnet block（图b） 和 linear attention layer（图c） 构建的。

具体模型细节见论文。

### 流式推理

decoder 的输入分为几个块，块的长度在预定义好的范围内。对每个块来生成音频。

## 实验（略）