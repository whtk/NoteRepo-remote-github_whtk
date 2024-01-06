> ICML 2021，平安科技

1. 提出非自回归的 EfficientTTS，不需要任何外部的 aligner，可以直接端到端地优化参数
2. 通过将 EfficientTTS 和不同的 FFN 组合，开发了一系列的 TTS 模型，包含 text-to-mel 和 text-to-waveform 的网络
3. 在语音质量、训练效率和合成速度上超过 Tacotron 2 和  Glow-TTS

## Introduction

1. 提出 EfficientTTS，贡献如下：
	1. 提出一种新的产生 soft 或 hard 单调对齐的方法，可以引入任何的 attention 机制而不受网络结构约束
	2. 提出 EfficientTTS，是 fully parallel, fully convolutional，可以端到端训练
	3. 基于 EfficientTTS 开发了一系列的 TTS，包含：
		1. EFTS-CNN
		2. EFTS-Flow
		3. EFTS-Wav

## 相关工作（略）

## 采用 IMV 的单调对齐建模

提出 index mapping vector (IMV)，采用 IMV 进行单调对齐建模。

### IMV 定义

设 $\alpha\in\mathcal{R}^{(T_{1},T_{2})}$ 为输入序列 $x\in\mathcal{R}^{(D_1,T_1)}$ 和输出序列 $y\in\mathcal{R}^{(D_2,{T}_2)}$ 之间的对齐矩阵，定义 IMV $\mathbb{\pi}$ 为 index vector $\boldsymbol{p}=\{0,1,\cdots,T_1-1\}$ 的加权和，权重为 $\alpha$：
$$\pi_j=\sum_{i=0}^{T_1-1}\alpha_{i,j}*p_i$$ 其中，$0\leq j\leq T_2-1,\boldsymbol{\pi}\in\mathcal{R}^{T_2},\sum_{i=0}^{T_1-1}\alpha_{i,j}=1$。

IMV 可以看成是每个输出 time step 的期望位置，这个位置在 $0-T_1-1$ 之间都有可能。

### 基于 IMV 的单调对齐

对齐矩阵的单调和连续性等价于：
$$0\leq\Delta\pi_i\leq1$$
其中 $\Delta\pi_i\:=\:\pi_i\:-\:\pi_{i-1},1\:\leq\:i\:\leq\:T_2\:-1$。

在 $\pi$ 为连续和单调之后，完整性等价于下述的边界条件：
$$\begin{aligned}
\pi_{0}& =0,  \\
\pi_{T_{2}-1}& =T_{1}-1. 
\end{aligned}$$
### 将 IMV 引入网络

提出两种方法将 IMV 引入 seq2seq 网络：: Soft Monotonic Alignment (SMA) 和 Hard Monotonic Alignment (HMA)。

SMA：
将上述的三个限制条件引入到训练的目标函数中：
$$\begin{aligned}
\mathcal{L}_\text{SMA}& =\lambda_0\|\Delta\pi|-\Delta\pi\|_1  \\
&+\lambda_1\||\Delta\pi-1|+(\Delta\pi-1)\|_1 \\
&+\lambda_2\|\frac{\pi_0}{T_1-1}\|_2 \\
&+\lambda_3\|\frac{\pi_{T_2-1}}{T_1-1}-1\|_2,
\end{aligned}$$
当 $\pi$ 满足上述条件时，$\mathcal{L}_\text{SMA}$ 为 0，因此，通过将上述损失函数引入训练过程中，可以在不修改网络结构的条件下实现连续单调对齐。

HMA：
上述方法虽然可以通过损失函数实现单调对齐，但是训练的过程可能很困难，因为刚开始训练的时候网络很难产生单调对齐，于是提出 HMA。首先，从对齐矩阵 $\alpha$ 中计算 $\pi^\prime$，此时的 $\pi^\prime$ 不是单调的，然后通过使用 RELU 激活强制 $\Delta \pi>0$ ，从而得到一个严格单调的 IMV：
$$\begin{gathered}
\Delta\pi_{j}^{\prime} =\pi_j^{\prime}-\pi_{j-1}^{\prime}, 0<j\leq T_{2}-1, \\
\Delta\pi_{j} =\mathrm{ReLU}(\Delta\pi_\mathrm{j}^{\prime}), 0<j\leq T_{2}-1, \\
\pi j =\begin{cases}0,&j=0\\\sum_{m=0}^j\Delta\pi_m.&0<j\leq T_2-1.\end{cases} 
\end{gathered}$$
然后为了显示 $\pi$ 的范围在 $[0,T_1-1]$ 之间，将其乘以一个正数：
$$\begin{aligned}\pi_j^*&=\pi_j*\frac{T_1-1}{\max(\boldsymbol{\pi})}\\&=\pi_j*\frac{T_1-1}{\pi_{T_2-1}},\quad\quad\quad0\leq j\leq T_2-1.\end{aligned}$$
目标是为了重构单调对齐，于是引入下面的变换，采用高斯核来重构对齐：
$$\alpha_{i,j}^{\prime}=\frac{\exp{(-\sigma^{-2}(p_i-\pi_j^*)^2)}}{\sum_{m=0}^{T_1-1}\exp{(-\sigma^{-2}(p_m-\pi_j^*)^2)}}$$


## EfficientTTS 架构

## EfficientTTS 系列模型

## 实验（略）