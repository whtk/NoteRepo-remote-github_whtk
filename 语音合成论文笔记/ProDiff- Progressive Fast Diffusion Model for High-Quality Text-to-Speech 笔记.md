> 2022，浙江大学

1. DDPM 的迭代采样很耗时
2. 提出 ProDiff，通过直接预测干净数据来参数化去噪模型，为了减少迭代次数，通过 知识蒸馏 来减少数据方差
3. 具体来说，去噪模型使用来自 N step DDIM teacher 生成的 mel谱 作为训练目标，然后将学习到的知识蒸馏到 N/2 的 step 中
4. 只要两次迭代就可以合成高保真的 mel谱，在单个NVIDIA 2080Ti GPU 实现了比实时快24倍的采样速度

## Introduction

TTS 的三个目标：
+ 高质量
+ 快速
+ 多样性，防止语音过于乏味，缓解  mode collapse

DDPM 在语音合成有两个阻碍：
+ 本质是基于 score matching 目标函数的梯度模型，需要很多迭代步才可以实现高性能
+ 减少迭代时，性能发生退化

本文分析用于 TTS 的 diffusion 的参数化，之前的模型通过估计数据密度的梯度来生成样本，从而需要多次迭代，而使用神经网络直接预测 clean 数据来参数化去噪模型的方法能够加速采样。提出 ProDiff：
+ 直接预测干净的数据 $\boldsymbol{x}$，不需要估计 score matching 的梯度
+ 通过知识蒸馏减少数据在目标侧的差异

## diffusion 背景（略）

## 参数化 diffusion 模型

当前 diffusion 参数化可以分两类：
+ 去噪模型学习数据对数密度的梯度，预测 $\boldsymbol{\epsilon}$，称为基于梯度的方法（其实就是 DDPM的方法）
+ 直接预测原始的 clean 数据 $\boldsymbol{x}_0$，优化重构误差，称为基于生成的方法
	+ 由于 $\boldsymbol{x}_t$ 有着不同的 noise level，采用一个基于梯度的模型来直接预测 $\boldsymbol{x}_{t-1}$ 肯定是很困难的，基于生成的方法不需要估计数据密度的梯度，只要估计 $\boldsymbol{x}_0$ 然后根据后验分布把噪声加回去
	+ 此时，$p_\theta\left(\boldsymbol{x}_0 \mid \boldsymbol{x}_t\right)$ 是显示的分布，是通过神经网络 $f_\theta\left(\boldsymbol{x}_t, t\right)$ 直接输入 $\boldsymbol{x}_t$ 输出 $\boldsymbol{x}_0$ 来建模的，然后根据后验分布 $q\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{x}_0\right)$ 采样生成 $\boldsymbol{x}_{t-1}$
	+ 此时损失就定义为数据空间 $\boldsymbol{x}$ 的均方误差，采用 SGD 优化：$$\mathcal{L}_\theta^{\mathrm{Gen}}=\left\|\boldsymbol{x}_\theta\left(\alpha_t \boldsymbol{x}_0+\sqrt{1-\alpha_t^2} \boldsymbol{\epsilon}\right)-\boldsymbol{x}_0\right\|_2^2, \boldsymbol{\epsilon} \sim \mathcal{N}(0, \boldsymbol{I})$$

## ProDiff

### Motivation

在 ProDiff 中提出两个关键来解决 diffusion 的问题：
+ 采用基于 生成器的参数化 加速采样，预测 clean data 可以加速
+ 通过知识蒸馏减少数据在目标侧的方差（差异）

### 教师模型的选择

通过实验发现，4-step 的基于生成的 diffusion 模型能够达到质量和速度的平衡，于是选择它作为教师模型。

### 知识蒸馏

DDIM 使用非马尔可夫生成过程来加速推理，同时还保留了 DDPM 相同的训练。受此启发，使用 sampler 直接预测 coarse-grained mel-spectrogram。

具体来说是，使用教师模型初始化 ProDiff，和之前一样从训练数据中采样并且加噪，但是不同的是，target value $\hat{\boldsymbol{x}}_0$ 是通过使用教师模型运行 两次 DDIM 的采样步来得到的。

然后再使学生模型的一次 DDIM step 和 教师模型的两次 DDIM 想匹配，从而减少了一半的 time step。

### 架构

基于 FastSpeech2 的结构，如图：![](image/Pasted%20image%2020230527213138.png)
包含：
+ phoneme encoder，把 phoneme embedding 转换成 hidden sequence，由 feed-forward transformer blocks 组成
+ variance adaptor 预测每个 phoneme 的duration，将 hidden sequence 的长度转成语音帧的长度，同时也给出一些能量、音高的特征，模型是 2层 1D 卷积 + ReLU + LN + Dropout + Linear
+ spectrogram denoiser，迭代地基于 hidden sequence 生成 mel-spectrograms，和 DiffSinger 差不多，采用 non-causal WaveNet 作为 spectrogram denoiser

### 训练损失

样本重构损失 Sample Reconstruction Loss：$$\mathcal{L}_\theta=\left\|\boldsymbol{x}_\theta\left(\alpha_t \boldsymbol{x}_0+\sqrt{1-\alpha_t^2} \boldsymbol{\epsilon}\right)-\hat{\boldsymbol{x}}_0\right\|_2^2, \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathrm{I})$$

Structural Similarity Index (SSIM) Loss 是一种用于测量图像质量的感知测度，能够捕获结构信息和纹理特征，值在 0-1 之间，越接近 1 越好：$$\mathcal{L}_{\mathrm{SSIM}}=1-\operatorname{SSIM}\left(\boldsymbol{x}_\theta\left(\alpha_t \boldsymbol{x}_0+\sqrt{1-\alpha_t^2} \boldsymbol{\epsilon}\right), \hat{\boldsymbol{x}}_0\right)$$

Variance Reconstruction Loss 音高、持续时间和能量的重构损失：$$\mathcal{L}_p=\|p-\hat{p}\|_2^2, \mathcal{L}_e=\|e-\hat{e}\|_2^2, \mathcal{L}_{d u r}=\|d-\hat{d}\|_2^2$$

### 训练和推理过程

训练：![](image/Pasted%20image%2020230527221719.png)

推理：![](image/Pasted%20image%2020230527221849.png)
推理的时候，迭代预测 $\boldsymbol{x}_0$，然后根据后验概率把噪声加上去，随着 time step 增加，生成的 mel谱细节更多。最后使用预训练的 vocoder 生成波形。

## 相关工作（略）

## 实验