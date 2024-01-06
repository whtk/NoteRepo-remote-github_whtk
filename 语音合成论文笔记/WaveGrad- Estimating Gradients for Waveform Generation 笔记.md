> ICLR，2021，JKU & Google

1. 提出 WaveGrad，是一个通过估计数据密度的梯度来进行波形生成的条件模型
2. 模型从高斯噪声开始，通过以 mel 谱 为条件的 gradient-based sampler 迭代更新信号
3. 可以通过调整步数在 推理速度和采样质量之间进行 trade off；且发现只要 6 步就可以生成高质量的样本
4. 效果超过 adversarial non-autoregressive baselines 而且 匹配 strong likelihood-based autoregressive baseline

> 是一个基于 DPM 的 vocoder，从噪声信号开始，以

## Introduction

1. 音频生成任务主要是自回归模型，但是速度慢
2. 也有很多非自回归模型，但是效果都不如自回归的好，包括：
	1. 基于  flow 的如：inverse autoregressive flows、generative flows、continuous normalizing flows
	2. 基于 GAN 的
	3. energy score
	4. VAE
	5. DSP
3. 提出 WaveGrad，是一个通过估计数据密度的梯度来进行波形生成的条件模型，是非自回归的，训练简单，只需要几步就可以生成样本
4. 本文贡献：
	1. 提出 WaveGrad
	2. 建立了两个模型，一个是基于 离散 step index；一个是基于连续的 scale；发现连续的效果更好
	3. 效果超过 adversarial non-autoregressive baselines 而且 匹配 strong likelihood-based autoregressive baseline

## 用于波形生成的 梯度估计

首先 score 定义为：
$$s(y)=\nabla_y \log p(y)$$
给定 score function $s(\cdot)$ 时，可以通过朗之万动力学进行采样：
$$\tilde{y}_{i+1}=\tilde{y}_i+\frac{\eta}{2} s\left(\tilde{y}_i\right)+\sqrt{\eta} z_i$$
其中，$\eta>0$ 为 step size，且 $z_i \sim \mathcal{N}(0, I)$。

通过训练神经网络来学习 score function，即可得到一个生成模型，然后使用朗之万动力学进行样本生成。

如果加上噪声，denoising score matching 的目标函数可以写为：
$$\mathbb{E}_{y \sim p(y)} \mathbb{E}_{\tilde{y} \sim q(\tilde{y} \mid y)}\left[\left\|s_\theta(\tilde{y})-\nabla_{\tilde{y}} \log q(\tilde{y} \mid y)\right\|_2^2\right]$$
其中，$p(\cdot)$ 为数据分布，$q(\cdot)$ 为噪声分布。

如果加上不同 level 的噪声，同时每个 level 的权重不同，则目标函数可以写为：
$$\sum_{\sigma \in S} \lambda(\sigma) \mathbb{E}_{y \sim p(y)} \mathbb{E}_{\tilde{y} \sim \mathcal{N}(y, \sigma)}\left[\left\|s_\theta(\tilde{y}, \sigma)+\frac{\tilde{y}-y}{\sigma^2}\right\|_2^2\right]$$
这里的 score function $s_\theta(\tilde{y},\sigma)$ 以 $\sigma$ 为条件，为噪声的标准差。 

WaveGrad 是上述方法的一个变体，它学习的是一个 条件生成模型 $p(y\mid x)$ 。

### 基于 DPM 的 WaveGrad

WaveGrad 建模条件分布 $p_\theta(y_0\mid x)$ ，其中 $y_0$ 为波形，$x$ 为条件（也就是 feature，如语言特征、mel 谱 、声学特征等），这个分布可以写为：
$$p_\theta\left(y_0 \mid x\right):=\int p_\theta\left(y_{0: N} \mid x\right) \mathrm{d} y_{1: N}$$
其中，$y_1, \ldots, y_N$ 为隐变量序列，每个都和 $y_0$ 的维度相同，后验分布 $q\left(y_{1: N} \mid y_0\right)$ 称为 diffusion 过程（也叫 forward 过程），通过马尔可夫链定义如下：
$$q\left(y_{1: N} \mid y_0\right):=\prod_{n=1}^N q\left(y_n \mid y_{n-1}\right)$$
每次迭代都添加高斯噪声：
$$q\left(y_n \mid y_{n-1}\right):=\mathcal{N}\left(y_n ; \sqrt{\left(1-\beta_n\right)} y_{n-1}, \beta_n I\right)$$
其中的 $\beta_1, \ldots, \beta_N$ 为 noise schedule。任意 step $n$ 可以计算为：
$$y_n=\sqrt{\bar{\alpha}_n} y_0+\sqrt{\left(1-\bar{\alpha}_n\right)} \epsilon$$
此噪声分布的梯度计算为：
$$\nabla_{y_n} \log q\left(y_n \mid y_0\right)=-\frac{\epsilon}{\sqrt{1-\bar{\alpha}_n}}$$
DDPM 通过重参数神经网络来建模 $\epsilon_n$，此时的目标函数类似于 denoising score matching：
$$\mathbb{E}_{n, \epsilon}\left[C_n\left\|\epsilon_\theta\left(\sqrt{\bar{\alpha}_n} y_0+\sqrt{1-\bar{\alpha}_n} \epsilon, x, n\right)-\epsilon\right\|_2^2\right]$$
其中 $C_n$ 是和 $\beta_n$ 有关的常数。DDPM 直接把这个系数忽略以实现一个加权的 VLB。

### noise schedule 和以 noise level 为条件

作者发现，noise schedule 的选择对高质量的生成非常重要，尤其是要减少迭代次数 $N$ 时。

另外 $N$ 的选择也很重要，是一个在样本质量和推理速度之间的 trade off 的参数。

在 WaveGrad 中，噪声条件是连续的 level 而非离散的 index ，此时损失变为：
$$
\mathbb{E}_{\bar{\alpha}, \epsilon}\left[\left\|\epsilon_\theta\left(\sqrt{\bar{\alpha}} y_0+\sqrt{1-\bar{\alpha}} \epsilon, x, \sqrt{\bar{\alpha}}\right)-\epsilon\right\|_1\right]
$$
这里就有一个问题要解决。在 DDPM 中，index 是从 $n \sim \operatorname{Uniform}(\{1, \ldots, N\})$ 中采样得到的，连续的时候就需要定义一个 sampling procedure 能够直接从 $\bar{\alpha}$ 中采样。

一个想法是从均匀分布中采样，但是实验发现效果不好。于是使用 hierarchical sampling 方法来模仿离散采样。首先定义有 $S$ 次迭代的 noise schedule，然后计算其对应的 $\sqrt{\bar{\alpha}_s}$：
$$l_0=1, \quad l_s=\sqrt{\prod_{i=1}^s\left(1-\beta_i\right)}$$
那么首先采样 $s \sim U(\{1, \ldots, S\})$，从而得到 $\left(l_{s-1}, l_s\right)$ 范围内的值，然后在这个范围内均匀采样得到 $\sqrt{\bar{\alpha}_s}$，整个模型架构如下：
![](image/Pasted%20image%2020230823214934.png)
好处就是模型只需要训练一次，就可以包含很大的路径空间。
> 也就是模型看到了很多不同的噪声，反正不只 $N$ 个。

同时使用了 FiLM 模块将输入的 mel 谱 和噪声数据进行组合。

训练好之后，也可以用不同的 $N$ 来进行采样，从而可以显式地在速度和质量进行 trade off。

整个采样和训练算法如下：
![](image/Pasted%20image%2020230823215355.png)

## 相关工作（略）

## 实验

在 LJ Speech 上 MOS 超过 WaveRNN。

