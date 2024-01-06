> Seoul National University，Interspeech 2021

1. 提出 Diff-TTS，实现高质量和高效的 TTS，采用 denoising diffusion 将噪声信号转换为 mel 谱
2. 为了将文本作为条件，提出了一个 基于似然的优化方法
3. 为了加速推理，采用加速采样方法可以更快地生成原始波形

> 没啥创新点，但毕竟是第一篇，就是把 DDPM 用于 mel 谱 生成，把 DDIM 用于加速采样，无了。

## Introduction

1. 本文的重点是声学模型
2. 提出 Diff-TTS，可以实现鲁棒、可控和高质量的合成，同时提出 log-likelihood-based 优化方法来训练模型
3. 同时引入 DDIM 中的加速采样方法
4. 贡献：
	1. 是第一个非自回归的 DDPM TTS 系统
	2. 只要 Tacotron2 和 Glow-TTS 一半的参数，就可以生成高质量的音频
	3. 加速采样方法可以在质量和速度之间进行 trade-off
	4. 甚至可以有效地控制韵律

## Diff-TTS

### TTS 中的 DDM

Diff-TTS 给定文本，将噪声分布转为 mel 谱，每次 diffusion transformation 对应分布：
$$q(x_t|x_{t-1},c)=\mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_tI)$$
整个 diffusion 过程就可以写为：
$$p_\theta(x_0\ldots,x_{T-1}|x_T,c)=\prod_{t=1}^Tp_\theta(x_{t-1}|x_t,c)$$
定义 $q(x_0|c)$ 为 mel 谱 分布，损失函数为：
$$minL(\theta)=\mathbb{E}_{x_0,\epsilon,t}\|\epsilon-\epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon,t,c)\|_1$$
Diff-TTS 不需要任何额外的损失，除了真实噪声和预测噪声之间的 L1 损失。

推理时的生成过程如下：
$$x_{t-1}=\frac1{\sqrt{\alpha_t}}(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t,c))+\sigma_tz_t$$

### 加速采样

直接搬运 DDIM 的方法。
> 。。。。。。

### 模型架构

![](image/Pasted%20image%2020231002231028.png)

包含：
+ Text Encoder：用的是 SpeedySpeech 中的架构
+ Duration Predictor 和 Length Regulator：用的是 FastSpeech 2 中的 duration predictor，采用  Montreal forced alignment 提取 label
+ Step Encoder 和 Decoder：decoder 就是用来预测噪声的，条件为 phoneme embedding 为 step embedding，step embedding 为 sinusoidal embedding，然后 phoneme embedding 和 step embedding 相加作为 condition

## 实验（略）

