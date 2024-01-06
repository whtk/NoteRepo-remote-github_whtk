> ICML，2021，Kakao

1. 提出并行的端到端 TTS，使用 normalizing flows 和对抗训练增强的 变分推理技术来提高生成模型的表达性
2. 提出了一个 stochastic duration predictor，从文本中合成韵律多样的语音

> 概括就是，本质是 CVAE，但是 CVAE 中的条件先验用了 normalizing flows 来建模，而且由于 条件是 text，存在 alignment 的问题，因此训练时用了 MAS 动态搜索最佳的 alignment。同时使用 MAS 得到的 duration 来训练一个 stochastic duration predictor，这个 predictor 也是 flow-based 模型，且由于 duration 的 label 是离散标量，采用了一些技巧来提高建模能力。最后进一步通过添加一个 discriminator 来进行对抗训练以提高合成质量。

## Introduction

1. two stage 的 TTS 需要 sequential training or fine-tuning
2. 最近的端到端的方法质量不太好
3. 提出并行端到端 TTS，采用 VAE 将 TTS 的两个模块通过 latent variable 连接，在条件先验分布中采用 normalizing flow，在波形建模端采用对抗训练
4. 提出 stochastic duration predictor 来解决 one-to-many 的问题
5. 两种方法都可以捕获语音的 variation

## 方法

![](image/Pasted%20image%2020230920103052.png)
图 a：训练过程
图 b：推理过程

### 变分推理

CVAE 最大化对数分布 $\log p_\theta(x|c)$ 的 ELBO：
$$\log p_\theta(x|c)\geq\mathbb{E}_{q_\phi(z|x)}\left[\log p_\theta(x|z){-}\log\frac{q_\phi(z|x)}{p_\theta(z|c)}\right]$$
其中，$p_{\theta}(z|c)$ 为 latent variable $z$ 在条件 $c$ 下的先验分布。$p_{\theta}(x|z)$ 为数据分布的对数似然，$q_{\phi}(z|x)$ 为近似的后验分布。
> CVAE 损失，是一个 overview，具体的细节在后面。

#### 重构损失

重构损失的 target 是 mel 谱而不是波形点，记为 $x_{mel}$，从 $z$ 中通过 decoder 上采样到波形域 $\hat{y}$，然后将 $\hat{y}$ 转到 mel 谱域 $\hat{x}_{mel }$（STFT），然后计算 L1 loss：
$$L_{recon}=\|x_{mel}-\hat{x}_{mel}\|_1$$
可以看成是基于拉普拉斯分布下的最大似然估计。
> 对应的是 CVAE 第一部分的 loss，但是这里的 CVAE 并不是标准的 VAE，因为 encoder 输入是 $x$，隐变量是 $z$，decoder 的输出却为 $\hat{y}$，而 $\hat{x}=g(\hat{y})$ 。
> 这一误差项可以提高 perceptual quality

#### KL 散度

首先条件 $c$ 是从文本中提取的 phoneme $c_{text}$ 和 phoneme 与 latent variable 之间的 alignment $A$。
> $A$ 的估计见下一节。

采用 linear-scale spectrogram（分辨率更高）作为 target $x_{lin}$，KL 散度为：
$$\begin{aligned}L_{kl}&=\log q_\phi(z|x_{lin})-\log p_\theta(z|c_{text},A),\\z&\sim q_\phi(z|x_{lin})=N(z;\mu_\phi(x_{lin}),\sigma_\phi(x_{lin}))\end{aligned}$$
然后使用 factorized normal distribution 参数化先验和后验分布，采用 normalizing flow $f_\theta$ 将简单分布转为复杂分布：
$$\begin{aligned}
p_{\theta}(z|c) =N(f_\theta(z);\mu_\theta(c),\sigma_\theta(c))\bigg|\det\frac{\partial f_\theta(z)}{\partial z}\bigg|,  \\
c=[c_{text},A]
\end{aligned}$$
> 这一部分是 CVAE 的第二项损失，首先 $q_\phi(z|x_{lin})=N(z;\mu_\phi(x_{lin})$ 为 CVAE encoder 的输出，也就是高斯分布的均值和方差。然后需要先验分布 $p_\theta(z|c_{text},A)$ ，而作者发现，通过采用 归一化流建模这一分布来使得分布变得更复杂，这一操作可以提高语音的表达性。
> 而流这一部分的先验分布，是通过 text encoder 输出一个 mean 和 variance 来得到的。

### Alignment 估计

#### MAS

采用 MAS 来最大化数据的似然：
$$\begin{aligned}A&=\arg\max_{\hat A}\log p(x|c_{text},\hat A)\\&=\arg\max_{\hat A}\log N(f(x);\mu(c_{text},\hat A),\sigma(c_{text},\hat A))\end{aligned}$$
但是目标函数是 ELBO，不是似然，于是重新定义 MAS 来最大化 ELBO：
$$\begin{gathered}
\arg\max_{\hat{A}}\left.\log p_\theta(x_{mel}|z)-\log\frac{q_\phi(z|x_{lin})}{p_\theta(z|c_{text},\hat{A})}\right. \\
=\arg\max_{\hat{A}}\log p_{\theta}(z|c_{text},\hat{A}) \\
=\log N(f_\theta(z);\mu_\theta(c_{text},\hat{A}),\sigma_\theta(c_{text},\hat{A})) 
\end{gathered}$$
到这可以发现，两个公式其实是差不多的，因此还是可以用原始的 MAS 来做对齐。

#### 基于文本的 Duration Predictor

设计 stochastic duration predictor，给定 phoneme，基于 duration distribution 采样得到 duration。

SDP 为 flow-based 模型，使用最大似然估计训练。但是直接优化最大似然很困难，因为 duration 是：
+ 离散整数，要用 flow-based 建模的话需要 dequantized
+ scalar，无法进行高维变换

采用 variational dequantization 和 variational data augmentation，引入 $u,v$，和 duration $d$ 的长度和维度一样。限制 $u \in [0,1)$，将 $d,v$ 在 channel 维度拼接来获得更高维的 latent representation，然后通过分布 $q_\phi(u,\nu|d,c_{text})$ 采样这两个值，然后 duration 的 对数似然的 ELBO 为：
$$\begin{gathered}
\log p_{\theta}(d|c_{text})\geq  \\
\mathbb{E}_{q_\phi(u,\nu|d,c_{text})}\bigg[\log\frac{p_\theta(d-u,\nu|c_{text})}{q_\phi(u,\nu|d,c_{text})}\bigg] 
\end{gathered}$$
采用 stop gradient 只更新这一部分模型的参数。

推理时从随机噪声采样，通过 inverse transform 得到 duration。
> 这个 flow 的先验分布是标准的高斯噪声（也就是初始的分布，对应图中的 noise）。

### 对抗训练

引入 discriminator $D$，区分 decoder $G$ 合成的和 GT waveform $y$。采用两个损失：
$$\begin{aligned}
&L_{adv}(D) =\mathbb{E}_{(y,z)}\bigg[(D(y)-1)^2+(D(G(z)))^2\bigg],  \\
&L_{adv}(G) =\mathbb{E}_z\bigg[(D(G(z))-1)^2\bigg],  \\
&L_{fm}(G) =\mathbb{E}_{(y,z)}\Big[\sum_{l=1}^T\frac1{N_l}\|D^l(y)-D^l(G(z))\|_1\Big] 
\end{aligned}$$
其中，least-squares loss function 用于对抗训练，feature-matching loss 用于训练生成器 $G$。

### 总损失

VAE 和 GAN 的损失合起来：
$$L_{vae}=L_{recon}+L_{kl}+L_{dur}+L_{adv}(G)+L_{fm}(G)$$

### 模型架构

包含：
+ posterior encoder
+ prior encoder（就是专门用在 CVAE 中用来建模先验条件分布的）
+ decoder
+ discriminator
+ stochastic duration predictor

#### POSTERIOR ENCODER

采用 WaveGlow 中的 non-causal WaveNet residual blocks。

#### PRIOR ENCODER

prior encoder  包含：
+ text encoder，处理输入 phoneme $c_{text}$，结构为  transformer encoder，最终得到 $h_{text}$，顶端还有一个 linear projection layer 得到 mean 和 variance 来构造归一化流的 prior distribution
+ normalizing flow $f_\theta$ ，几层 affine coupling layers，且确保 Jacobian determinant

#### Decoder

为 HiFi-GAN V1 generator。

#### DISCRIMINATOR

采用 multi-period discriminator，为 mixture of Markovian
window-based sub-discriminators。

#### STOCHASTIC DURATION PREDICTOR

stochastic duration predictor 从条件输入 $h_{text}$ 中估计 duration 的分布。包含几层 residual blocks with dilated 和 depth-separable convolutional layers，也用了 neural spline flows，

## 实验（略）

