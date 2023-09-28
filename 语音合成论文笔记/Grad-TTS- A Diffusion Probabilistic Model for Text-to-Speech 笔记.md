> ICML 2021 华为诺亚实验室

1. 提出 Grad-TTS，使用 score-based decoder 逐步从噪声产生 mel 谱，通过 Monotonic Alignment Search 对齐文本
2. 使用 SDE 框架将传统的 DPM 模型推广到从不同参数的噪声中重构数据，且可以在质量和推理速度之间进行 trade off
3. MOS 值可以和 SOTA 的方法相竞争

> 简单来说，这篇论文就是把 mel 谱 的生成过程用 diffusion 来建模，但是：
> 	在一般的 TTS 中，通常是使用 text encoder output representation 作为 mel decoder 的输入
> 	在一般的 diffusion 模型中，通常是从噪声开始生成样本（先验分布为噪声）
> 因此，这里就相当于把 text encoder output representation 作为 diffusion 模型的先验来进行 diffusion 过程，由于此时的先验分布不是高斯噪声，所以作者推导了任意高斯分布 $\mathcal{N}(\mu, \Sigma)$ 下的 diffusion 过程。

## Introduction

1. TTS 系统包含两个部分：
	1. feature generator 将文本转为时频域的声学特征
	2. vocoder 将这些特征转换为波形
2. 有很多并行的非自回归 vocoder，如基于 Normalizing Flows 的 Parallel WaveNet、Wave- Glow，也有基于 GAN 的如 Parallel Wave- GAN、HiFi-GAN
3. 对于 feature generator，Tacotron2 和 Transformer-TTS 可以实现高质量的合成，但是计算不高效、也有一些发音的问题；后面 FastSpeech、Parallel Tacotron 通过采用非自回归方式逐步改进推理速度和发音问题，但是又需要 teacher model 来得到对齐；最后，Non-Attentive Tacotron framework 通过采用 VAE 来隐式学习 duration
4. WaveGrad 和 DiffWave 这两种基于 DPM 的 vocoder 已经可以产生很好的语音且接近强自回归的 baseline；但是没有基于 DPM 的 feature generator
5. 本文引入 Grad-TTS，基于 score-based decoder 的 feature generator，encoder 的输出通过 decoder，将噪声转换为 mel 谱，且模型可以显式地在 质量和推理速度之间进行 trade off，而且只要几十个迭代就可以生成高质量的输出，并且还可以 E2E 地训练

## DPM 

以 SDE 的方式定义 DPM 模型：
$$d X_t=b\left(X_t, t\right) d t+a\left(X_t, t\right) d W_t$$
其中，$W_t$ 为标准的布朗运动，系数 $b,a$ 分别称为漂移系数和扩散系数。

且作者求解的问题是，无限时间下的 forward diffusion，而且是将任意的分布最终转化为 $\mathcal{N}(\mu, \Sigma)$ 而非高斯噪声 $\mathcal{N}(0, I)$。

下面给出推广后的 forward diffusion、 reverse diffusion 和 损失函数的定义。

### forward diffusion

给定无限时间，forward diffusion 需要将任意分布转换为高斯分布。

如果 $n$ 维的随机过程 $X_t$ 满足以下 SDE：
$$d X_t=\frac{1}{2} \Sigma^{-1}\left(\mu-X_t\right) \beta_t d t+\sqrt{\beta_t} d W_t, \quad t \in[0, T]$$
其中，$\beta_t$ 为 noise schedule，且向量 $\mu$，对角矩阵 $\Sigma$ 中的元素都是正的，那么其解为：
$$\begin{aligned}
X_t & =\left(I-e^{-\frac{1}{2} \Sigma^{-1} \int_0^t \beta_s d s}\right) \mu+e^{-\frac{1}{2} \Sigma^{-1} \int_0^t \beta_s d s} X_0 +\int_0^t \sqrt{\beta_s} e^{-\frac{1}{2} \Sigma^{-1} \int_s^t \beta_u d u} d W_s
\end{aligned}$$
> 对角矩阵的 exponential 是 其中每个元素的 exponential。

令：
$$\begin{aligned}
\rho\left(X_0, \Sigma, \mu, t\right) & =\left(I-e^{-\frac{1}{2} \Sigma^{-1} \int_0^t \beta_s d s}\right) \mu \\
& +e^{-\frac{1}{2} \Sigma^{-1} \int_0^t \beta_s d s} X_0
\end{aligned}$$
且：
$$\lambda(\Sigma, t)=\Sigma\left(I-e^{-\Sigma^{-1} \int_0^t \beta_s d s}\right)$$
则根据伊藤积分性质，给定 $X_0$ 后 $X_t$ 的条件分布为高斯分布：
$$\operatorname{Law}\left(X_t \mid X_0\right)=\mathcal{N}\left(\rho\left(X_0, \Sigma, \mu, t\right), \lambda(\Sigma, t)\right)$$
这意味着，在无限时间下，对于任意的 noise schedule $\beta_t$ 有 $\lim _{t \rightarrow \infty} e^{-\int_0^t \beta_s d s}=0$ ，从而：
$$X_t \mid X_0 \stackrel{d}{\rightarrow} \mathcal{N}(\mu, \Sigma)$$
所以随机变量 $X_t$ 依分布收敛到 $\mathcal{N}(\mu, \Sigma)$，且和 $X_0$ 独立。这正好是需要的特性：满足前述 SDE 方程的 forward diffusion 可以将任意数据分布 $\operatorname{Law}(X_0)$ 转为高斯分布 $\mathcal{N}(\mu, \Sigma)$。

### reverse diffusion

早期的 DPM 的 reverse 过程都训练用于近似 forward diffusion 的路径。但在 SDE 下，reverse 过程有显式解，在前文的例子中，有对应于 reverse diffusion 的 SDE：
$$\begin{aligned}
d X_t= & \left(\frac{1}{2} \Sigma^{-1}\left(\mu-X_t\right)-\nabla \log p_t\left(X_t\right)\right) \beta_t d t \\
& +\sqrt{\beta_t} d \widetilde{W_t}, \quad t \in[0, T]
\end{aligned}$$
其中，$\widetilde{W_t}$ reverse-time 布朗运动，$p_t$ 为随机变量 $X_t$ 的 PDF，此 SDE 通过从 $X_T$ 开始反向求解。

同时还有一个对应的 ODE：
$$d X_t=\frac{1}{2}\left(\Sigma^{-1}\left(\mu-X_t\right)-\nabla \log p_t\left(X_t\right)\right) \beta_t d t$$

因此，如果有神经网络 $s_\theta\left(X_t, t\right)$ 来估计噪声数据的对数密度的梯度 $\nabla \log p_t\left(X_t\right)$，则可以通过从 $\mathcal{N}(\mu, \Sigma)$ 中采样 $X_t$ 来建模数据分布 $\operatorname{Law}(X_0)$，且通过对应的 reverse SDE 或 ODE 来数值求解。

### 损失函数

如果从分布 $\mathcal{N}(0, \lambda(\Sigma, t))$ 中首先采样 $\epsilon_t$，然后：
$$X_t=\rho\left(X_0, \Sigma, \mu, t\right)+\epsilon_t$$
那么在 $X_t$ 下对数密度的梯度为：
$$\nabla \log p_{0 t}\left(X_t \mid X_0\right)=-\lambda(\Sigma, t)^{-1} \epsilon_t$$
此时损失函数为：
$$\mathcal{L}_t\left(X_0\right)=\mathbb{E}_{\epsilon_t}\left[\left\|s_\theta\left(X_t, t\right)+\lambda(\Sigma, t)^{-1} \epsilon_t\right\|_2^2\right]$$
其中的 $\epsilon_t$ 是采样得到的值，$X_t$ 是根据上述公式计算得到的。

## Grad-TTS

提出的 acoustic feature generator  包含三个部分：
+ encoder
+ duration predictor
+ decoder

整个架构如图：
![](image/Pasted%20image%2020230825151440.png)

Grad-TTS 和 Glow-TTS 结构很相似，不同点在于 deocder 用的是 diffusion。

### 推理

长为 $L$ 的输入文本序列 $x_{1:L}$ （通常是汉字或 phoneme），目标是生成 mel 谱 $y_{1:F}$ 其中 $F$ 为声学特征的数量。

encoder 首先将输入 $x_{1:L}$ 转化为特征 $\tilde{\mu}_{1:L}$，然后 duration predictor 产生 $\tilde{\mu}_{1:L}$ 和 $\mu_{1:F}$ 之间的 hard monotonic alignment $A$。函数 $A$ 是在 $[1, F] \cap \mathbb{N}$ 和 $[1, L] \cap \mathbb{N}$ 之间的单调投射映射，对任意的 $j\in[1,F]$，令 $\mu_j=\tilde{\mu}_{A(j)}$。简单来说，duration predictor 的作用就是预测 text input 中的每个元素的持续帧数。可以通过对预测的 duration 乘以一个因子来控制合成的时间（语速）。输出序列 $\mu_{1:F}$ 然后通过 DPM 构成的 decoder，神经网络 $s_\theta\left(X_t, \mu, t\right)$ 定义了一个 ODE：
$$d X_t=\frac{1}{2}\left(\mu-X_t-s_\theta\left(X_t, \mu, t\right)\right) \beta_t d t$$
然后通过一阶欧拉法求解。序列 $\mu$ 被用于定义最终的分布 $X_T \sim \mathcal{N}(\mu, I)$，且 $\beta_t$ 和 $T$ 是预先被定义好的。step size $h$ 为采样的超参数，用于在 质量和速度之间进行 trade off。

一些注解：
+ 选择 ODE 而不是 SDE：实践发现 ODE 效果更好
+ 选择 $\Sigma=I$ 来简化生成过程
+ 把 $\mu$ 作为输入可以告诉网络一些额外的信息，从而提高性能

还发现，引入一个温度系数 $\tau$，从 $\mathcal{N}(\mu, \tau^{-1}\Sigma)$ 中采样 $X_t$ 效果会变好。

### 训练

模型训练的目标是最小化 encoder 输出 $\mu$ 和 真实的 mel 谱 $y$ 之间的距离。将 encoder 的输出 $\tilde{\mu}$ 视为正态分布 $\mathcal{N}(\tilde{\mu},I)$，从而得到 encoder 的 NLL loss 为：
$$\mathcal{L}_{enc}=-\sum_{j=1}^F\log\varphi(y_j;\tilde{\mu}_{A(j)},I),$$
其中，$\varphi$ 是 $\mathcal{N}(\tilde{\mu},I)$ 的 PDF。
> 原则上，没有 ecoder loss 也是可以训练 grad-tts 的；实际发现不用的话会不好学习 alignment

由于需要同时学习 encoder 的参数和 alignment，一起学可能会很困难，因此采用 [Glow-TTS- A Generative Flow for Text-to-Speech via Monotonic Alignment Search 笔记](Glow-TTS-%20A%20Generative%20Flow%20for%20Text-to-Speech%20via%20Monotonic%20Alignment%20Search%20笔记.md) 中的方法迭代学习。然后在前几步就采用 MAS。

同样也需要一个 duration predictor，采用的也是 MSE 训练：
$$\begin{gathered}
d_{i}= \begin{aligned}\log\sum_{j=1}^{F}\mathbb{I}_{\{A^*(j)=i\}},\quad i=1,..,L,\end{aligned} \\
\mathcal{L}_{dp}=MSE(DP(sg[\tilde{\mu}]),d), 
\end{gathered}$$
同样也用了 stop gradient。

DPM 模型就直接用前面推的公式，另 $\Sigma=I$，从而：
$$\lambda_t=1-e^{-\int_0^t\beta_sds}$$
总的 diffusion  loss 是：
$$\mathcal{L}_{diff}=\mathbb{E}_{X_0,t}\left[\lambda_t\mathbb{E}_{\xi_t}\left[\left\|s_\theta(X_t,\mu,t)+\frac{\xi_t}{\sqrt{\lambda_t}}\right\|_2^2\right]\right]$$
其中，$X_0$ 表示 目标 mel 谱 $y$，$t$ 从 $[0,T]$ 的均匀分布中采样，$\xi_t$ 来自分布 $\mathcal{N}(0,I)$。加噪过程为：
$$X_t=\rho(X_0,I,\mu,t)+\sqrt{\lambda_t}\xi_t$$
采用的是加权损失，权重为 $\lambda_t$。

### 模型架构

对于 encoder 和 duration predictor，采用 Glow-TTS 相同的架构。

decoder 网络用的是 U-Net 架构，但是用了更少的 channel 和 feature map 来减少参数。

$\mu$ 通过和 $X_t$ 拼接得到一个额外的 channel 来引入。

## 实验
