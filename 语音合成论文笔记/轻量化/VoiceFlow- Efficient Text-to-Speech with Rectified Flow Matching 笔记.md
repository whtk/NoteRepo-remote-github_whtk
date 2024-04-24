> ICASSP 2024，上交、俞凯

1. diffusion 复杂的采样过程降低了合成效率
2. 提出 VoiceFlow，采用 rectified flow matching 实现的 acoustic model，可以在有限的采样步骤下实现高质量合成
    1. 将生成 mel-spectrograms 的过程建模成一个 ODE，条件是输入的文本，然后估计其 vector field
    2. rectified flow 可以提高合成效率
3. 在单说话人和多说话人数据集上，VoiceFlow 的合成质量优于 diffusion 模型

## Introduction

1. 通过求解 SDE 或 ODE，diffusion 模型可以稳定训练，生成高质量的样本
2. diffusion 的缺点是效率低，需要大量采样步骤，导致推理时延迟大
3. flow matching 生成模型：
    1. flow matching 模型直接建模 ODE 隐含的 vector field
    2. 通过训练 flow matching 模型，可以得到更简单的公式和更好的质量
    3. rectified flow 进一步简化了 ODE 轨迹，提高了采样效率
4. rectified flow matching 可以在很少的采样步骤下获得高质量的样本
5. 提出在 TTS acoustic model 中使用 rectified flow matching，构建 ODE 来在噪声分布和 mel-spectrogram 之间 flow，同时用 phones 和 duration 作为条件：
    1. 学习估计 vector field
    2. 通过 flow rectification 过程，从训练好的 flow matching 模型中生成样本再训练自己，从而在更少的步骤下生成的 mel-spectrograms


## Flow Matching 和 Rectified Flow

### Flow Matching 生成模型

给定数据分布为 $p_1(\boldsymbol{x}_1)$，先验分布为 $p_0(\boldsymbol{x}_0)$。生成模型将 $\boldsymbol{x}_0 \sim p_0(\boldsymbol{x}_0)$ 映射到数据 $\boldsymbol{x}_1$。
> 如，diffusion 模型通过构造一个特殊的 SDE，然后估计由它产生的概率路径 $p_t(\boldsymbol{x}_t)$ 的 score function。通过求解 SDE 或概率流 ODE 及其概率路径来实现采样。

flow matching 直接建模概率路径 $p_t(\boldsymbol{x}_t)$。考虑任意 ODE：
$$\mathrm{d}\boldsymbol{x}_{t}=\boldsymbol{v}_{t}(\boldsymbol{x}_{t})\mathrm{d}t$$
其中 $\boldsymbol{v}_t(\cdot)$ 为 vector field，$t \in [0, 1]$。这个 ODE 与概率路径 $p_t(\boldsymbol{x}_t)$ 相关，满足连续方程 $\frac{\mathrm{d}}{\mathrm{d}t}\log p_t(\boldsymbol{x})+\mathrm{div}(p_t(\boldsymbol{x})\boldsymbol{v}_t(\boldsymbol{x}))\:=\:0$ 。如果神经网络可以准确估计 $\boldsymbol{v}_t(\cdot)$，那么通过数值求解上式中的 ODE 就足以生成真实数据。

然而需要实例化 vector field。给定数据样本为 $\boldsymbol{x}_1$，假设概率路径为 $p_t(\boldsymbol{x} | \boldsymbol{x}_1)$，边界条件为 $p_{t=0}(\boldsymbol{x} | \boldsymbol{x}_1) = p_0(\boldsymbol{x})$ 和 $p_{t=1}(\boldsymbol{x} | \boldsymbol{x}_1) = N(\boldsymbol{x} | \boldsymbol{x}_1, \sigma^2I)$，其中 $\sigma$ 足够小。根据连续方程，会有一个相关的 vector field $\boldsymbol{v}_t(\boldsymbol{x} | \boldsymbol{x}_1)$。且通过神经网络 $u_\theta$ 估计条件 vector field 等价于估计无条件 vector field，即：
$$\begin{aligned}&\min_\theta\mathbb{E}_{t,p_t(\boldsymbol{x})}\|\boldsymbol{u}_\theta(\boldsymbol{x},t)-\boldsymbol{v}_t(\boldsymbol{x})\|^2\\&\equiv\min_\theta\mathbb{E}_{t,p_1(\boldsymbol{x}_1),p_t(\boldsymbol{x}|\boldsymbol{x}_1)}\|\boldsymbol{u}_\theta(\boldsymbol{x},t)-\boldsymbol{v}_t(\boldsymbol{x}\mid\boldsymbol{x}_1)\|^2.\end{aligned}$$

通过设计简单的条件概率路径 $p_t(\boldsymbol{x} | \boldsymbol{x}_1)$ 和对应的 $\boldsymbol{v}_t(\boldsymbol{x} | \boldsymbol{x}_1)$，可以从 $p_t(\boldsymbol{x} | \boldsymbol{x}_1)$ 中采样并最小化上式。
> 例如，使用高斯路径 $p_t(\boldsymbol{x} | \boldsymbol{x}_1) = N(\boldsymbol{x} | \mu_t(\boldsymbol{x}_1), \sigma_t(\boldsymbol{x}_1)^2I)$ 和线性 vector field $\boldsymbol{v}_t(\boldsymbol{x} | \boldsymbol{x}_1) = \sigma_t'(\boldsymbol{x}_1)(\boldsymbol{x} - \mu_t(\boldsymbol{x}_1)) + \mu_t'(\boldsymbol{x}_1)$。

实际上，任何条件 $z$ 对 $p_t(\boldsymbol{x} | z)$ 都会得到类似于上式的优化目标。于是可以在噪声样本 $\boldsymbol{x}_0$ 上进行额外的 conditioning，形成概率路径 $p_t(\boldsymbol{x} | \boldsymbol{x}_0, \boldsymbol{x}_1) = N(\boldsymbol{x} | t\boldsymbol{x}_1 + (1 - t)\boldsymbol{x}_0, \sigma^2I)$。因此条件 vector field 变为 $\boldsymbol{v}_t(\boldsymbol{x} | \boldsymbol{x}_0, \boldsymbol{x}_1) = \boldsymbol{x}_1 - \boldsymbol{x}_0$，是一条指向 $\boldsymbol{x}_1$ 的直线。在这种形式中，训练生成模型只需要以下步骤：
1. 从数据中采样 $\boldsymbol{x}_1$，从任意噪声分布 $p_0(\boldsymbol{x}_0)$ 中采样 $\boldsymbol{x}_0$；
2. 在 $t \in [0, 1]$ 上采样一个时间，然后 $\boldsymbol{x}_t \sim N(t\boldsymbol{x}_1 + (1 - t)\boldsymbol{x}_0, \sigma^2I)$；
3. 在损失 $\|\boldsymbol{u}_\theta(\boldsymbol{x}, t) - (\boldsymbol{x}_1 - \boldsymbol{x}_0)\|^2$ 上进行梯度下降
即所谓的 conditional flow matching，其优于 diffusion 的模型，并与最优传输理论相关。

### 基于 Rectified Flow 的采样效率提升

rectified flow 的条件也是 $\boldsymbol{x}_1$ 和 $\boldsymbol{x}_0$。假设 flow matching 模型通过 ODE 从噪声 $\boldsymbol{x}_0$ 生成数据 $\hat{\boldsymbol{x}}_1$。换句话说，$\boldsymbol{x}_0$ 和 $\hat{\boldsymbol{x}}_1$ 是 ODE 轨迹的起点和终点。然后，这个 flow matching 模型再次训练，但条件是 $\boldsymbol{v}_t(\boldsymbol{x} | \boldsymbol{x}_0, {\boldsymbol{x}}_1)$ 和 $p_t(\boldsymbol{x} | \boldsymbol{x}_0, {\boldsymbol{x}}_1)$，而不是独立采样的 $\boldsymbol{x}_0, \boldsymbol{x}_1$。此 flow rectification 步骤可以迭代多次，用递归 $z^{k+1}_0, z^{k+1}_0 = {FM}(z^{k}_0, z^{k}_1)$ 表示，其中 FM 是 flow matching 模型，$(z_0^0, z_1^0) = (\boldsymbol{x}_0, \boldsymbol{x}_1)$ 是独立采样的噪声和数据样本。

rectified flow 使 flow matching 模型的采样轨迹更直。因为 ODE 轨迹在求解时不能相交，所以轨迹很可能不会像训练时的条件 vector field 那样直。然而，通过在相同轨迹的端点上再次训练 flow matching 模型，模型学会找到连接这些噪声和数据的更短路径。这种直线倾向在理论上是有保证的。通过纠正轨迹，flow matching 模型将能够更高效地采样数据，而不需要进行更多的 ODE 模拟步骤。

## VoiceFlow

### 基于 Flow Matching 的 acoustic model

为了在 TTS 中使用 flow matching 模型，将其作为一个非自回归条件生成问题。其中 mel-spectrogram $\boldsymbol{x}_1 \in \mathbb{R}^d$ 为目标数据，然后从标准高斯分布 $N(0, I)$ 中采样噪声 $\boldsymbol{x}_0 \in \mathbb{R}^d$。
用的是 [DiffSinger- Singing Voice Synthesis via Shallow Diffusion Mechanism 笔记](../歌声合成/DiffSinger-%20Singing%20Voice%20Synthesis%20via%20Shallow%20Diffusion%20Mechanism%20笔记.md) 中的显示 duration 模块，将重复后的 latent phone 表示记为 $\boldsymbol{y}$，其中每个 phone 的 latent embedding 根据其时长重复。$\boldsymbol{y}$ 为生成过程的条件，假设 $\boldsymbol{v}_t(\boldsymbol{x}_t | \boldsymbol{y}) \in \mathbb{R}^d$ 是 ODE $d\boldsymbol{x}_t = \boldsymbol{v}_t(\boldsymbol{x}_t | \boldsymbol{y})\mathrm{d}t$ 的 underlying vector field。假设这个 ODE 连接噪声分布 $p_0(\boldsymbol{x}_0 | \boldsymbol{y}) = N(0, I)$ 和给定文本的 mel 分布 $p_1(\boldsymbol{x}_1 | \boldsymbol{y}) = p_{\text{mel}}(\boldsymbol{x}_1 | \boldsymbol{y})$。目标是准确估计给定条件 $\boldsymbol{y}$ 的 vector field $\boldsymbol{v}_t$，然后就可以通过从 $t = 0$ 到 $t = 1$ 求解此 ODE 来生成 mel-spectrogram。

选择使用噪声样本 $\boldsymbol{x}_0$ 和数据样本 $\boldsymbol{x}_1$ 来构建条件概率路径：
$$p_t(\boldsymbol{x}\mid\boldsymbol{x}_0,\boldsymbol{x}_1,\boldsymbol{y})=\mathcal{N}(\boldsymbol{x}\mid t\boldsymbol{x}_1+(1-t)\boldsymbol{x}_0,\sigma^2\boldsymbol{I})$$
其中 $\sigma$ 是一个足够小的常数。路径的端点分别是 $t = 0$ 时的 $N(\boldsymbol{x}_0, \sigma^2\boldsymbol{I})$ 和 $t = 1$ 时的 $N(\boldsymbol{x}_1, \sigma^2\boldsymbol{I})$。这些路径还确定了一个概率路径 $p_t(\boldsymbol{x} | \boldsymbol{y})$，其边界近似于噪声分布 $p_0(\boldsymbol{x}_0 | \boldsymbol{y})$ 和 mel 分布 $p_1(\boldsymbol{x}_1 | \boldsymbol{y})$。

上式指定了一族沿直线移动的高斯分布。相关的 vector field 可以简单地为 $\boldsymbol{v}_t(\boldsymbol{x} | \boldsymbol{x}_0, \boldsymbol{x}_1, \boldsymbol{y}) = \boldsymbol{x}_1 - \boldsymbol{x}_0$，也是一条恒定的直线。

然后使用神经网络 $u_\theta$ 估计 vector field。目标是：
$$\min_\theta\mathbb{E}_{t,p_1(\boldsymbol{x}_1|\boldsymbol{y}),p_0(\boldsymbol{x}_0|\boldsymbol{y}),p_t(\boldsymbol{x}_t|\boldsymbol{x}_0,\boldsymbol{x}_1,\boldsymbol{y})}\|\boldsymbol{u}_\theta(\boldsymbol{x}_t,\boldsymbol{y},t)-(\boldsymbol{x}_1-\boldsymbol{x}_0)\|^2$$

对应的 flow matching 损失记为 $L_{\text{FM}}$。训练 VoiceFlow 的总损失函数为 $L = L_{\text{FM}} + L_{\text{dur}}$，其中 $L_{\text{dur}}$ 是 duration predictor 的 MSE loss。

VoiceFlow 的整个 acoustic model 包括 text encoder、duration predictor、duration adaptor 和 vector field estimator，如图：
![](image/Pasted%20image%2020240422215445.png)

+ text encoder 将输入 phones 转换为 latent space，然后预测每个 phone 的 duration 并输入到 duration adaptor
+ 重复的 frame-level 序列 $\boldsymbol{y}$ 作为条件输入到 vector field estimator
+ vector field estimator 使用 [Grad-TTS- A Diffusion Probabilistic Model for Text-to-Speech 笔记](../Grad-TTS-%20A%20Diffusion%20Probabilistic%20Model%20for%20Text-to-Speech%20笔记.md) 中相同的 U-Net 架构，将条件 $\boldsymbol{y}$ 和采样的 $\boldsymbol{x}_t$ 拼接后输入到 estimator，时间 $t$ 通过一些全连接层后加到每次 residual block 的 hidden variable 中

在多说话人条件下，条件将变为文本 $\boldsymbol{y}$ 和 speaker representation $\boldsymbol{s}$。

### 采样和 Flow Rectification Step

vector field estimator $u_\theta$ 可以以期望逼近 $\boldsymbol{v}_t$。然后，ODE $d\boldsymbol{x}_t = \boldsymbol{u}_\theta(\boldsymbol{x}_t, \boldsymbol{y}, t)\mathrm{d}t$ 可以离散化，用于采样给定文本 $\boldsymbol{y}$ 的合成 mel-spectrogram $\boldsymbol{x}_1$。可以直接使用 Euler、Runge-Kutta、Dormand-Prince 等现成的 ODE solver 进行采样。以 Euler 方法为例，每个采样步骤为：
$$\boldsymbol{\hat{x}}_{{\frac{k+1}{N}}}=\boldsymbol{\hat{x}}_{{\frac{k}{N}}}+\frac{1}{N}\boldsymbol{u}_{\theta}\left(\boldsymbol{\hat{x}}_{{\frac{k}{N}}},\boldsymbol{y},\frac{k}{N}\right),k=0,1,...,N-1$$
其中 $\boldsymbol{\hat{x}}_0 \sim p_0(\boldsymbol{x}_0 | \boldsymbol{y})$ 为初始点，$\boldsymbol{\hat{x}}_1$ 为生成的样本。不管是哪种离散化方法，solver 都会产生一系列沿着 ODE 轨迹的样本 $\{\boldsymbol{\hat{x}}_{k/N}\}$，逐渐逼近一个真实的 spectrogram。

然后用 rectified flow 进一步拉直 ODE 轨迹。对于训练集中的每个 utterance，抽取一个噪声样本 $\boldsymbol{x}^\prime_0$ 进行 ODE solver，得到给定文本 $\boldsymbol{y}$ 的 $\boldsymbol{\hat{x}}_1$。然后将样本对 $(\boldsymbol{x}^\prime_0, \boldsymbol{\hat{x}}_1)$ 再次输入到 VoiceFlow 中，用于纠正 vector field estimator。在这个 flow rectification 步骤中，新的训练准则为：
$$\min_\theta\mathbb{E}_{t,p(\boldsymbol{x}_0^{\prime},\hat{\boldsymbol{x}}_1|\boldsymbol{y}),p_t(\boldsymbol{x}_t|\boldsymbol{x}_0^{\prime},\hat{\boldsymbol{x}}_1,\boldsymbol{y})}\|\boldsymbol{u}_\theta(\boldsymbol{x}_t,\boldsymbol{y},t)-(\boldsymbol{\hat{x}}_1-\boldsymbol{x}_0^{\prime})\|^2$$

与前面式子的唯一区别是使用配对的 $(\boldsymbol{x}^\prime_0, \hat{\boldsymbol{x}}_1)$ 而不是独立采样。上式中每个 spectrogram 样本 $\hat{\boldsymbol{x}}_1$ 都与同一轨迹中的一个噪声样本相关联。这样，vector field estimator 被要求找到一个更直接的采样轨迹连接 $(\boldsymbol{x}^\prime_0, \hat{\boldsymbol{x}}_1)$，从而在很大程度上提高了采样效率。
> 在为 rectified flow 生成数据时，模型输入的是真实的 duration。从而确保模型输入更自然的语音，降低了不准确的 duration 预测导致模型性能下降的风险。```
sudo apt-get install -y ffmpeg
```

下面的算法总结了训练 VoiceFlow 的整个过程，包括 flow rectification：
![](image/Pasted%20image%2020240422220416.png)

## 实验（略）
