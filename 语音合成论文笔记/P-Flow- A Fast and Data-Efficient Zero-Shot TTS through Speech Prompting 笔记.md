> NIPS 2023，NVIDIA，首尔大学

1. 现有的 zero-shot TTS 存在以下几个问题：
	1. 缺乏鲁棒性
	2. 采样速度慢
	3. 依赖于预训练的 codec 表征
2. 提出 P-Flow，快速的、data- efficient zero-shot  TTS，采用 speech prompt 实现说话人自适应，然后用 flow matching generative decoder 实现高质量和快速的语音合成
	1. speech-prompted text encoder 采用 speech prompts 和 text 来生成speaker- conditional 的表征
	2. flow matching generative decoder 用这些表征来合成语音

## Introduction

1. 用 codec 得到的表征，又复杂又耗计算，而且还没有可解释性；本文就采用标准的 mel 谱
2. 同时为了提高推理速度，用最近的 ODE-based 生成模型 —— Flow Matching
3. 本文贡献：
	1. 提出一种 speech prompt 方法，超过了 speaker  embedding，可以提供 in-context learning 的能力
	2. 采用 flow matching 模型提高速度和质量
	3. 可以在更少的训练数据、更少的 encoder 下实现 comparable 的性能

## 相关工作（略）

## 方法

### P-Flow

P-Flow 训练和 mask-autoencoders 类似，给定 <文本，语音> 对，定义语音的 mel 谱 为 $x$，文本为 $c$，$m^p$  为 indicator mask，用于随机 mask 3s 的音频段。定义 $(1-m^p)\cdot x$ ，其中 $p$ 表示这个变量会被替换为任意的 3s 的 prompt。

P-Flow 的训练目标为：给定 $c$ 和 $x^p$ 的条件下重构 $x$，即学习条件概率 $p(x|c,x^p)$。

引入一个 text encoder $f_{enc}$（架构用的是非自回归的 transformer ），输入 文本 $c$ 和随机段 $x^p$ 来生成 speaker- conditional 的 text representation $h_{c}=f_{enc}(x^{p},c)$，然后通过 flow- matching 生成模型将其映射为 mel 谱，最后通过 vocoder 生成语音。

![](image/Pasted%20image%2020231128211250.png)

如上如，模型输入随机段 $x^p$ 和文本 $c$ 作为 speech prompt，然后将两者投影到相同的维度作为 text encoder 的输入。

为了训练 speech-prompted text encoder，采用了一个 encoder loss 来直接最小化 text encoder representation 和 mel 谱之间的距离。这个 loss 用于减少 sampling 步数，使得 encoder 可以将 speaker 信息加到生成的 text representation 中。

由于 $h_c$ 和 $x$ 的长度不同，采用 MAS 来得到对齐 $A=MAS(h_{c},x)$，最小化对齐后的 text encoder 输出和 mel 谱之间的 L2 距离。根据对齐 $A$，确定每个文本 token 的 duration $d$，然后通过复制 encoder representation 来扩展 encoder 输出 $h_c$。这个对齐过程得到的 text encoder 输出 $h$ 和 mel 谱 $x$ 对齐。重构 loss 定义为 $L_{enc}=MSE(h,x)$。

实际用的时候发现，即使 $x^p$ 是随机选的，训练的时候模型会 collapse 为简单复制粘贴 $x^p$，于是把 $x^p$ 这段的 loss mask 掉，此时的 loss 定义为：
$$L_{enc}^p=MSE(h\cdot m^p,x\cdot m^p)$$
> 也就是不计算 mask 部分的 loss

通过最小化这个 loss，encoder 尽可能多地从 speech prompt 中提取 speaker 信息，以生成一个和给定 speech x 相似的输出 $h$，从而增强 speaker 自适应的 in-context 能力。

为了实现 TTS，采用 flow-matching 生成模型作为 decoder 建模概率分布 $p(x|c,x^p)\:=\:p(x|h)$。decoder 模型建模的是 CNF 的条件向量场 $v_t(\cdot|h)$。loss 用的是 flow-matching loss $L_{cfm}$（这里也用了 mask $m^p$）。

然后用的是类似 Glow-TTS 中的 duration predictor，输入为 text encoder 的输出。这个模块用于估计对数 duration $\log\widehat{d}$，其目标函数为，最小化与 MAS 得到的 $\log d$ 之间的 MSE loss。

总的训练损失为 $L=L_{enc}^{p}+L_{cfm}^{p}+L_{dur}$。推理的时候用从参考样本中的随机段作为 prompt，还有文本作为输入。然后用 duration predictor 得到的 $\hat{d}$，拓展之后通过 flow-matching 得到 mel 谱。

### Flow Matching Decoder

采用 flow matching 来建模条件分布 $p(x|h)$。flow matching 用于拟合数据发布 $p_1(x)$ 和简单分布 $p_0(x)$ 之间的时间相关的概率路径。采用 Conditional Flow Matching，其路径更简单直接，从而可以得到更少的采样次数。为了简单，下面分析中忽略条件 $h$。

采用下述 ODE 定义 flow $\phi:[0,1]\times\mathbb{R}^d\to\mathbb{R}^d$ 为两个密度函数之间的映射：
$$\frac d{dt}\phi_t(x)=v_t(\phi_t(x));\quad\phi_0(x)=x$$

其中，$v_t(x)$ 是时间相关的向量场，表示 probability flow 沿着时间维度的轨迹。$v_t(x)$ 是可学习的，记为 $v_t(\phi(x);\theta)$。推理采样时，从采样分布 $p_0$ 中采样作为初始条件，然后求解 ODE。

实际使用时，很难确定 marginal flow $\phi_t(x)$ ，因此将其定义为多个条件 flow 的边缘化：
$$\phi_{t,x_1}(x)=\sigma_t(x_1)x+\mu_t(x_1)$$

其中，$\sigma_t(x_1)$ 和 $\mu_t(x_1)$ 是以时间为条件的仿射变换，用于参数化 Gaussian 分布 $p_1$ 和 $p_0$ 之间的转换。$q(x_1)$ 是实际的数据分布。$p_1$ 是 $q$ 的高斯混合近似，通过给每个样本加入小量的白噪声来实现（$\sigma_{\min}$ 设置为 0.01）。而且可以指定轨迹，而不受 SDE 公式中的随机性的影响。

可以采样简单的线性轨迹，得到如下参数化的 $\phi_t$：
$$\mu_t(x)=tx_1,\:\sigma_t(x)=1-(1-\sigma_{\min})t$$

此时采用 conditional flow matching 目标函数训练 vector field：
$$L_{CFM}(\theta)=\mathbb{E}_{t\thicksim U[0,1],x_1\thicksim q(x_1),x_0\thicksim p(x_0)}\|v_t(\phi_{t,x_1}(x_0);\theta)-\frac d{dt}\phi_{t,x_1}(x_0)\|^2$$

从而最终的 CFM 目标函数为：
$$L_{CFM}(\theta)=\mathbb{E}_{t,q(x_1),p(x_0)}\|\upsilon_t(\phi_{t,x_1}(x_0);\theta)-(x_1-(1-\sigma_{\min})x_0)\|^2$$

flow matching decoder 建模 $p(x|h)$，因为 $h$ 是 text encoder 的输出，来自 $x^p\in x$，所以需要 mask 掉 $x^p$ 部分的 loss。此时 $v_t(x_t;\theta)$ 参数化为 $\hat{v}_{\theta}(x_t,h,t)$，这里 $t$ 用 continuous sinusoidal embedding 表示。得到 mask CFM 目标函数：
$$L_{CFM}^p(\theta)=\mathbb{E}_{t,q(x_1),p(x_0)}\|m^p\cdot(\hat{v}_\theta(\phi_{t,x_1}(x_0),h,t)-(x_1-(1-\sigma_{\min})x_0))\|^2$$

CFM loss 边缘化 conditional vector field 得到 marginal vector field 用于采样。本文作者发现，conditional flow matching 可以得到足够简单的轨迹，因此在推理时用 10 步 Euler 方法求解 ODE。N Euler steps 的采样过程的递推如下：
$$x_0\sim\mathcal{N}(0,I);\quad x_{t+\frac1N}=x_t+\frac1N\hat{v}_\theta(x_t,h,t)$$

采用 classifier-free guidance 来进一步增强 pronunciation clarity。本文采用 GLIDE 中的方法，通过减去空文本序列的轨迹来放大 text-conditional sampling trajectory。这里采用类似的公式，通过计算平均特征向量 $h$ 来指导采样轨迹，记为 $\bar{h}$。$\bar{h}$ 通过沿时间轴对 $h$ 进行平均得到固定大小的向量，然后沿时间维度重复。$\gamma$ 为 guidance scale。此时  guidance-amplified Euler 公式如下：
$$x_{t+\frac1N}=x_t+\frac1N(\hat{v}_\theta(x_t,h,t)+\gamma(\hat{v}_\theta(x_t,h,t)-\hat{v}_\theta(x_t,\bar{h},t))$$
****
### 模型细节

模型包括 3 个主要部分：
+ 基于 prompt 的 text encoder
+ duration predictor 
+ 基于 Wavenet 的 flow matching decoder

实验表明，即使 text encoder 只有 3M 参数，也能实现很好的 zero-shot。

## 实验

模型在单张 NVIDIA A100 GPU 上训练 800K 次，batch size 为 64。采用 AdamW 优化器，学习率为 0.0001。G2P 模型将文本预处理为 IPA。推理时，flow matching decoder 用 10 步 Euler 方法，guidance scale 为 1。mel-spectrogram 转 waveform 用预训练的 universal Hifi-GAN。

窗口大小为 1024，hop length 为 256。3s 的 mel-spectrogram 作为 speech prompt，长度为 $\lceil \frac{3\cdot 22050}{256} \rceil=259$。

数据集为 LibriTTS，包含 580 小时的数据，来自 2456 个说话人。训练时只用超过 3s 的数据，得到 256 小时的子集。评估时用 LibriSpeech test-clean，保证和训练数据没有重叠。所有数据集重采样为 22kHz。

评估时，和两个 zero-shot speaker-adaptive TTS 模型 YourTTS 和 VALL-E 进行比较。

从 LibriSpeech test-clean 数据集中提取 4-10s 的样本，共 2.2 小时。对于每个配对数据 $(x_i,c_i)$，从另一个样本 $x_j$ 中提取 3s 的参考 speech $x_{p_j}$，生成文本 $c_i$ 的合成 speech $x_{gen}$。