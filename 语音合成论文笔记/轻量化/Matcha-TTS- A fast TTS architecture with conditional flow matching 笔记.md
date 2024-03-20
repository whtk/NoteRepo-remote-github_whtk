> ICASSP 2024，KTH Royal Institute of Technology

1. 提出 Matcha-TTS，快速的 TTS 声学模型，采用 optimal-transport conditional flow matching（OT-CFM）训练
2. Matcha-TTS 采用 ODE-based 解码器，能够在更少的合成步骤中产生高质量输出
3. Matcha-TTS 是 probabilistic 且非自回归的，可以在没有外部 alignments 的情况下从头学
4. Matcha-TTS 占用内存小，MOS 效果好

## Introduction

1. DPM 的采样速度慢
2. 提出 Matcha-TTS，基于 CNFs 的 TTS 声学模型，两个创新点：
	1. 在 decoder 中采用 1D CNNs 和 Transformer 来减少内存消耗，提高合成速度
    2. 采用 OT-CFM 训练模型，比传统 CNFs 和 score-matching probability flow ODEs 更简单，能够在更少的 step 中实现准确合成
3. 实验结果表明，Matcha-TTS 在速度和合成质量之间实现 trade-off，且不需要外部 aligner

## 背景

1. LinearSpeech 使用 RoPE，相比于 relative embeddings 有着计算和内存优势
2. 提出的 Matcha-TTS 在 encoder 中使用了 RoPE，是第一个使用 RoPE 的 SDE 或 ODE-based TTS 方法
3. 现代 TTS 架构的 decoder 网络设计不同
    1. Glow-TTS 和 OverFlow 使用 dilated 1D-convolutions
    2. DPM-based 方法使用 1D convolutions
    3. Grad-TTS 使用 U-Net with 2D-convolutions
    4. FastSpeech 1 和 2 使用 (1D) Transformers
    5. Matcha-TTS 使用 1D U-Net
4. 一些 TTS 系统依赖于外部 alignments，大多数系统能够同时学习说话和 alignments，但是需要鼓励或强制单调 alignments
    1. Glow-TTS 和 VITS 使用 monotonic alignment search（MAS）
    2. Grad-TTS 使用 MAS-based 机制
    3. Matcha-TTS 使用相同的方法进行 alignments 和 duration modelling，且在所有 decoder feedforward layers 使用 snake beta activations

### Flow matching

1. 有些高质量的 TTS 系统使用 DPMs 或 discrete-time normalising flows，很少用 continuous-time flows
2. Voicebox 和 Matcha-TTS 都是基于 CFM ，区别在于：
    1. Voicebox 是 TTS、denoising 和 text-guided acoustic infilling 的组合，而 Matcha-TTS 是纯 TTS 模型
    2. Voicebox 使用 convolutional positional encoding，而 Matcha-TTS 使用 RoPE
    3. Voicebox 消耗 330M 参数，是 Matcha-TTS 的 18 倍，且使用外部 alignments，而 Matcha-TTS 不使用

## 方法

### OT-CMF

定义 $\boldsymbol{x}$ 为 $\mathbb{R}^d$ 维数据空间中的观测值，采样自未知的数据分布 $q(\boldsymbol{x})$。概率密度路径是一个时间相关的概率密度函数 $p_t: [0, 1] \times \mathbb{R}^d \rightarrow \mathbb{R}_{>0}$。生成样本的一种方法是构造概率密度路径 $p_t$，其中 $t \in [0, 1]$，且 $p_0(\boldsymbol{x}) = N(\boldsymbol{x}; 0, I)$ 是先验分布，使得 $p_1(\boldsymbol{x})$ 近似于数据分布 $q(\boldsymbol{x})$。例如，CNFs 首先定义一个向量场 $\boldsymbol{v}_t: [0, 1] \times \mathbb{R}^d \rightarrow \mathbb{R}^d$，通过 ODE 生成流 $\phi_t: [0, 1] \times \mathbb{R}^d \rightarrow \mathbb{R}^d$：
$$\frac d{dt}\phi_t(\boldsymbol{x})=\boldsymbol{v}_t(\phi_t(\boldsymbol{x}));\quad\phi_0(\boldsymbol{x})=\boldsymbol{x}$$
这生成了路径 $p_t$ （样本点的边缘概率分布）。我们可以通过求解上述方程的初值问题从近似的数据分布 $p_1$ 中采样。

假设存在一个已知的向量场 $\boldsymbol{u}_t$，从 $p_0$ 生成概率路径 $p_t$ 到 $p_1 \approx q$。flow matching loss 为：
$$\mathcal{L}_{\mathrm{FM}}(\theta)=\mathbb{E}_{t,p_t(\boldsymbol{x})}\|\boldsymbol{u}_t(\boldsymbol{x})-\boldsymbol{v}_t(\boldsymbol{x};\theta)\|^2,$$
其中 $t \sim U[0, 1]$，$\boldsymbol{v}_t(\boldsymbol{x}; \theta)$ 是参数为 $\theta$ 的神经网络。然而，flow matching 在实践中是不可行的，因为很难获得向量场 $\boldsymbol{u}_t$ 和目标概率 $p_t$。因此，conditional flow matching 考虑：
$$\mathcal{L}_{\mathrm{CFM}}(\theta)=\mathbb{E}_{t,q(\boldsymbol{x}_1),p_t(\boldsymbol{x}|\boldsymbol{x}_1)}\|\boldsymbol{u}_t(\boldsymbol{x}|\boldsymbol{x}_1)-\boldsymbol{v}_t(\boldsymbol{x};\theta)\|^2.$$

这将不可行的边缘概率密度和向量场替换为条件概率密度和条件向量场。同时可以证明，$\mathcal{L}_{\mathrm{CFM}}(\theta)$ 和 $\mathcal{L}_{\mathrm{FM}}(\theta)$ 对 $\theta$ 的梯度是相同的。

Matcha-TTS 使用 optimal-transport conditional flow matching（OT-CFM）训练，OT-CFM loss function 为：
$$\mathcal{L}(\theta)=\mathbb{E}_{t,q(\boldsymbol{x}_1),p_0(\boldsymbol{x}_0)}\|\boldsymbol{u}_t^\mathrm{OT}(\phi_t^\mathrm{OT}(\boldsymbol{x})|\boldsymbol{x}_1)-\boldsymbol{v}_t(\phi_t^\mathrm{OT}(\boldsymbol{x})|\boldsymbol{\mu};\theta)\|^2,$$
其中 $\phi^\mathrm{OT}_t(\boldsymbol{x})=(1-(1-\sigma_{\min})t)\boldsymbol{x}_0+t\boldsymbol{x}_1$，其中每个数据 $\boldsymbol{x}_1$ 与一个随机样本 $\boldsymbol{x}_0 \sim N(0, I)$ 匹配。其梯度向量场 $\boldsymbol{u}_t^\mathrm{OT}(\phi_t^\mathrm{OT}(\boldsymbol{x}_0)|\boldsymbol{x}_1)=\boldsymbol{x}_1-(1-\sigma_{\min})\boldsymbol{x}_0$，是线性的、时不变的，只依赖于 $\boldsymbol{x}_0$ 和 $\boldsymbol{x}_1$。这些特性使得训练更容易、更快、性能更好。

在 Matcha-TTS 中，$\boldsymbol{x}_1$ 是声学帧，$\boldsymbol{\mu}$ 是这些帧的条件均值，从文本中预测，$\sigma_{\min}$ 是一个小的超参数（在我们的实验中为 $1e-4$）。

### 架构

Matcha-TTS 是一个非自回归的 encoder-decoder 架构。如图：
![](image/Pasted%20image%2020240319110228.png)

text encoder 和 duration predictor 用的是 [Glow-TTS- A Generative Flow for Text-to-Speech via Monotonic Alignment Search 笔记](../Glow-TTS-%20A%20Generative%20Flow%20for%20Text-to-Speech%20via%20Monotonic%20Alignment%20Search%20笔记.md) 中的，但是采用 rotational position embeddings 而非相对位置编码。alignment 和 duration model 的训练采用 MAS，预测的 durations 向上取整，用于上采样（复制）encoder 输出的向量，得到 $\boldsymbol{\mu}$。这个均值用于 decoder 的 condition，用于预测向量场 $\boldsymbol{v}_t(\phi^\mathrm{OT}_t(\boldsymbol{x}_0)|\boldsymbol{\mu};\theta)$，但不用作初始噪声样本 $\boldsymbol{x}_0$ 的均值（与 Grad-TTS 不同）。

decoder 的架构如下：
![](image/Pasted%20image%2020240319110843.png)

本质为 U-Net，包含 1D 卷积残差块来下采样和上采样输入，嵌入了 flow-matching 的 step $t \in [0, 1]$。每个残差块后面跟着一个 Transformer block，其 feedforward nets 使用 snake beta activations。这些 Transformers 不使用任何位置编码。比 Grad-TTS 使用的 2D 卷积 U-Net 更快，消耗更少的内存。

## 实验