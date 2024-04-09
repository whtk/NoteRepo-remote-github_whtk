> ICASSP 2023，上交，俞凯 组

1. 强度可控的 emotional TTS 仍然是一个有挑战性的任务，大多数现有方法需要外部计算强度，导致结果质量下降
2. 提出 EmoDiff，基于 diffusion 的 TTS 模型，可以通过 soft-label guidance 进行情感强度控制
    1. 与指定情感的 one-hot vector 不同，EmoDiff 使用 soft label 进行引导，其中指定情感和 Neutral 的值分别设置为 α 和 1 − α。这里的 α 表示情感强度，范围为 0 到 1
3. 实验表明 EmoDiff 可以精确控制情感强度，同时保持高音质
4. 在去噪过程中进行采样，可以生成具有指定情感强度的多样化语音

> 本质就是，用 diffusion 来建模情感，然后用的是 classifier guidance 的思路，引入 soft-label 来控制不同情感对梯度的贡献从而控制不同的情感强度。

## Introduction

1. 主流的 emotional TTS 模型只能在给定情感标签的情况下合成情感语音，而无法控制强度
2. relative attributes rank (RAR) 是定义和获取情感强度的最常用方法，但是这是一个手动构建和分离的阶段，可能导致训练偏差
3. 也有在 emotional embedding space 上的操作，但是 embedding space 的结构影响模型性能，需要额外的约束
4. 提出 soft-label guidance 技术，基于 denoising diffusion models 中的 classifier guidance 
5. 提出 EmoDiff，一个 emotional TTS 模型，具有足够的强度可控性：
    1. 先训练一个 emotion-unconditional acoustic model
    2. 在 diffusion trajectory 上训练一个 emotion classifier
    3. 在推理时，使用 classifier 和 soft emotion label 引导反向去噪过程，其中指定情感和 Neutral 的值分别设置为 α 和 1 − α，而不是 one-hot 分布
6. 实验表明 EmoDiff 可以精确控制情感强度，同时保持高音质，且生成多样化语音样本

## 带有 classifier guidance 的 diffusion 模型

diffusion 模型可以避免 GAN 中的训练不稳定性和模式崩溃问题，优于之前的方法。

EmoDiff 基于 [Grad-TTS- A Diffusion Probabilistic Model for Text-to-Speech 笔记](../Grad-TTS-%20A%20Diffusion%20Probabilistic%20Model%20for%20Text-to-Speech%20笔记.md)，定义 $\boldsymbol{x} \in \mathbb{R}^d$ 为 mel-spectrogram 的一帧，构建前向 SDE：
$$\mathrm{d}\boldsymbol{x}_t=\frac12\boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}-\boldsymbol{x}_t)\beta_t\mathrm{d}t+\sqrt{\beta_t}\mathrm{d}\boldsymbol{B}_t$$
其中 $\boldsymbol{B}_t$ 为标准 Brownian 运动，$t \in [0, 1]$ 为 SDE 时间索引，$\beta_t$ 为 noise scheduler，使得 $\beta_t$ 递增且 $\exp(-\int_0^1 \beta_s \mathrm{d}s) \approx 0$。因此有 $p_1(\boldsymbol{x}_1) \approx \mathcal{N}(\boldsymbol{x};\boldsymbol{\mu},\boldsymbol{\Sigma})$。这个 SDE 也表示了条件分布 $\boldsymbol{x}_t|\boldsymbol{x}_0 \sim \mathcal{N}(\rho(\boldsymbol{x}_0,\boldsymbol{\Sigma},\boldsymbol{\mu},t),\lambda(\boldsymbol{\Sigma},t))$，其中 $\rho(\cdot),\lambda(\cdot)$ 都有封闭形式。因此可以直接从 $\boldsymbol{x}_0$ 中采样 $\boldsymbol{x}_t$。实际情况中，将 $\boldsymbol{\Sigma}$ 设置为单位矩阵，因此 $\lambda(\boldsymbol{\Sigma},t)$ 变为 $\lambda_t \boldsymbol{I}$，其中 $\lambda_t$ 是一个具有已知封闭形式的标量。同时将最终的分布 $p_1(\boldsymbol{x}_1)$ 以文本为条件，即令 $\boldsymbol{\mu} = \boldsymbol{\mu}_\theta(\boldsymbol{y})$，其中 $\boldsymbol{y}$ 是该帧的对齐后的音素表征。

上式中的 SDE 有一个 reverse-time counterpart：
$$\mathrm{d}\boldsymbol{x}_{t}=\left(\frac{1}{2}\boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}-\boldsymbol{x}_{t})-\nabla_{{\boldsymbol{x}}}\log p_{t}(\boldsymbol{x}_{t})\right)\beta_{t}\mathrm{d}t\boldsymbol{+}\sqrt{\beta_{t}}\mathrm{d}\widetilde{\boldsymbol{B}}_{t}$$
其中 $\nabla \log p_t(\boldsymbol{x}_t)$ 是待估计的 score function，$\beta_t$ 是 reverse-time Brownian 运动。它与前向 SDE 共享分布轨迹。因此，从 $x_1 \sim \mathcal{N}(\boldsymbol{\mu},\boldsymbol{\Sigma})$ 开始求解，最终会得到一个真实样本 $x_0 \sim p(\boldsymbol{x}_0 | \boldsymbol{y})$。训练一个神经网络 $s_\theta(\boldsymbol{x}_t,\boldsymbol{y},t)$ 估计 score function，使用 score-matching 目标：
$$\min_\theta\mathcal{L}=\mathbb{E}_{\boldsymbol{x}_0,\boldsymbol{y},t}[\lambda_t\|\boldsymbol{s}_\theta(\boldsymbol{x}_t,\boldsymbol{y},t)-\nabla_{\boldsymbol{x}_t}\log p(\boldsymbol{x}_t\mid\boldsymbol{x}_0)\|^2].$$

### 基于 classifier guidance 的条件采样

Denoising diffusion 也可用于建模条件概率 $p(\boldsymbol{x}|\boldsymbol{c})$，其中 $\boldsymbol{c}$ 是一个类标签。假设一个无条件生成模型 $p(\boldsymbol{x})$ 和一个分类器 $p(\boldsymbol{c}|\boldsymbol{x})$，根据贝叶斯公式：
$$\nabla_{{\boldsymbol{x}}}\log p(\boldsymbol{x}\mid c)=\nabla_{{\boldsymbol{x}}}\log p(c\mid\boldsymbol{x})+\nabla_{{\boldsymbol{x}}}\log p(\boldsymbol{x}).$$

在 diffusion 中，为了从条件分布 $p(\boldsymbol{x}|\boldsymbol{c})$ 中采样，需要估计 score function $\nabla_{\boldsymbol{x}}\log p(\boldsymbol{x}_t|\boldsymbol{c})$。上式可以发现，只需要将分类器的梯度添加到无条件模型中即可。这种条件采样方法称为 classifier guidance，也用于无监督 TTS。

具体使用时，会对分类器梯度进行缩放，以控制 guidance 的强度。即 $\nabla_{\boldsymbol{x}}\log p(\boldsymbol{c}|\boldsymbol{x})$ 变成 $\gamma\nabla_{\boldsymbol{x}}\log p(\boldsymbol{c}|\boldsymbol{x})$，其中 $\gamma \geq 0$ 称为 guidance level。$\gamma$ 越大，样本与类别高度相关，反之多样性越强。

与普通分类器不同，这里的分类器的输入是 SDE 轨迹上的所有 $\boldsymbol{x}_t$，而不仅仅是干净的 $\boldsymbol{x}_0$。时间索引 $t$ 可以是 $[0, 1]$ 中的任何值。因此，分类器也可以表示为 $p(\boldsymbol{c}|\boldsymbol{x}_t,t)$。

上式可以有效地控制类别标签 $c$ 的采样，但不能直接用于 soft-label，即带有 intensity weight 的 label，因为此时的 guidance $p(\boldsymbol{c}|\boldsymbol{x})$ 未被定义。下面将其扩展到情感强度控制。

## EmoDiff

### Unconditional Acoustic Model and Classifier Training

EmoDiff 的训练主要包括无条件 acoustic model 和 emotion classifier 的训练，结构如图：
![](image/Pasted%20image%2020240408213126.png)

首先在 emotional 数据上训练一个基于 diffusion 的 acoustic model，但不提供 emotion 条件，称为“unconditional acoustic model 训练”。模型基于 GradTTS，采用 forced aligners 进行显式持续时间建模。此阶段的训练损失是 $L_{\text{dur}} + L_{\text{diff}}$，其中 $L_{\text{dur}}$ 是对数坐标下持续时间的 l2 loss，$L_{\text{diff}}$ 是的 diffusion loss。还采用先验损失 $L_{\text{prior}} = -\log\mathcal{N}(\boldsymbol{x}_0;\boldsymbol{\mu},\boldsymbol{I})$ 加速收敛。图 a 中使用 $L_{\text{diff}}$ 表示 diffusion 和 prior loss。

训练后，acoustic model 可以在给定输入音素序列 $\boldsymbol{y}$ 下，估计 noise mel 谱 $\boldsymbol{x}_t$ 的 score function，即 $\nabla\log p(\boldsymbol{x}_t|\boldsymbol{y})$（不考虑 emotion label）。

然后用一个 emotion classifier 来判断 noise mel 谱 $\boldsymbol{x}_t$ 的情感类别 $e$。由于文本条件 $\boldsymbol{y}$ 是一直有的，分类器建模为 $p(e|\boldsymbol{x}_t,\boldsymbol{y},t)$，其输入包含：SDE time step $t$，noise mel 谱 $\boldsymbol{x}_t$ 和依赖于音素的高斯均值 $\boldsymbol{\mu}$。采用的是标准的交叉熵损失进行训练。这个阶段会冻结 acoustic model 参数，只更新 emotion classifier 的权重。

由于文本条件 $\boldsymbol{y}$ 一直存在，为了简化符号，将分类器表示为 $p(e|\boldsymbol{x})$。

### 基于 soft-label guidance 的情感强度可控采样

假设情感的数量是 $m$，每个情感 $e_i$ 都有一个 one-hot vector 形式 $e_i \in \mathbb{R}^m,i \in \{0,1,...,m-1\}$。对于每个 $e_i$，只有第 $i$ 维是 1。然后 用 $e_0$ 表示 Neutral。对于一个带有强度 $\alpha$ 的 $e_i$，定义为 $d = \alpha e_i + (1 - \alpha)e_0$。那么分类器 $p(d|\boldsymbol{x})$ 的 log-probability 梯度可以定义为：
$$\nabla_{\boldsymbol{x}}\log p(\boldsymbol{d}\mid\boldsymbol{x})\triangleq\alpha\nabla_{\boldsymbol{x}}\log p(e_i\mid\boldsymbol{x})+(1-\alpha)\nabla_{\boldsymbol{x}}\log p(e_0\mid\boldsymbol{x})$$
即，强度 $\alpha$ 代表了情感 $e_i$ 对 $\boldsymbol{x}$ 采样轨迹的贡献。$\alpha$ 越大，采样轨迹上的“力”越大，越指向 $e_i$，否则指向 $e_0$。因此条件分布的对数梯度可以扩展为：
$$\begin{aligned}\nabla_{\boldsymbol{x}}\log p(\boldsymbol{x}\mid\boldsymbol{d})=\alpha\nabla_{\boldsymbol{x}}\log p(e_i\mid\boldsymbol{x})+(1-\alpha)\nabla_{\boldsymbol{x}}\log p(e_0\mid\boldsymbol{x})+\nabla_{\boldsymbol{x}}\log p(\boldsymbol{x})\end{aligned}$$
当强度 $\alpha$ 为 1.0（100% 情感 $e_i$）或 0.0（100% Neutral）时，上述操作简化为标准的 classifier guidance 形式。因此可以在采样过程中使用 soft-label guidance，生成具有指定情感 $d = \alpha e_i + (1 - \alpha)e_0$ 和强度 $\alpha$ 的样本。

图 c 为强度可控的采样过程。在输入 acoustic model 并获得依赖于音素的 $\boldsymbol{\mu}$ 后，从 $t=1$ 到 $t=0$ 数值求解 reverse-time SDE。每次更新中，将当前 $\boldsymbol{x}_t$ 输入分类器并得到输出概率 $p_t(\cdot|\boldsymbol{x}_t)$。然后计算 guidance，然后用 guidance level $\gamma$ 进行缩放。最终得到 $\hat{\boldsymbol{x}}_0$，其不仅与输入文本相关，还与目标情感 $d$ 和强度 $\alpha$ 相对应。

除了强度控制，soft-label guidance 还可以用于更复杂的混合情感控制。将所有情感的组合 $d = \sum_{i=0}^{m-1}w_ie_i$，其中 $w_i \in [0, 1]$，$\sum_{i=0}^{m-1}w_i = 1$，上述公式可以推广为：
$$\nabla_{\boldsymbol{x}}\log p(\boldsymbol{d}\mid\boldsymbol{x})\triangleq\sum_{i=0}^{m-1}w_i\nabla_{\boldsymbol{x}}\log p(e_i\mid\boldsymbol{x}).$$

同理条件分布的对数梯度也可以表示为这种更一般的形式。从概率上看，因为组合权重 $\{w_i\}$ 可以视为基本情感 $\{e_i\}$ 上的分类分布 $p_e(\cdot)$，上述公式等价于：
$$\begin{aligned}
\nabla_{\boldsymbol{x}}\log p(\boldsymbol{d}\mid\boldsymbol{x})& \triangleq\mathbb{E}_{e\sim p_e}\nabla_{\boldsymbol{x}}\log p(e\mid\boldsymbol{x})  \\
&=-\nabla_{\boldsymbol{x}}\operatorname{CE}\left[p_e(\cdot),p(\cdot\mid\boldsymbol{x})\right]
\end{aligned}$$
其中 CE 是交叉熵函数。上式表明，当沿着 $\nabla_{\boldsymbol{x}}\log p(\boldsymbol{d}|\boldsymbol{x})$ 采样时，实际上是在减小目标情感分布 $p_e$ 和分类器输出 $p(\cdot|\boldsymbol{x})$ 的交叉熵。交叉熵的梯度可以引导采样过程。因此，这种 soft-label guidance 技术可以组合多个基本情感，控制任意复杂的情感。

## 实验和结果（略）
