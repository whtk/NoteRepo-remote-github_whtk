> ICASSP 2024，Universität Hamburg，德国

1. 语音情感转换（speech emotion conversion）用于将语音中的情感转换为目标情感，同时保留内容和说话人身份
2. 本文关注于 in-the-wild 场景，不依赖于 parallel data
3. 提出基于 diffusion 的 EmoConv-Diff 用于情感转换，训练时，基于 emotion 对输入语音进行重建，推理时，使用 target emotion embedding 进行转换。和基于 categorical representations 的情感转换不同，这里使用连续的 arousal dimension 来表示情感，同时实现 intensity 控制
4. 在 MSP-Podcast v1.10 实验，结果表明，可以合成具有可控目标情感的语音

> 本质也是将语音解耦为 emotion、lexical 和 speaker 信息，然后重构+替换 进行情感转换。和 ICASSP 2024 的另一篇文章 Zero Shot Audio To Audio Emotion Transfer With Speaker Disentanglement 有一定的相似性。

## Introduction

1. emotion-conditioned 的语音合成仍然是一个挑战；语音情感转换（SEC）为其一个子领域，目标是将一个语音转到另一个语音，且转换的是情感，但是保留内容和说话人身份
2. SEC 中的情感可以表示为 categorical 或 continuous representations，情感是一个 fuzzy 的概念，categorical representations 无法捕捉情感的细微差别，而 continuous representations 使用 arousal 和 valence 两个表示情感，但是 audio 模态通常只能捕捉 arousal 维度，因此本文使用 continuous arousal dimension 来表示情感，且可以直接实现 intensity control
3. 当前的 SEC 通常在高质量的 acted-out 数据上训练，但是对噪声和真实场景中的变化敏感，且 acted-out 需要平行数据；本文则专注于 non-parallel in-the-wild 数据
> 平行数据，即每个 source utterance 都需要一个 target emotion 的 GT utterance
4. 此外，parallel utterances 中的 disentanglement 很难，需要将 source utterance 分解为 emotion、lexical 和 speaker 信息，然后合成 target emotion 的语音
5. 本文提出了一种基于 diffusion 的方法，训练时，基于 emotion 对 source utterance 进行重建，推理时，使用 target emotion embedding 进行转换。贡献如下：
    + 提出了一种不依赖于 parallel utterances 的 emotion-conditioned diffusion 模型
    + 模型在 non-parallel in-the-wild speech utterances 上训练
    + 首次解决了 SEC 的 non-parallel in-the-wild data 问题，且首次使用 diffusion models
    + 在极端 target emotions 上改进了 HiFiGAN

## Diffusion 模型

### EMOCONV-Diff

将 SEC 任务定义为：给定 source utterance 的 mel 谱 $\mathbf{X}_{l,s,e}$（简写为 $\mathbf{X}_{0}$），其包含 lexical content $l$、speaekr identify $s$ 和 emotion 信息 $e$，目标是，生成一个新的 mel 谱 $\hat{\mathbf{X}}_{l,s,\bar{e}}$，只将 arousal 信息转换为目标值 $\bar{e}$。

提出的 EmoConv-Diff 如图：
![](image/Pasted%20image%2020240411160813.png)

包含：
+ 一组 encoder，用于 encode 要解耦的属性
+ 基于 diffusion 的 decoder，用于解耦属性，进行 emotion-controllable 的语音合成。输出是 mel 谱 $\hat{\mathbf{X}}_{l,s,\bar{e}} \in \mathbb{R}^{n \times T}$，最后用预训练的 HiFiGAN vocoder 转为时域信号

### encoder

EmoConv-Diff 包含三个 encoder：
+ phoneme encoder $\phi(\cdot)$，采用 mel 特征来编码 lexical content，定义 $\mathbf{Y} := \phi(\mathbf{X}_0)$，是 source audio 的 "average voice" 的表征，其维度和 source mel $\mathbf{X}_0 \in \mathbb{R}^{n \times T}$ 相同，模型用的是预训练的 transformer-based encoder
+ speaker encoder $S(\cdot)$，用于编码 speaker identity，输出是 $d$ 维 speaker representation $S(\cdot) \in \mathbb{R}^{128}$，用的是预训练的 speaker verification 模型
+ emotion encoder $E(\cdot)$，用于编码 emotional 信息，输出是 $1024$ 维 emotion representation $E(\cdot) \in \mathbb{R}^{1024}$，通过 fine-tune Wav2Vec2-Large- Robust 模型得到

### 基于 diffusion 的 decoder

用的是 [Grad-TTS- A Diffusion Probabilistic Model for Text-to-Speech 笔记](../Grad-TTS-%20A%20Diffusion%20Probabilistic%20Model%20for%20Text-to-Speech%20笔记.md) 中的结构，设 $t$ 为 time-step。对于 $0 \leq t \leq 1$，前向 SDE 为：
$$\mathrm{d}\mathbf{X}_t=\frac12\beta_t(\mathbf{Y}-\mathbf{X}_t)\mathrm{d}t+\sqrt{\beta_t}\mathrm{d}\mathbf{w}$$
其中 $\mathbf{w}$ 是标准 Wiener 过程，$\mathbf{X}_t$ 是当前的 process state，初始条件为 $\mathbf{X}_0 = \mathbf{X}_{l,s,e}$，$\beta_t$ 是 noise schedule。$\mathbf{X}_t$ 服从高斯分布，称为 perturbation kernel：
$$p_{0t}(\mathbf{X}_t|\mathbf{X}_0,\mathbf{Y})=\mathcal{N}_{\mathbb{C}}\left(\mathbf{X}_t;\boldsymbol{\mu}(\mathbf{X}_0,\mathbf{Y},t),\sigma(t)^2\mathbf{I}\right)$$
其中 $\boldsymbol{\mu}(\mathbf{X}_0,\mathbf{Y},t)$ 计算为：
$$\mu(\mathbf{X}_0,\mathbf{Y},t)=\alpha_t\mathbf{X}_0+\left(1-\alpha_t\right)\mathbf{Y}$$
其中 $\alpha_t = e^{-2\int_0^t\beta_s\mathrm{d}s}$，方差计算为：
$$\sigma(t)^2=\begin{pmatrix}1-\alpha_t^2\end{pmatrix}\mathbf{I}$$
将 $\alpha_t$ 的表示为 $\beta_t$，而 $\beta_t = b_0 + t(b_1 - b_0)$，选择 $b_0, b_1 > 0$ 使得 $\alpha_1 \approx 0$。此时均值的变化是一个插值过程，$t = 0$ 开始的分布是 source $\mathbf{X}_0$ 的分布，在 $t = 1$ 时，分布是 "average voice" phoneme features $\mathbf{Y}$。前向 SDE 有一个 reverse SDE：
$$\mathrm{d}\mathbf{X}_t=\left[-\frac12\beta_t(\mathbf{Y}-\mathbf{X}_t)+\beta_t\nabla_{\mathbf{X}_t}\log p_t(\mathbf{X}_t|\mathbf{Y})\right]\mathrm{d}t+\beta_t\mathrm{d}\widetilde{\mathbf{w}}$$
其中 $\mathrm{d}\widetilde{\mathbf{w}}$ 是反向 Wiener 过程。反向过程与前向过程相同，即反向 SDE 从 "average voice" 分布开始，$t = 0$ 时，分布是 source-targets。

然后用 score model $\theta(\mathbf{X}_t,\mathbf{Y},S(\mathbf{X}_0),E(\mathbf{X}_0),t)$，或简写为 $s_{\theta}(\mathbf{X}_t,t)$，用于近似 score function $\nabla_{\mathbf{X}_t}\log p_t(\mathbf{X}_t|\mathbf{Y})$，即 log-density 的梯度。使用 U-Net 架构作为 score model $s_{\theta}$。通过训练 $s_{\theta}$，采用反向 SDE 从 "average voice" $Y$，基于 speaker identity $S(\mathbf{X}_0)$ 和 emotion embeddings $E(\mathbf{X}_0)$ 来生成 source target $\mathbf{X}_0$ 。
> diffusion-based decoder 在学习语音属性 $l, s, e$ 解耦的同时，还要重构 $\mathbf{X}_0$。从而可以在训练过程中不需要平行数据

推理时，使用 target emotion embedding $E(\bar{e})$ 来将 source utterance 的情感转换为 target emotion。target emotion embedding $E(\bar{e})$ 定义为情感类别 $\bar{e}$ 对应的参考 utterance 的平均 emotion embedding：
$$E(\bar{e}):=\frac1{|A_p(\bar{e})|}\sum_{\mathbf{X}_0\in A_p(\bar{e})}E(\mathbf{X}_0)$$
其中集合 $A_p(\bar{e})$ 定义为属于特定 target arousal $\bar{e}$ 的前 $p = 20\%$ 的样本。

### 损失函数

score model 使用 score matching loss 进行训练，目标是近似 score function。score matching loss 定义为：
$$\mathcal{L}_s(\mathbf{X}_t)=\mathbb{E}_{\epsilon_t}\left[||\mathbf{s}_\theta(\mathbf{X}_t,t)+\sigma(t)^{-1}\epsilon_t||_2^2\right]$$
其中 $\mathbf{X}_t = \mu(t) + \sigma(t)\epsilon_t$，$\epsilon_t$ 从 $\mathcal{N}(0,\sigma(t))$ 中采样。此外，为了更好的 emotion conditioning，使用 mel 谱重构损失 $\mathcal{L}_m$，计算 L1-norm：
$$\mathcal{L}_m(\hat{\mathbf{X}}_0)=\sum_x\|\mathbf{X}_0-\hat{\mathbf{X}}_0\|_1$$
其中 $\hat{\mathbf{X}}_0$ 是合成语音的 mel 谱。在训练 score model 时，计算 $\hat{\mathbf{X}}_0$ 要很多步（需要求解完整的 reverse SDE），于是使用单步近似 $\hat{\mathbf{X}}_0$，只依赖于 $\mathbf{X}_t, s_{\theta}, \mathbf{Y}$。使用 Tweedie's formula 近似 $\hat{\mathbf{X}}_0$：
$$\hat{\mathbf{X}}_0=\frac{\hat{\mu}(t)-(1-\alpha_t)\:\mathbf{Y}}{\alpha_t}\:$$
其中 $\hat{\mu}(t)$ 是 $\mu(t)$ 的估计，定义为 $\hat{\mu}(t) = \mathbf{X}_t - (s_{\theta}(\mathbf{X}_t,t) \cdot \sigma(t)^2)$。最终的损失函数为：
$$\mathcal{L}(\mathbf{X}_t,\hat{\mathbf{X}}_0)=\mathcal{L}_s(\mathbf{X}_t)+\lambda_t\mathcal{L}_m(\hat{\mathbf{X}}_0)$$
其中 $\lambda_t$ 取决于当前的 diffusion 的 time-step $t$。由于 $\mathbf{X}_t$ 对于较大的 $t$ 更多的噪声，设置 $\lambda_t = 1 - t^2$，$t$ 越小权重越大，$t$ 越大权重逐渐减小。

## 实验和结果（略）
