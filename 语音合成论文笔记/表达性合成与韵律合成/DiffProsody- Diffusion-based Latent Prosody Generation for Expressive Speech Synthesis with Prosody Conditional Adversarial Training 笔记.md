> 高丽大学，2023.7.31 preprint 

1. 传统的韵律建模依赖于自回归方法来预测量化的 prosody vector，但是会出现长时依赖问题，而且推理很慢
2. 提出 prosody vector，采用 diffusion-based latent prosody generator 和 prosody conditional adversarial training 来合成 expressive speech
3. 提出的  prosody generator 很有效，且 prosody conditional discriminator 可以通过模仿韵律提高生成的质量
4. 采用 denoising diffusion generative adversarial networks 来提高韵律的生成速度，比传统的生成模型快 16 倍

> 结构为 text encoder + mel 谱 decoder，然后用 diffusion 来建模韵律，然后对于 diffusion 的中间变量也引入了所谓的对抗训练。

## Introduction

1. prosody 包含了 pitch、energy 和 duration 等 speech properties，在 expressive speech 的合成中起着至关重要的作用
2. 采用 reference encoder 提取 prosody vector 的效果很好，但是在没有 GT prosody 信息的情况下，无法反映语音中的 prosody 细节
3. 独立地建模 prosodic features 可能会导致不自然的结果，因为这些特征之间有内在的相关性
4. 提出 DiffProsody，采用 diffusion-based latent prosody generator（DLPG）和 prosody conditional adversarial training 来生成 expressive speech
5. 贡献包含：
	1. 提出 diffusion-based latent prosody 建模方法，能够生成高质量的 latent prosody representation，从而增强合成语音的表达性，采用 denoising diffusion generative adversarial networks (DDGANs) 来减少 time step 数量，从而加速 AR 和 DDPM
	2. 提出 prosody conditional adversarial training，确保 TTS 模块准确反映 prosody

## 相关工作（略）

## DiffProsody

DiffProsody 通过引入 diffusion-based latent prosody generator（DLPG）和 prosody conditional adversarial training 来增强语音合成，整体结构和流程如下图：
![](image/Pasted%20image%2020240407192957.png)

第一阶段，训练 TTS 模块和 prosody encoder，其输入为文本序列和 reference Mel-spectrogram。prosody conditional discriminator 判断来自 prosody encoder 和 Mel-spectrogram 的 prosody vector。

第二阶段，训练 DLPG 来采样与输入文本和 speaker 相对应的 prosody vector。

推理时，合成语音过程中不依赖 reference Mel-spectrogram，而是使用 DLPG 的输出，得到指定 prosody 的 expressive speech。

### TTS 模块

TTS 模块使用 speaker 和 prosody vectors 作为条件将文本转换为 Mel-spectrograms，如上图 a。包括 text encoder 和 decoder。text encoder 在 phoneme 和 word levels 处理文本，如上图 c。输入文本 $x_{txt}$ 通过 phoneme encoder $E_p$ 和 word encoder $E_w$ 转换为 text hidden representation $h_{txt}$。$E_p$ 输入为 phoneme-level text $x_{ph}$，$E_w$ 输入为 word-level text $x_{wd}$。$h_{txt}$ 为 $E_p(x_{ph})$ 和 $E_w(x_{wd})$ 输出进行 element-wise 求和，然后拓展到  phoneme-level：
$$\mathbf{h}_{txt}=E_p(\mathbf{x}_{ph})+expand(E_w(\mathbf{x}_{wd}))$$
expand 操作将 word-level features 拓展到 phoneme-level。然后把 $h_{txt}$ 和 speaker hidden representation $h_{spk}$ 输入到 prosody 模块得到量化的 prosody vector $z_{pros}$。
> $h_{spk}$ 通过预训练的 speaker encoder 获取，使用 Resemblyzer 开源模块提取 $h_{spk}$。

第一阶段的训练中，prosody encoder 的输入为 target Mel-spectrogram。
> 推理时，将 $h_{txt}$ 和 $h_{spk}$ 输入 DLPG 得到 $z^{\prime}_{pros}$（不依赖 reference Mel-spectrogram）。

通过对 latent vectors $h_{txt}$、$h_{spk}$ 和 $z_{pros}$ 拓展到 phoneme-level 来组合这些信息，然后进行 element-wise 相加：
$$\mathbf{h}_{total}=\mathbf{h}_{txt}+\mathbf{h}_{spk}+\mathbf{z}_{pros}$$

phoneme duration 通过 duration predictor 建模，输入为 $h_{total}$，预测 frame-level 的 phoneme duration：
$$dur'=DP(\mathbf{h}_{total})$$

length regulator LR 使用 phoneme duration $dur$ 将输入拓展到 frame-level。然后通过 $D_{mel}$ 将拓展的 $h_{total}$ 转换为 Mel-spectrogram $y^{\prime}$：
$$\mathbf{y}'=D_{mel}(LR(\mathbf{h}_{total},dur))$$

TTS 模块使用了 MSE 和 SSIM loss 训练。对于 duration，则使用 MSE loss：
$$\begin{aligned}\mathcal{L}_{rec}&=\mathcal{L}_{MSE}(\mathbf{y},\mathbf{y}^{\prime})+\mathcal{L}_{SSIM}(\mathbf{y},\mathbf{y}^{\prime}).\\\\\mathcal{L}_{dur}&=\mathcal{L}_{MSE}(dur,dur^{\prime}).\end{aligned}$$

### Prosody 模块

图 b 为 prosody 模块，包含：
+ prosody encoder $E_{pros}$，用于从 reference Mel-spectrogram 中得到 prosody vector
+ DLPG，输入 text 和 speaker hidden states 来生成 prosody vector
+ codebook $Z=\{z_k\}_{k=1}^{K}\in\mathbb{R}^{K\times dz}$，$K$ 为 codebook 大小，$dz$ 为 code 的维度。

训练 $E_{pros}$ 时，使用 low-frequency band Mel-spectrogram 代替 full-band Mel-spectrogram，以减轻 disentanglement。$E_{pros}$ 结构如上图 d，包含两个卷积 stack 和一个 word-level pooling 层。

为了提取目标 prosody，$E_{pros}$ 使用 target Mel-spectrogram $y[0:N]$ 的最低的 $N$ 个 bins、 $h_{txt}$ 和 $h_{spk}$ 作为输入。输出为 prosody vector $h_{pros}\in\mathbb{R}^{L\times dz}$，$L$ 为输入文本的 word-level 长度：
$$\mathbf{h}_{pros}=E_{pros}(\mathbf{y}_{[0:N]},\mathbf{h}_{txt},\mathbf{h}_{spk})$$

推理时，使用第二阶段训练的 prosody generator 得到 prosody vector $h^{\prime}_{pros}$：
$$\mathbf{h}'_{pros}=DLPG(\mathbf{h}_{txt},\mathbf{h}_{spk}).$$

为了得到离散的 prosody token sequence $z_{pros}\in\mathbb{R}^{L\times dz}$，VQ 层 $Z$ 将每个 prosody vector $h_{pros}\in\mathbb{R}^{dz}$ 映射到最近的 codebook entry $z_k\in\mathbb{R}^{dz}$：
$$\mathbf{z}_{pros}^i=\underset{\mathbf{z}_k\in Z}{\operatorname*{\arg\min}}||\mathbf{h}_{pros}^i-\mathbf{z}_k||_2\mathrm{~for~}i=1\mathrm{~to~}L,$$
其中 $z_{pros}^i$ 为 $z_{pros}$ 的第 $i$ 个元素。第一阶段，TTS 模块与 codebook $Z$ 和 prosody encoder $E_{pros}$ 联合训练，损失为：
$$\mathcal{L}_{vq}=||sg[\mathbf{h}_{pros}]-\mathbf{z}_{pros}||_2^2+\beta||\mathbf{h}_{pros}-sg[\mathbf{z}_{pros}]||_2^2,$$
其中 $sg[·]$ 表示 stop-gradient 算子。用 exponential moving average (EMA) 来提高学习效率，用于 codebook 更新。

### Prosody conditional 对抗学习

提出 prosody conditional discriminators (PCDs) 用于不同长度的输入（受 multi-length window discriminators 启发）。PCD 结构如上图 e。PCD 输入为 Mel-spectrogram $y$ 和量化的 prosody vector $z_{pros}$，判断其是原始的还是生成的。

模型包含两个轻量级 CNN 和全连接。一个 CNN 接收 Mel-spectrogram，另一个接收 $z_{pros}$ 和 $y$ 的组合。为了匹配相应的 PCD，随机裁剪 Mel-spectrogram 和拓展的 $z_{pros}$ 的长度。采用 least square GAN loss：
$$\begin{gathered}\mathcal{L}_D=\sum_i[\mathbb{E}[(PCD^i(\mathbf{y}^{\prime},\mathbf{z}_{pros}))^2]\\+\mathbb{E}[(PCD^i(\mathbf{y},\mathbf{z}_{pros})-1)^2]],\\\mathcal{L}_G=\sum_i\mathbb{E}[(PCD^i(\mathbf{y}^{\prime},\mathbf{z}_{pros})-1)^2],\end{gathered}$$
其中 $\mathcal{L}_D$ 为 discriminator 的训练目标，$\mathcal{L}_G$ 为 TTS 模块的。TTS 模块的最终目标 $\mathcal{L}_{TTS}$：
$$\mathcal{L}_{TTS}=\mathcal{L}_{rec}+\mathcal{L}_{dur}+\mathcal{L}_{vq}+\lambda_1\mathcal{L}_G,$$
$\lambda_1$ 为对抗损失的权重。

### 基于 diffusion 的 latent prosody generator

DLPG 的训练过程如图：
![](image/Pasted%20image%2020240407200647.png)

目标是生成从第一阶段训练的 prosody encoder 中提取的 $h_{pros}$（假设模型训练生成的是 $h^{\prime}_{pros}$），模型输入为 $h_{spk}$ 和 $h_{txt}$（$h_{pros}$ 相当于 diffusion 中的 $x_0$） 。DLPG 的 generator $G_{\theta}$ 直接生成 $x^{\prime}_0$：
$$\mathbf{x}_0^{\prime}=G_\theta(\mathbf{x}_t,t,\mathbf{h}_{spk},\mathbf{h}_{txt})$$
其中 $t$ 为 diffusion 的 time step。为了进行对抗训练，$x^{\prime}_{t-1}$ 通过 posterior sampling $q(x^{\prime}_{t-1}|x_t,x^{\prime}_0)$ 从 $x_t$ 和 $x^{\prime}_0$ 得到。然后，time-dependent discriminator $D_{\phi}$ 确定 $x_{t-1}$（$x_{t-1}$ 是从 $x_0$ 的 forward process 中得到的）和 $x^{\prime}_{t-1}$（通过 posterior sampling 得到的）关于 $t$ 和 $x_t$ 的 compatibility，条件为 $h_{spk}$ 和 $h_{txt}$。$G_{\theta}$ 的目标函数定义如下：
$$\begin{gathered}\mathcal{L}_{G_\theta}^{adv}=\sum_{t\geq1}\mathbb{E}[(D_\phi(\mathbf{x}_{t-1}^{\prime},\mathbf{x}_t,t,\mathbf{h}_{txt},\mathbf{h}_{spk})-1)^2],\\\\\mathcal{L}_{G_\theta}^{rec}=L_{MAE}(\mathbf{x}_0,\hat{\mathbf{x}}_0),\end{gathered}$$
其中 $L_{G_\theta}^{adv}$ 为对抗损失，$L_{G_\theta}^{rec}$ 为重构损失。总的 generator loss $L_{G_\theta}$：
$$\mathcal{L}_{G_\theta}=\mathcal{L}_{G_\theta}^{rec}+\lambda_2\mathcal{L}_{G_\theta}^{adv},$$
其中 $\lambda_2$ 为对抗损失的权重。$D_{\phi}$ 的目标函数为：
$$\begin{aligned}\mathcal{L}_{D_\phi}&=\sum_{t\geq1}[\mathbb{E}[D_\phi(\mathbf{x}_{t-1}^{\prime},\mathbf{x}_t,t,\mathbf{h}_{txt},\mathbf{h}_{spk})^2]\\&+\mathbb{E}[(D_\phi(\mathbf{x}_{t-1},\mathbf{x}_t,t,\mathbf{h}_{txt},\mathbf{h}_{spk})-1)^2]].\end{aligned}$$

DLPG 利用 DDGAN 框架，可以在仅仅几个 time step 内实现稳定和高质量的结果。推理时，$G_{\theta}$ 迭代 $T$ 次生成 $x^{\prime}_0$。$x_T$ 为正态分布。$h^{\prime}_{pros}$ 作为最终 $x^{\prime}_0$ 得到。最终的 prosody vector $z_{pros}$ 通过 $h^{\prime}_{pros}$ 进行量化得到。

### 推理

推理过程如下：
1. 从 text encoder 提取 $h_{txt}$，从预训练的 speaker encoder 提取 $h_{spk}$
2. DLPG 使用 $h_{txt}$ 和 $h_{spk}$ 生成 $h^{\prime}_{pros}$
3. $h^{\prime}_{pros}$ 映射到 codebook $Z$，得到 prosody vector $z_{pros}$
4. decoder $D_{mel}$ 使用 $h_{txt}$、$h_{spk}$ 和 $z_{pros}$ 生成 Mel-spectrogram $y^{\prime}$。需要将 $h_{txt}$、$h_{spk}$ 和 $z_{pros}$ 拓展到 frame-level。phoneme-duration 由 duration predictor 预测得到。
5. Mel-spectrogram $y^{\prime}$ 通过预训练的 vocoder 转换为 raw waveform。

## 实验结果和讨论（略）
