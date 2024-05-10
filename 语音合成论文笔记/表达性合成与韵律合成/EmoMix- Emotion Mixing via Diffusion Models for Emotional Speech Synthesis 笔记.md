> 平安科技、中科大，Interspeech 2023

1. 现有的 ETTS 主要关注合成有限的情感类型，且不能很好地控制强度
2. 提出 EmoMix，可以生成指定强度或混合情感的语音：
    1. 基于 diffusion 和预训练的语音情感识别模型提取情感嵌入
    2. 通过仅一次采样，结合不同情感条件下的 diffusion 模型预测的噪声，实现混合情感合成
    3. 通过混合不同程度的中性和特定主要情感来控制强度

## Introduction

1. 人类有大约 34,000 种不同的情感，甚至可以同时存在多种情感状态
2. 有人提出了 8 种主要情感：sadness, disgust, joy, fear, anger, anticipation, surprise, 和 trust，其他情感可以看作是这些主要情感的衍生或混合
3. 提出 EmoMix，采用 DPM 和预训练的 SER 模型：
    1. 通过 SER 模型提取的情感嵌入作为额外条件，使得 DPM 反向生成情感语音
    2. 通过引入图像中的语义混合任务的方法，避免直接建模混合情感
    3. 通过仅一次采样，结合不同情感条件下的 DDPM 模型预测的噪声，合成混合情感
    4. 通过中性和特定情感的组合来控制情感的强度

## EmoMix

### Score-based Diffusion

结构基于 [Grad-TTS- A Diffusion Probabilistic Model for Text-to-Speech 笔记](../Grad-TTS-%20A%20Diffusion%20Probabilistic%20Model%20for%20Text-to-Speech%20笔记.md)，把 TTS 任务看成一个 SDE。GradTTS 定义了一个 diffusion 过程，可以将任何数据分布 $X_0$ 转换为标准正态分布 $X_T$：
$$dX_t=-\frac12X_t\beta_tdt+\sqrt{\beta_t}dW_t,\quad t\in[0,T]$$
其中，$\beta_t$ 是预定义的噪声时间表，$W_t$ 是维纳过程。SDE 的 reverse 过程遵循 diffusion 过程的反向轨迹：
$$dX_t=\left(-\frac12X_t-\nabla_{X_t}\log p_t\left(X_t\right)\right)\beta_tdt+\sqrt{\beta_t}d\widetilde{W_t}$$
GradTTS 在采样时用离散的 reverse SDE，从标准高斯噪声 $X_T$ 生成数据 $X_0$：
$$X_{t-\frac1N}=X_t+\frac{\beta_t}N\left(\frac12X_t+\nabla_{X_t}\log p_t(X_t)\right)+\sqrt{\frac{\beta_t}N}z_t,$$
其中，$N$ 是离散 reverse 过程的步数，$z_t$ 是标准高斯噪声。GradTTS 将 $T$ 设为 1，一个 step 为 $\frac{1}{N}$，$t\in\{\frac{1}{N}, \frac{2}{N}, \ldots, 1\}$。

给定数据 $X_0$，从上式中得到的分布中采样 $X_t$，估计 score $\nabla_{X_t}\log p_t(X_t)$：
$$X_t\mid X_0\sim\mathcal{N}\left(\rho\left(X_0,t\right),\lambda(t)\right)$$
其中，$\rho(X_0,t)={\mathrm{e}^{-\frac12\int_0^t\beta_sds}}X_0$，$\lambda(t)=I-e^{-\int_0^t\beta_sds}$。上式得到 score $\nabla_{X_t}\log p_t(X_t\mid X_0)=-\lambda(t)^{-1}\epsilon_t$，其中 $\epsilon_t$ 是用于从 $X_0$ 中采样 $X_t$ 的标准高斯噪声。为了估计 score，对所有 $t\in[0,T]$ 训练 $\epsilon_\theta(X_t,t,\mu, s,e)$，其中 $\mu$ 是 phoneme-dependent 的高斯均值，说话人 $s$ 和情感 $e$ 为 condition。

本文主要关注情感。忽略式中的 $\mu$ 和 $s$，网络简化为 $\epsilon_\theta(X_t,t,e)$。此时损失函数为：
$$\mathcal{L}_{diff}=\mathbb{E}_{\boldsymbol{x}_0,t,\boldsymbol{e},\epsilon_t}[\lambda_t||\epsilon_\theta(\boldsymbol{x}_t,t,\boldsymbol{e})+\lambda(t)^{-1}\epsilon_t||_2^2]$$

### 基于 SER 的 emotion conditioning

采用预训练的 SER 模型，从参考语音中产生连续的 emotion embedding $e$。

SER 模型用的是 [3-d convolutional re- current neural networks with attention model for speech emotion recognition 笔记](../../语音情感识别论文笔记/3-d%20convolutional%20re-%20current%20neural%20networks%20with%20attention%20model%20for%20speech%20emotion%20recognition%20笔记.md) 中的模型：
+ 3-D CNN 层将 mel-spectrum 和其导数作为输入，提取 latent（包含 emotion 
+ BLSTM 和 attention 层生成 emotion embedding $e$。对于 speaker 条件，使用 wav2vec 2.0 模型捕获 speaker 声学条件 $s$

如图：
![](image/Pasted%20image%2020240510102613.png)

模型基于 GradTTS，不同之处在于预测 duration 时基 emotion 和 speaker 条件。Hidden representations $\tilde{\mu}$ 包含了输入文本、emotion embedding $e$ 和 speaker embedding $s$ 的信息。

用另一个 SER + denoiser 进一步减小参考语音和合成语音之间的情感风格差距。风格损失使用 CV中 常用的 gram 矩阵。为了保持合成语音中参考语音的情感韵律，提出风格重建损失：
$$\mathcal{L}_{\mathrm{style}}=\sum_j\|G_j(\hat{m})-G_j(m)\|_F^2$$

其中，$G_j(x)$ 是 SER 模型中 3-D CNN 的第 $j$ 层特征图的 gram 矩阵。$m$ 和 $\hat{m}$ 分别表示参考 mel-spectrogram 和合成 mel-spectrogram。风格重建损失强制合成语音与参考语音具有相似的风格。最终训练目标为：
$$\mathcal{L}=\mathcal{L}_{\mathrm{dur}}+\mathcal{L}_{\mathrm{diff}}+\mathcal{L}_{\mathrm{prior}}+\gamma\mathcal{L}_{\mathrm{style}}$$

其中，$\mathcal{L}_{\mathrm{dur}}$ 是对数坐标下的 duration 的 $l_2$ 损失，$\mathcal{L}_{\mathrm{diff}}$ 是  diffusion 损失。$\gamma$ 是超参数，为 $1e-4$。还采用了 GradTTS 中的先验损失 $\mathcal{L}_{\mathrm{prior}}$ 来加速收敛。

### Emotion Mixing

目标是在推理时合成具有混合情感或具有不同强度的单一情感的语音。如图中绿色部分所示。mix 方法最开始用于 CV 中解决语义混合任务，目的是修改图像中给定对象的某一部分的内容，同时保留其布局语义。

在推理时，EmoMix 在采样步骤 $K_{\max}$ 后，通过替换条件向量来混合两个不同的情感。对相同 emotion 的一组音频样本的 emotion embedding 进行平均来避免单个参考音频导致的不稳定。
+ 先从 base emotion condition $e_1$（如，Happy）开始进行 denoising，合成 coarse base emotion prosody，改过程一直进行到 step $K_{\max}$
+ 然后通过从 $K_{\min}$ 开始的混合 emotion $e_2$（例如，Surprise）的进行 denoising，获得混合情感（如，Excitement）
+ 而在 step $K_{\max}$ 到 $K_{\min}$ 之间，用 noise combine 方法来更好地保留 base emotion，防止其被混合情感覆盖

而对于多个情感条件，通过以下规则来合成多种情感风格：
$$\boldsymbol{\epsilon}_\theta\left(\boldsymbol{x}_t,t,e_{mix}\right)=\sum_{i=1}^M\gamma_i\boldsymbol{\epsilon}_\theta\left(\boldsymbol{x}_t,t,e_i\right)$$

其中，$\gamma_i$ 是每个条件 $e_i$ 的权重，满足 $\sum_{i=1}^M\gamma_i=1$，用于控制每种情感的强度。$M$ 是混合情感类别的数量。

EmoMix 可以通过各种组合来混合情感，且不需要训练新的模型。此采样过程可以解释为增加以下条件分布的联合概率：
$$\sum_{i=1}^M\gamma_i\boldsymbol{\epsilon}_\theta\left(\boldsymbol{x}_t,t,e_i\right)\propto-\nabla\boldsymbol{x}_t\log\prod_{i=1}^Mp\left(\boldsymbol{x}_t\mid e_{\mathrm{tar},i}\right)^{\gamma_i}$$
其中，$e_{\mathrm{tar},i}$ 是指定的目标情感条件。

同时也可以通过混合 neural condition 和 某种情感 的噪声，用不同的 $\gamma$ 在 neural 和 target emotion 之间平滑地进行插值，以控制情感的强度。

## 实验

SER 模型在 IEMOCAP 数据集上训练来获得 emotion feature $e$。

用 ESD 数据集的英文部分来进行实验，包含 10 个说话人，与 IEMOCAP 中的 5 种情感类型相同，包含：Sad, Surprise, Happy, Neutral 和 Angry。采用 ESD 数据集的数据划分，Angry 作为未知的 primary emotion。

$\epsilon_\theta$ 包含 U-Net 和 linear attention 模块，用的是 GradTTS 中的：
+ batch size 为 32
+ Adam 优化器，lr 为 $10^{-4}$，训练 100 万步
+ 使用 Montreal Forced Aligner (MFA) 提取的语音和文本的对齐信息用于训练 duration predictor
+ 用 Hifi-GAN 作为声码器

结果：
![](image/Pasted%20image%2020240510153240.png)

+ vocoder 对语音质量影响很小
+ EmoMix 在 SMOS 上超过 baseline，MOS 相当，MCD 最低
+ 在合成未知情感（Angry）时，EmoMix 仍然表现良好

消融实验：
![](image/Pasted%20image%2020240510153553.png)

+ GradTTS (w/ emo label)：把离散的 emotion labels 作为条件，而不是 SER 提取的 emotion features，质量和相似度分数下降
+ 没有 style reconstruction loss 会导致 CSMOS 下降，表明风格损失可以强制 EmoMix 合成参考音频中的目标情感


设置 $K_{\max}$ 和 $K_{\min}$ 为 $0.6T$ 和 $0.2T$，只变化 $\gamma$ 来控制混合情感的强度，$T$ 设置为 10。为了避免 base emotion 被完全覆盖，$\gamma$ 不大于 0.8。

评估了两种不同的混合情感：Excitement 和 Disappointment。分别以 Happy 和 Sad 为 base emotion 来混合 Surprise。用另一个 SER 的 softmax 层的分类概率来分析混合情感的性能：
+ 当对应于 Surprise 的噪声的百分比的增加时，Surprise 的概率持续上升，Happy 和 Sad 的概率下降

下表为不同混合强度下 10 种混合情感的 CMOS 结果：
![](image/Pasted%20image%2020240510154613.png)
EmoMix 的 mix 方法在语音质量上略有下降，而 MixedEmotion 的 mix 方法导致明显的质量下降

下图为每个 emotion category 的 confusion matrix：
![](image/Pasted%20image%2020240510154156.png)
说明模型可以准确地反映所需的强度。