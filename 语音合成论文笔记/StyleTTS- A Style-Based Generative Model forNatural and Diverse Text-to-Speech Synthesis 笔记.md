> 2023，preprint，哥伦比亚大学

1. 现有的 parallel TTS 很难处理处理最优对对齐，因为 speech 和 duration 是独立生成的
2. 提出 StyleTTS，提出 Transferable Monotonic Aligner (TMA) 和 duration-invariant 数据增强，可以在单/多说话人数据集上超过现有的 baseline
3. 通过自监督学习，StyleTTS 可以生成和参考语音相同的情感和韵律，而无需显式的标签

## Introduction

1. 通过在 encoder 的输出中拼接 speaker embedding 或者引入 style vector 并不能有效地捕获声学特征中的变化量
2. 通过 conditional normalization 方法将风格进行迁移在很多任务中很有效，但是在语音合成中用的少
3. parallel TTS 通常也采用 external aligner 来做对齐，但是不是在 TTS 任务中训练的，因此不是最优解，甚至出现过拟合问题
4. 提出 StyleTTS，引入基于风格的生成模型，采用 AdaIN 将来自参考音频的 style vectors 集成进去；同时采用一种新的 Transferable Monotonic Aligner (TMA) 来找到最优的对齐

> （输入为文本 + 参考语音）其实创新点就是在 decoder 生成阶段，引入了 AdaIN 模块，然后用在 ASR 任务上预训练的 aligner 作为预训练模型，最后一个创新就是对输入的 mel 谱 在时域做数据增强，然后给出了一堆的损失函数。。。
> two stage 的好处就是，可以用第一个 stage 训练得到的表征来训练第二个 stage 的模型。

## 方法

### 框架

给定输入 phonemes $t\in\mathcal{T}$ 和 任意参考 mel 谱 $x\in\mathcal{X}$，目标是生成文本 $t$ 对应的 mel 谱 但是能够反映 $x$ 的风格的 $\tilde{x}$。

这里的风格定义为，除了 内容 之外的，包括不限于：prosodic pattern, lexical stress, formants transition, speaking rate, and speaker identity 等

框架包含 8 个模块，分为 3 类：
+ 语音生成模块：text encoder、style encoder、decoder
+ 文本预测模块：duration 和 prosody predictor
+ utility 模块（只在训练时候用）：包含 discriminator、text aligner 和 pitch extractor.

![](image/Pasted%20image%2020231121164709.png)

Text Encoder $T$：将 phoneme 转为 hidden representation $\boldsymbol{h_\mathrm{text}}=T(\boldsymbol{t})$，包含 3 层 CNN + LSTM

Text Aligner $A$：生成 mel 谱 和 phoneme 之间的对齐 $d_{\mathrm{align}}$，在重构阶段还同时训练了一个 decoder $G$。以 [Tacotron 2- Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions 笔记](Tacotron%202-%20Natural%20TTS%20Synthesis%20by%20Conditioning%20WaveNet%20on%20Mel%20Spectrogram%20Predictions%20笔记.md) 的 decoder 为模型，首先使用 LibriSpeech  在 ASR 任务上进行预训练（因为 ASR 包含了 语音-文本 对），然后和本文的 decoder $G$ 同时 fine tune 训练。

Style Encoder $E$：encoder 从 mel 谱 中提取 风格向量 $s=E(\boldsymbol{x})$，包含 4 个 residual 模块，然后接一个时间轴上的平均池化。

Pitch Extractor $F$：直接提取 F0（多少赫兹）而不做任何后处理，训练一个 pitch extractor $F$ 来和 decoder $G$ 端到端训练。$F$ 是一个 JDC 网络，采用 YIN 算法在 LibriSpeech 上训练。然后 fine tune，最后得到预测的 pitch $p_x=F(\boldsymbol{x})$。

Decoder $G$：用于重构 mel 谱，即 $\hat{x}=G\left(\boldsymbol{h_\mathrm{text}}\cdot\boldsymbol{d_\mathrm{align}},\boldsymbol{s,p_x},n_{\boldsymbol{x}}\right)$，其中的 $\boldsymbol{h_\mathrm{text}}\cdot\boldsymbol{d_\mathrm{align}}$ 为 phoneme 对齐后的表征，$n_{\boldsymbol{x}}$ 为每帧的能量，decoder 包含 7 层带有 AdaIN 的 residual block，定义为：
$$\mathrm{AdaIN}(c,s)=L_\sigma(s)\frac{c-\mu(c)}{\sigma(c)}+L_\mu(s)$$
其中的 $c$ 是单通道的 feature map，$s$ 是 style vector。

Discriminator $D$：用 discriminator 来实现更好的合成质量，结构和 style encoder 一致

Duration Predictor $S$：包含三层的 BLSTM（$R$） + AdaIN + 线性投影层。通过 $h_{\mathrm{prosody}}=R\left(\boldsymbol{h}_{\mathrm{text}}\right)$ 作为 $P$ 的输入，BLSTM 与  pitch predictor $P$ 共享。

Prosody Predictor $P$：输入文本和 style vector，预测 pitch $\hat{p}_{x}$ 和 能量 $\hat{n}_x$。生成的 $h_{\mathrm{prosody}}$ 回通过两个 三层的带有 AdaIN 的 residual 模块分别预测 pitch 和 能量。

### 训练目标函数

分为两个阶段：
+ 阶段1，学习 从 text, pitch, energy, and style 中 重构 mel 谱
+ 阶段2，固定除了 duration 和 prosody predictors 这两个模块外的其他模块，训练从文本中预测 duration, pitch, 和 energy

#### 阶段 1 目标函数

mel 谱 重构：
给定 mel 谱 和其对应的文本，L1 重构损失为：
$$\mathcal{L}_{\mathrm{mel}}=\mathbb{E}_{\boldsymbol{x},\boldsymbol{t}}\left[\left\|\boldsymbol{x}-G\left(\boldsymbol{h}_{\mathrm{text}}\cdot\boldsymbol{a}_{\mathrm{align}},\boldsymbol{s},p_{\boldsymbol{x}},n_{\boldsymbol{x}}\right)\right\|_{1}\right]$$

这里的 $a_{\mathrm{align}}=A(x,t)$ 是来自 text aligner 的  attention 对齐。为了端到端训练 decoder 和 aligner，采用下述策略：
+ 一半的时间采用 raw attention，从而允许梯度反向传播到 text aligner
+ 另一半时间，采用不可微的 MAS 训练  decoder 产生 alignment
> 这里注意，如果所有的时间都用 hard alignment，则和 FastSpeech 用的外部 aligner 一致；如果所有的时间都用 soft alignment，就和 Cotatron 一样。

TMA 目标函数：
采用 sequence-to-sequence ASR 目标函数 fine tune aligner：
$$\mathcal{L}_{\mathrm{s}2\mathrm{s}}=\mathbb{E}_{\boldsymbol{x},\boldsymbol{t}}\left[\sum_{i=1}^N\mathbf{C}\mathrm{E}(\boldsymbol{t}_i,\hat{\boldsymbol{t}}_i)\right]$$
其中，$N$ 为 $t$ 中 phoneme 的数量，$\hat{t}_i$ 为第 $i$ 个预测的 phoneme。

同时由于 align 不一定单调，采用 L1 损失强迫其接近单调：
$$\mathcal{L}_\text{mono}=\mathbb{E}_{\boldsymbol{x},\boldsymbol{t}}\left[\left\|\boldsymbol{a}_{\mathrm{align}}-\boldsymbol{a}_{\mathrm{hard}}\right\|_1\right]$$
其中 $\boldsymbol{a}_{\mathrm{hard}}$ 是通过 MAS 得到的。

对抗目标函数：
有两个对抗损失，原始的交叉熵损失和一个额外的 feature-matching 损失：
$$\begin{aligned}\mathcal{L}_{\mathrm{adv}}&=\mathbb{E}_{\boldsymbol{x},\boldsymbol{t}}\left[\log D(\boldsymbol{x})+\log\left(1-D(\hat{\boldsymbol{x}})\right)\right],\\\\\mathcal{L}_{\mathrm{fm}}&=\mathbb{E}_{\boldsymbol{x},\boldsymbol{t}}\left[\sum_{l=1}^T\frac1{N_l}\left\|D^l(\boldsymbol{x})-D^l(\hat{\boldsymbol{x}})\right\|_1\right],\end{aligned}$$
> feature matching loss 是用来训练生成器那部分的。

阶段 1 损失函数汇总：
$$\begin{aligned}\min_{G,A,E,F,T}\max_D&\mathcal{L}_{\mathrm{mel}}+\lambda_{\mathrm{s}2s}\mathcal{L}_{\mathrm{s}2s}+\lambda_{\mathrm{mono}}\mathcal{L}_{\mathrm{mono}}\\&+\lambda_{\mathrm{adv}}\mathcal{L}_{\mathrm{adv}}+\lambda_{\mathrm{fm}}\mathcal{L}_{\mathrm{fm}}\end{aligned}$$
> 阶段 1 的输入就是文本和其对应的 mel 谱，任务就是纯重构，不涉及任何其他花里胡哨的操作，目的应该是为了解耦。

#### 阶段 2 目标函数

Duration prediction：采用 L1 损失训练 duration predictor：
$$\mathcal{L}_\mathrm{dur}=\mathbb{E}_d\left[\left\|d-d_\mathrm{pred}\right\|_1\right]$$
其中 $d$ 为 GT duration（对 $a_{\mathrm{align}}$ 在时间维度上求和，$a_{\mathrm{align}}$ 是阶段 1 得到的那个对齐）。${d_\mathrm{pred}}=L(R(\boldsymbol{h_\mathrm{text}},\boldsymbol{s}))$ 为从 style $\boldsymbol{s}$ 和 文本 中预测的 duration。

Prosody prediction：采用一种新的数据增强方法来训练 prosody predictor。具体来说，不直接使用 原始 mel 谱 的 GT alignment、pitch、energy。首先采用 1-D 双线性插值对 mel 谱 在时域进行数据增强（拉伸或者压缩）得到 $\tilde{x}$，这个过程中语音的速度改变了，但是 pitch 和 energy 不变。从而 prosody predictor 可以在预测时学习这种不变性。
> 也就是，prosody predictor 可以学会预测 pitch 和 energy 但是不受到 duration 的影响。

采用 F0 和 energy 重构损失：
$$\begin{gathered}
\mathcal{L}_{\text{f0}} =\mathbb{E}_{p_{\boldsymbol{x}}}\left[\left\|p_{\boldsymbol{\tilde{x}}}-P_{p}\left(S\left(\boldsymbol{h}_{\mathrm{text}},s\right)\cdot\boldsymbol{\tilde{a}}_{\mathrm{align}}\right)\right\|_{1}\right] \\
\mathcal{L}_{\text{n}} =\mathbb{E}_{\tilde{\boldsymbol{x}}}\left[\left\|n_{\tilde{\boldsymbol{x}}}-P_{n}\left(S\left(\boldsymbol{h}_{\mathrm{text}},s\right)\cdot\boldsymbol{\tilde{\boldsymbol{a}}}_{\mathrm{align}}\right)\right\|_{1}\right] 
\end{gathered}$$
其中 $p_{\boldsymbol{\tilde{x}}},\:n_{\boldsymbol{\tilde{x}}}\mathrm{~and~}\tilde{\boldsymbol{a}}_{\mathrm{align}}$ 为增强后的数据集中的特征。$P_p$ 为预测的 pitch，$P_n$ 为预测的 energy。

Decoder reconstruction：对于增强后的数据也有一个重构损失：
$$\mathcal{L}_{\mathrm{de}}=\mathbb{E}_{\boldsymbol{\tilde{x}},\boldsymbol{t}}\left[\left\|\hat{\boldsymbol{x}}-G\left(\boldsymbol{h}_{\mathrm{text}}\cdot\tilde{\boldsymbol{a}}_{\mathrm{align}},s,\hat{p},\hat{n}\right)\right\|_1\right]$$
这里的 $\hat{p},\hat{n}$ 为预测的 pitch 和 energy，而 $\hat{\boldsymbol{x}}=G\left(h_\mathrm{text}\cdot\tilde{\boldsymbol{a}}_\mathrm{align},s,\tilde{\boldsymbol{p}},\|\tilde{\boldsymbol{x}}\|\right)$ ，两者的区别其实就在于 pitch 和 energy 上。
> 注意：
> 1. 这个 loss 的目的不是训练 generator $G$ 的！而是用来确保 pitch predictor 能够生成合适的 pitch 和 energy。
> 2. $\|\tilde{\boldsymbol{x}}\|$ 其实就是 $\tilde{\boldsymbol{n}}$。

阶段 2 总的目标函数为：
$$\min_{S,L,P}\:\mathcal{L}_\text{de}+\lambda_\text{dur}{ \mathcal{L}_\text{dur}} + \lambda _ { f 0 }\mathcal{L}_{\text{f}0}+\lambda_n\mathcal{L}_n$$

## 实验（略）

