> InterSpeech 2023，Yonsei University

1. 有些文章采用 latent representation 而非 mel 谱 作为中间特征，但是生成不行
2. 本文提出在 latent representation 添加 prosody embedding 来提高性能；训练的时候从mel 谱中提取 prosody embedding，推理时则采用 GAN 从文本中预测

## Introduction

1. 提出 AILTTS，single stage 的 轻量化 TTS 模型，可以提供语音合成中的 speech variance 信息：
	1. 采用 prosody encoder，输入 mel 谱，提取 prosody-related 特征，称为 prosody embedding
	2. 把 prosody embedding 作为训练过程中的条件
	3. 还有一个 prosody predictor 从文本中预测 embedding
	4. 在 prosody predictor 中采用 GAN 来增强估计能力

## 相关工作（略）

本文的 backbone [LiteTTS- A Lightweight Mel-spectrogram-free Text-to-wave SynthesizerBased on Generative Adversarial Networks 笔记](LiteTTS-%20A%20Lightweight%20Mel-spectrogram-free%20Text-to-wave%20SynthesizerBased%20on%20Generative%20Adversarial%20Networks%20笔记.md) 虽然参数和计算量更少，但是合成质量不行，于是采用 conditional discriminator 和 对抗训练。

本文还将 prosody embedding 引入 内部 aligner，相比于传统的对齐，更鲁棒也更快速。

## 方法

### 概览

下图为采用 LiteTTS 作为 backbone 的框图：
![](image/Pasted%20image%2020240122105834.png)

包含：
+ phoneme encoder
+ prosody encoder (posterior)
+ prosody predictor (prior)
+ 内部 aligner，包含 duration predictor
+ auxiliary predictor
+ vocoder

整个训练过程为：
+ 首先把 phoneme encoder 的输出 $h_{ph}$ 作为 Q，把 prosody encoder 的输出作为 K 和 V，通过 attention 计算 phoneme-level 的 prosody embedding $h_{pr}$ 
+ 然后把 $h_{ph}+h_{pr}$ 通过 internal aligner 进行对齐，得到对齐后的 embedding 为 $I$
+ 最后 $I$ 通过 vocoder 得到波形

为了降低架构的复杂度，采用 LiteTTS 中的轻量化的 transformer-based encoder 作为这些结构：phoneme encoder、prosody encoder、prosody predictor

### Prosody predictor with conditional discriminator

![](image/Pasted%20image%2020240122153602.png)

prosody predictor 从 phoneme embedding $h_{ph}$ 中预测 prosody embedding $h_{pr}$，用的是包含各种判别器的生成模型，generator 就是 prosody predictor，discriminator 则用于区分 target prosody embedding $h_{pr}$ 和预测的 $\tilde{h}_{pr}$，采用  projection-based conditional discriminator 来把 phonetic embedding $h_{ph}$ 作为 condition，同时还采用 feature mapping loss 辅助训练。

在设计 discriminator 时 还用了两个额外的技巧：
+ discriminator 用于区分对齐后的 prosody embedding 而非原始的 phoneme-level 的 prosody embedding
+ 设置一个和 vocoder 有着相同感受野的 discriminator，从而使得 discriminator 可以捕获 prosody embedding 的多样化的特征

整个 prosody predictor 的损失包括对抗损失、重构损失 和 feature mapping 损失：
$$\begin{gathered}
\mathcal{L}_{G}=\mathbb{E}_{(\tilde{H}_{pr},H_{ph})}[(D(H_{pr},H_{ph})-1)^{2}]+\mathcal{L}_{recon}+\mathcal{L}_{fm},\:(1) \\
\mathcal{L}_{D}=\mathbb{E}_{(H_{pr},\tilde{H}_{pr},H_{ph})}[(D(H_{pr},H_{ph})-1)^{2}+(D(\tilde{H}_{pr},H_{ph}))^{2}], \\
\mathcal{L}_{recon}=||\tilde{H}_{pr}-H_{pr}||_1,\:\mathcal{L}_{fm}=\sum_{i=1}^7||\tilde{F}_{pr}^i-F_{pr}^i||_1,\:(2) 
\end{gathered}$$
其中 $H()$ 表示 mel 谱 尺度下的 embedding，$F$ 为 feature map。

### Prosody-conditioned internal aligner

采用 likelihood-based internal aligner，即 [One TTS Alignment To Rule Them All 笔记](../对齐/One%20TTS%20Alignment%20To%20Rule%20Them%20All%20笔记.md) 中，最大化单调对齐的似然。然后选择最可能的路径来提取 duration，最后还有一个 KL 散度来确保 soft 和 hard 对齐尽可能相似。

其输入为 joint embeddings $h_{ph}+h_{pr}$，相比于只使用 $h_{ph}$，加上 $h_{pr}$ 之后包含更多的 local acoustic information，从而能够更简单地学习到 duration。

### 总训练损失

总损失定义如下：
$$\mathcal{L}_{total}=\mathcal{L}_{var}+\mathcal{L}_{align}+\mathcal{L}_{pred}+\mathcal{L}_{voc}+\mathcal{L}_{aux}$$
其中 $\mathcal{L}_{var}$ 包含 pitch and energy prediction 损失，$\mathcal{L}_{align}$ 为 aligner 的损失（包含 duration predictor 的损失），$\mathcal{L}_{pred}$ 为 prosody prediction 损失，$\mathcal{L}_{voc}$ 为 vocoder 损失，$\mathcal{L}_{aux}$ 定义为目标 mel 谱和预测 mel 谱之间的 L1 损失。

最后，图中的 auxiliary predictor 输入为 $I$，结构和上图 discriminator 中的 * 号的部分一致，但是：
+ 输出 channel 数是 mel 谱的维度
+ 用了 Layer normalization

仅在训练的时候使用，用于为 vocoder 提供声学信息。

## 实验（略）
