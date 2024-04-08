> interspeech 2023 workshop，俄罗斯，VK 公司

1. TTS 中的情感建模很重要
2. 提出 EmoSpeech 本文在 FastSpeech2 的基础上提出了一系列修改，用于合成情感语音，在 MOS 和情感识别准确率方面超过现有模型
3. 包含条件机制，处理文本中情感的不均匀分布，使得情感以不同强度水平贡献到每个音素

> 创新点：用已有模型得到情感 embedding，然后用  Conditional Layer Norm (CLN) 引入情感，还有一个所谓的 conditional cross attention，在 attention 机制中引入情感。

## Introduction

1. 本文关注高负载环境，选择 FastSpeech2 而非 diffusion 作为研究起点
2. 主要贡献：
    1. 提出了 FastSpeech2 模型架构的扩展，可以合成情感语音
    2. 模型在 MOS 和情感识别准确率方面优于现有的 FastSpeech2 扩展模型，而不会带来推理速度延迟
    3. 提出了一种条件机制，可以考虑语音语调和情感强度之间的关系，EmoSpeech 根据情感关注句子的每个部分

## 相关工作

ETTS 方法可以可以根据用的条件数据的性质分为三类：
1. 使用分类标签表示一个或多个情感
2. 使用具有所需情感状态的参考语音
3. 使用目标情感状态的文本描述作为条件

第一种方法通常用于处理带标签的数据集，因为它可以通过引入嵌入查找表来简单地实现条件。本文主要关注快速 ETTS 模型，可以在给定说话人和情感下合成语音，因此采用第一种方法。

传统的合成情感语音的方法是在 Tacotron2 上构建，将分类标签与 pre-net 输出连接。

AdaSpeech 和 GANSpeech 是 FastSpeech2 较好的改进。

AdaSpeech 引入了 CLN（ Conditional Layer Norm (CLN)），用于计算 scale 和 bias：
$$y=\mathrm{f(c)}\cdot\:\frac{\mathrm{x-mean}}{\mathrm{var}}+\mathrm{f(c)}$$
其中 $\mathrm{f}$ 是线性层，$\mathrm{x}$ 是归一化隐藏状态，$\mathrm{c}$ 是条件。AdaSpeech 将编码器中的所有 layer normalizations 都替换为 CLN。AdaSpeech4 在  zero-shot 情况下，在编码器和解码器中集成 CLN 以获得更好的性能。

FastSpeech2 在多说话人设置下训练时会出现质量下降问题，GANSpeech 提出两阶段训练。一阶段使用重构损失训练 FastSpeech2，第二阶段使用 JCU 判别器进行对抗训练，包括有条件和无条件部分。FastSpeech2 和 JCU 鉴别器通过最小二乘目标进行优化：
$$\begin{aligned}
&L_{adv}(\mathrm{D})= \frac12\mathbb{E}_\mathrm{c}\left[D(\hat{\mathrm{x}})^2+D(\hat{\mathrm{x}},\mathrm{c})^2\right]  \\
&+\frac12\mathbb{E}_{(\mathrm{x},\mathrm{c})}\left[(D(\mathrm{x})-1)^2+(D(\mathrm{x},\mathrm{c})-1)^2\right] \\
&L_{adv}(\mathrm G)= \frac12\mathbb{E}_\mathrm{c}\left[\left(D(\hat{\mathrm{x}})-1\right)^2+\left(D(\hat{\mathrm{x}},\mathrm{s})-1\right)^2\right]. 
\end{aligned}$$
还用了 feature mapping loss 来计算真实和生成的 mel 谱的 JCU 鉴别器特征图之间的 L1 损失，以提高模型质量和稳定性：
$$L_{fm}(G,D)=\mathbb{E}_{\mathrm{x,c}}[\mathbb{E}_{\mathrm{l}}(D_l(\hat{\mathrm{x}},\mathrm{c})-D_l(\mathrm{x},\mathrm{c}))]$$

其中 $D_l$ 是 JCU 鉴别器层 $l$ 的输出。GANSpeech 训练如下：
$$L_{total}=L_{rec}+L_{adv}(D)+L_{adv}(G)+\alpha_{fm}\cdot L_{fm}$$

## 模型描述

### FastSpeech2

见 [FastSpeech 2- Fast and High-Quality End-to-End Text to Speech 笔记](../FastSpeech%202-%20Fast%20and%20High-Quality%20End-to-End%20Text%20to%20Speech%20笔记.md)。

FastSpeech2 为非自回归声学模型，输入为 token 序列，生成 mel 谱，然后通过 vocoder 上采样到波形。

包含 encoder、variance adapter 和 decoder。encoder 从文本提取特征， variance adapter 将声学和持续时间信息添加到输入，decoder 基于这些信息生成 mel 谱。

encoder 将 token 序列转换为 token representation $h\in\mathbb{R}^{n\times\text{hid}}$，其中 $n$ 和 $\text{hid}$ 分别是序列长度和维度。variance adapter 包含 3 个 predictor 和 length regulator。predictor 将 $h\in\mathbb{R}^{n\times\text{hid}}$ 作为输入，输出每个 token 的音高、能量和持续时间 $(p, e, d)$。length regulator 根据 $d\in\mathbb{R}^n$ 将 $h\in\mathbb{R}^{n\times\text{hid}}$ 累积为 $p, e\in\mathbb{R}^n$。length regulator 的输出是 $h\in\mathbb{R}^{m\times\text{hid}}$，其中 $m=\sum_{i=0}^n d_i$。得到的 hidden representation 通过 decoder，输出预测的 mel 谱 $y\in\mathbb{R}^{m\times c}$。通过重构损失学习从输入文本序列生成 mel 频谱图：
$$L_{rec}=||\mathrm{y}-\hat{\mathrm{y}}||+||\mathrm{d}-\hat{\mathrm{d}}||^{2}+||\mathrm{e}-\hat{\mathrm{e}}||^{2}+||\mathrm{p}-\hat{\mathrm{p}}||^{2},$$

其中 $\hat{\mathrm{y}},\hat{\mathrm{d}},\hat{\mathrm{e}},\hat{\mathrm{p}}$ 是预测的 mel 频谱图、持续时间、音高和能量。

### Conditioning Embedding

使用 embedding lookup tables 从 FastSpeech2 构建 EmoSpeech，作为 speaker 和 emotion 条件。将 speaker 和 emotion embedding 拼接得到 $c$。把它添加到 encoder 的输出中，得到的结果输入到 variance adaptor。

### eGeMAPS Predictor

FastSpeech2 的 variance adaptor 可以通过添加额外的 predictor 进行扩展。EmoSpeech 中，添加 eGeMAPS predictor (EMP) 到 variance adaptor，预测来自 eGeMAPS 的 k 个特征。
> eGeMAPS 一共有 88 个特征，选择其中两个特征进行预测，为 80th and 50th percentile of logarithmic F0。

EMP 的目的是为了给 utterance 添加更多与目标情感相关的语音描述信息。EMP 与 pitch 和 energy predictor 具有相同的架构，但是在 utterance level 而不是 token level。

### Conditional LayerNorm

将 CLN用到 EmoSpeech 的 encoder 和 decoder block 中，发现比传统的 Layer Norm 效果更好。将来自 embedding lookup tables 的 speaker 和 emotion embedding 拼接得到 $c$。

### Conditional Cross Attention

情感语音的一个特征是 expressive intonation。有时说话人会强调句子的某些部分，使得情感更加明显。传统的方法是给每个文本 token 添加相同权重的情感嵌入。这里引入了 Conditional Cross-Attention (CCA) block 到 encoder 和 decoder，根据给定的情感重新调整 token 的权重。

将 speaker 和 emotion embedding 拼接得到 $c$，将 Self-Attention 输出记为 $h\in\mathbb{R}^{n\times\text{hid}}$，CCA 使用 Self-Attention 层的 $W_q,W_k,W_v$ 矩阵，得到 $Q = W_q\cdot h, K = W_k\cdot c, V = W_v\cdot c$，然后重新调整 hidden：
$$\mathrm{w}=\mathrm{softmax}(\frac{\mathrm{Q}\cdot\mathrm{K}^T}{\sqrt{d}},\mathrm{dim}=1)\\\mathrm{cca}=\mathrm{w}\cdot V$$
这个操作可以看作为每一层添加一个独特的情感 token。

在 CCA 中也用了 multi-head。CCA 可以替换将 conditioning embedding 拓展到序列长度后添加到 encoder 输出的操作。因此在 EmoSpeech 中不再进行这个操作。

### 对抗训练

使用 GANSpeech 的方法进行对抗训练，而且 JCU 判别器的条件架构适用于这里的多说话人和多情感设定。用的 GANSpeech 相同的架构设置和训练目标，但是在只有一个训练 stage 来训练判别器和 EmoSpeech。

## 数据预处理（略）

## 实验和结果（略）
