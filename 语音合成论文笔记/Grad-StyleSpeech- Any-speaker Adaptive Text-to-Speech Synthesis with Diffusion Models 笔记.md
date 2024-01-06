> ICASSP 2023，AITRICS，KAIST

1. 现有的 any-speaker adaptive TTS 的性能不好，因为模仿目标说话人的风格是次优的
2. 提出 Grad-StyleSpeech，是基于 diffusion 的任意说话人的 adaptive TTS

> 没有创新，学术裁缝

## Introduction

1. 提出 Grad-StyleSpeech，采用 score-based diffusion model 从任意说话人的几秒的参考语音中合成高质量的语音
2. 采用 style-based generative model 来考虑 target speaker 的风格
3. 提出 hierarchical transformer encoder 来生成representative 的 先验噪声分布

## 方法

给定文本 $\boldsymbol{x}=[x_1,\ldots,x_n]$ 和参考语音 $\boldsymbol{Y}=[y_1,\ldots,{y}_m]\in\mathbb{R}^{m\times 80}$，目标是生成 GT 语音 $\tilde{\boldsymbol{Y}}$，训练阶段，$\tilde{\boldsymbol{Y}}$ 和 $\boldsymbol{Y}$ 是一样的，推理的时候不一样。

模型包含三个部分：
+ mel style encoder，将参考语音编码到 style vector
+ hierarchical transformer encoder，基于文本和 style vector 生成表征
+ diffusion 模型，通过降噪过程来生成 mel 谱

如图：
![](image/Pasted%20image%2020231207095747.png)

### Mel-Style Encoder

采用 [Meta-StyleSpeech- Multi-Speaker Adaptive Text-to-Speech Generation 笔记](Meta-StyleSpeech-%20Multi-Speaker%20Adaptive%20Text-to-Speech%20Generation%20笔记.md) 中的 mel-style encoder ，最终得到 $\boldsymbol{s}=h_\psi(\boldsymbol{Y})$，其中 $s\in\mathbb{R}^{d^{\prime}}$。

### Score-based Diffusion Model

用的是 [Grad-TTS- A Diffusion Probabilistic Model for Text-to-Speech 笔记](Grad-TTS-%20A%20Diffusion%20Probabilistic%20Model%20for%20Text-to-Speech%20笔记.md) 中的方法，用 SDE 来表示降噪过程。

Grad-TTS 中，提出从数据驱动的先验分布 $\mathcal{N}(\mu,I)$ 中做降噪。这里的 $\mu$ 是从神经网络中提取的 text- 和 style- condition representation。

> 完全和 Grad-TTS 一致。

### Hierarchical Transformer Encoder

先用多个 transformer block 形成的 text encoder 将文本映射到表征 $H\:=\:f_\lambda(\boldsymbol{x})\:\in \mathbb{R}^{n\times d}$ ，然后用 [One TTS Alignment To Rule Them All 笔记](对齐/One%20TTS%20Alignment%20To%20Rule%20Them%20All%20笔记.md) 中提出的无监督对齐学习框架来计算对齐，然后根据对齐来调整长度 $\operatorname{Align}(\boldsymbol{H},\boldsymbol{x},\boldsymbol{Y})=\tilde{\boldsymbol{H}}\in\mathbb{R}^{m\times d}$。然后用 duration predictor 来预测每个 phoneme 的 duration，最后，把调整后的序列 通过 style-adaptive transformer 模块来得到 speaker-adaptive hidden representations $\boldsymbol{\mu}=g_\phi(\tilde{\boldsymbol{H}},\boldsymbol{s})$。用的是 Meta-style speech 中提出的  Style-Adaptive Layer Normalization (SALN) 方法将 style 耦合进去。

