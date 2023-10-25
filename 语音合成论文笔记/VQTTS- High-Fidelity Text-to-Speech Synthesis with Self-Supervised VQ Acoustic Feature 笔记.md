> interspeech 2022，上海交大，人工智能实验室，俞凯

1. 提出 VQTTS，包含  AM txt2vec  和 vocoder vec2wav，采用自监督的 vector-quantized(VQ) 声学特征而非 mel 谱
2. 其中 txt2vec 是一个分类模型（而非传统的回归模型）
3. vec2wav 使用一个额外的 feature encoder + HiFiGAN generator
4. 可以在自然度方面实现 SOTA 的性能

## Introduction

1. 提出 VQTTS，txt2vec 只需要考虑时间轴上的相关性（而不需要预测在时间和频率上都高度相关的 mel 谱）

## Self-Supervised VQ Acoustic Feature

自监督的 VQ 特征很多已经被用在 ASR 中，如 Vq-wav2vec、wav2vec 2.0。

## VQTTS

已有工作表明，从 VQ acoustic feature 重构波形需要额外的韵律特征。

本文采用三维韵律特征 log pitch, energy and probability of
voice(POV)，然后归一化到 0 均值和单位方差。
> 下面将 VQ 和韵律特征合起来称为 VQ&pro。

VQTTS 包含两个部分：
+ 声学模型 txt2vec 从 phoneme 序列中预测 VQ&pros 
+ vocoder vec2wav 中 VQ&pros 中得到波形

### txt2vec

首先为所有的 phoneme 标上 phoneme-level（PL） 的 prosody。txt2vec 总体架构如下：
![](image/Pasted%20image%2020231016150132.png)
text encoder 包含 6 个  Conformer blocks，将输入 phoneme 编码为 hidden states $\mathbf{h}$，然后送到 PL prosody controller 用于预测 prosody label，通过 duration predictor 得到的 duration 进行拓展。decoder 包含 3 个  Conformer blocks，然后输出通过 LSTM + softmax 激活，用于实现 VQ acoustic feature classification。然后把 decoder 的输出和 VQ acoustic feature 拼接，通过 4 层卷积+layer norm+dropout。

phoneme duration 和 prosody feature 分别采用 L2 和 L1 损失训练，VQ acoustic feature 采用交叉熵损失，总损失如下：
$$\mathcal{L}_{\mathrm{txt2vec}}=\mathcal{L}_{\mathrm{PL\_lab}}+\mathcal{L}_{\mathrm{dur}}+\mathcal{L}_{\mathrm{VQ}}+\mathcal{L}_{\mathrm{pros}}$$

#### Phoneme-level prosody labelling

3 维归一化的 prosody feature 记为 $\mathbf{p}$，然后分别计算 $\Delta\mathbf{p},\Delta^2\mathbf{p}$，最终得到 9 维的特征 $[\mathbf{p},\Delta\mathbf{p},\Delta^2\mathbf{p}]$，然后在所有的帧上计算平均，最终每个 phoneme prosody 都得到一个向量。然后采用 k-means 进行聚类得到 $n$ 类，把聚类中心的 index 作为 PL prosody label。

PL prosody controller 结构如下：
![](image/Pasted%20image%2020231016151423.png)

其训练用于从 text encoder 的输出 $\mathbf{h}$ 中用 LSTM 预测 PL prosody labels。

推理的时候，这两个 LSTM 都采用 beam search decoding 的方法。

### vec2wav

vec2wav 模型架构如图：
![](image/Pasted%20image%2020231016153537.png)
VQ acoustic feature 和 prosody feature 分别通过卷积层，然后拼接之后再通过一个卷积层，然后通过 feature encoder 和 HifiGAN generator。feature encoder 用于 smoothing 不连续的 VQ acoustic feature，HifiGAN generator 用于生成最后的波形。

发现从 0 开始训练 vec2wav 很难收敛（在仅有 HiFiGAN loss 情况下），于是提出  multi-task warmup 的技巧，从 feature encoder 的输出中额外采用一个线性层来预测 mel 谱，此时损失函数变成：
$$\mathcal{L}_{\text{vec}2\text{wav}} = \mathcal{L}_{\text{Hi fi}\mathrm{GAN}}+\alpha\mathcal{L}_{\mathrm{mel}}$$
warmup 之后，移除掉这个任务，即令 $\alpha=0$。

## 实验和结果（略）

这里的 VQ-acoustic feature 是怎么来的呢？
用的是 预训练的 k-means-based vq-wav2vec 模型来提取 VQ acoustic feature。