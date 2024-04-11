> ICASSP 2024，Indian Institute of Science

1. audio-to-audio (A2A) 风格迁移用于将 source audio 的风格特征替换为 target audio 的风格特征，同时保留 source audio 的内容
2. 提出  Zero-shot 情感风格迁移 (ZEST)，将语音解耦为 semantic token、speaker representation 和 emotion embedding，训练一个框架，其给定语音信号来重构 pitch contour，然后训练一个 decoder 重构语音信号：
    1. 采用基于自监督的 reconstruction loss 训练模型
    2. 在转换过程中，emotion embedding 从 target audio 中获取，其他特征从 source audio 中获取
3. 实验表明，在没有平行训练数据下，模型也可以实现 zero shot 情感迁移

> 本质来说就是解耦+重构，训练的时候是简单的解耦+重构目标，推理的时候是解耦+替换+重构。然后解耦成三个部分：语义、说话人信息、情感信息。

## Introduction

1. 风格迁移是将 source sample 的情感转换为 target sample 的情感风格，同时保留 source 的其他属性
2. 语音转换主要是转换说话者的身份，但语音也包含说话者的情感特征
3. 之前的情感转换方法中，emotion targt 被视为离散的 label，但是其实 emotion 是一个细粒度的属性，强制将 emotion 属性转换为离散 label 可能无法捕捉到人类语音的多样化情感，作者认为自然的情感转换是将 target audio 中的情感转移到 source audio 中，即 A2A 情感风格迁移
4. speech 的 representation learning 取得了显著进展，wav2vec 模型通过 masked language modeling (MLM) 目标在自监督学习 (SSL) 设置中得到改进，speaker representation 主要通过监督模型进行，从 speaker、content 和 pitch contours 的因子表示中重构 speech 的模型表明 Tacotron、AudioLM 和 HiFi-GAN 可以生成高质量的 speech
5. 提出 ZEST 框架，将给定的 audio 解耦为：
    + semantic tokens（使用 HuBERT model）
    + speaker representations（x-vectors）
    + emotion embeddings（从预训练的 emotion classifier 中获取），且这 speaker 和 emotion 信息是解耦的
6. 由于 F0 也包含 content、speaker 和 emotion 信息，使用 cross-attention 预测给定 utterance 的 F0 contour
7. 有了 speech、speaker 和 emotion 的三个表征和 F0 contour 后，用 HiFi-GAN decoder 重构语音，训练过程中，不使用文本或平行训练，只从 target audio 中引入 emotion embedding 进行风格转移
8. 在 emotion speech dataset (ESD) 上进行实验，贡献如下：
    + 提出使用 HuBERT 的 semantic tokens、speaker embeddings 和 emotion embeddings 预测给定音频的 pitch contour
    + 使用对抗训练实现 speaker-emotion 解耦
    + 实现 zero shot 情感迁移，包括未知的情感类别、新的说话人和新的文本情况

## 相关工作

采用 World vocoder 的 A2A EST：早期使用 target speaker 的 F0 和 spectral components 的统计信息，这里则用了 speaker、emotion 和 content embeddings 进行情感风格迁移，目标是将 reference speech 的情感风格转移到 source speech 信号，而不仅仅是修改情感类别

表达性 TTS：有些工作用 speaker disentanglement 从文本生成情感语音，Emovox 使用 phonetic transcription 进行情感语音转换，这里则不使用 source 或 target speech 的任何语言或 phonetic transcription

非平行和未知情感转换：有使用 attention 模型进行 EST，但是要求 source 和 target speech 来自同一个说话者，限制了 EST 的应用

## 方法

### Content encoder

使用 [HuBERT- Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units 笔记](../../语音自监督模型论文阅读笔记/HuBERT-%20Self-Supervised%20Speech%20Representation%20Learning%20by%20Masked%20Prediction%20of%20Hidden%20Units%20笔记.md) 模型作为 content encoder，每个 speech segment 得到 vector representations，然后用 k-means 聚类转为离散 tokens。

### 情感无关的 speaker encoder

使用 [ECAPA-TDNN- Emphasized channel attention, propagation and aggregation in tdnn based speaker verification 笔记](../../声纹识别论文阅读笔记/ECAPA-TDNN-%20Emphasized%20channel%20attention,%20propagation%20and%20aggregation%20in%20tdnn%20based%20speaker%20verification%20笔记.md) 模型提取音频的 speaker embeddings（模型在 VoxCeleb 数据集上进行了预训练，会对 frame-level embeddings 进行 utterance level pooling，得到 x-vectors），为了 suppress emotion 信息，添加两个全连接层到 x-vector ，并使用 emotion 对抗损失进行训练，得到所谓的 Emotion Agnostic Speaker Encoder (EASE) vectors。损失函数为：
$$\large\mathcal{L}_{tot-spkr}=\mathcal{L}_{ce}^{spkr}-\lambda_{adv}^{emo}\mathcal{L}_{ce}^{emo}$$
> 这个损失使得模型可以区分不同的说话人，但是不同区分不同的情感。似乎和 GRL 梯度反转层的效果一致？

### Speaker Adversarial Classifier of Emotions(SACE)

基于 wav2vec2 表征设计 speaker adversarial classifier of emotions (SACE) 分类器，使用预训练的 wav2vec 模型提取原始语音波形的特征，其中的 CNN 特征提取器保持冻结，transformer 层和两个 position wise FF 层在 Emotional Speech Dataset (ESD) 进行情感识别训练，采用 speaker adversarial loss 进行训练，整个 utterance 的平均表征作为 emotion embedding。

### Pitch Contour Predictor

pitch (F0) 预测器框架如下：
![](image/Pasted%20image%2020240410163946.png)

语音信号的 HuBERT token 通过 embedding 层转为向量序列，记为 $Q=\{q_1,...,q_T\}_{t=1}^T$，作为 cross-attention 的 query，frame-level SACE embeddings 和 speaker embedding (EASE) 作为 cross attention 的 key-value 对（如图，两个特征拓展后相加），然后通过 1D-CNN 预测 F0 contour。GT F0 contour 使用 YAAPT 算法得到，使用预测和GT之间的 L1 loss 训练。

### 语音重构

语音重构框架如下图 a（即训练过程）：
![](image/Pasted%20image%2020240410164230.png)

用 HuBERT tokens、SACE embedding、EASE vector 和预测的 F0 contour 来进行重构。HuBERT tokens 通过可学习的 embedding 层转为向量序列。为了在重构阶段添加上下文信息，tokens 和预测的 F0 分别通过两个网络（CNN 和 BLSTM），最后通过 HiFi-GAN vocoder 重构。
> 具体怎么引入 HiFiGAN 的好像没有细说？

### 情感转换

情感转换框架如上图 b。流程如下：
+ 从 source speech 中提取 HuBERT tokens 和 speaker vector ，从 target speech 中提取 emotion embedding
> 由于 emotion embedding 序列从 reference speech 中提取，可能与 source speech 的长度不同，但是 cross-attention 的 query 来自于 source 语音的 HuBERT tokens 驱动，所以转换过程中生成的 F0 contour 与 source 语音的长度匹配。
+ HuBERT tokens、speaker vector、predicted F0 contour 和 emotion embedding 输入到预训练的 HiFi-GAN 来生成转换后的语音

## 实验和结果（略）
