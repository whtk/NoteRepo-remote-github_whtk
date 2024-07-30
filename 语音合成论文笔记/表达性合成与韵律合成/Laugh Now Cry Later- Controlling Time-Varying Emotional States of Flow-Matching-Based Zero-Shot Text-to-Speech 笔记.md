> 2024 preprint，台大、微软

1. 人类改变语调时，通常伴随着 non- verbal vocalizations （NVs）如笑声和哭声，而大多数 TTS 很难生成
2. 提出 EmoCtrl-TTS，一个 emotion-controllable zero-shot TTS，可以生成带有 NVs 的高情感语音
3. EmoCtrl-TTS 使用 arousal 和 valence 值，以及 laughter embeddings 来调节 flow-matching-based zero-shot TTS
4. 使用 27000 小时的 expressive data 进行训练，可以在 speech-to-speech translation 场景中模仿 audio prompts 的情感，同时可以捕捉情感变化，生成各种 NVs

## Introduction

1. 提出 EmoCtrl-TTS，一个 emotion-controllable zero-shot TTS，可以生成带有 NVs 的高情感语音
2. 可以模仿 audio prompt 的 voice characteristics 和 emotion 来生成 speech
3. 基于 [Voicebox- Text-Guided Multilingual Universal Speech Generation at Scale 笔记](../Voicebox-%20Text-Guided%20Multilingual%20Universal%20Speech%20Generation%20at%20Scale%20笔记.md)，采用 valence 和 arousal 值来模仿 emotion 的 time-varying 特征
4. 使用 laughter embeddings，可以生成 laughter 和其他 NVs，包括 crying

## 相关工作

### TTS 中的情感控制
现有的带有情感能力的 TTS：
![](image/Pasted%20image%2020240726165414.png)

需要考虑以下几点：
1. TTS 系统是否可以控制 fine-grained emotional attributes
2. 是否可以生成 NVs
3. 训练数据的规模
4. 说话人数量
5. 训练数据是 staged data 还是 real data

### 基于 Flow-matching 的 TTS

1. Voicebox 是第一个使用 conditional flow matching 训练 zero-shot TTS 的模型
2. ELaTE 用于生成 natural laughing speech，但只测试了 laughing speech，没有测试其他 NVs

## EmoCtrl-TTS

### 概述

EmoCtrl-TTS 训练和推理过程如图：
![](image/Pasted%20image%2020240726170142.png)

给定带有文本 $y$ 的训练样本 $s$，提取 mel-filterbank 特征 $\hat{s} \in \mathbb{R}^{F \times T}$，其中 $F$ 表示特征维度，$T$ 表示序列长度。使用 force alignment 和 phoneme embedding 层来获得 frame-wise phoneme embedding $a \in \mathbb{R}^{D^{phn} \times T}$，其中 $D^{phn}$ 是 phoneme embedding 维度。frame-wise embeddings 表征 NV 为 $h \in \mathbb{R}^{D^{NV} \times T}$ 和 emotion $e \in \mathbb{R}^{D^{emo} \times T}$，其中 $D^{NV}$ 和 $D^{emo}$ 分别表示 NV 和 emotion embeddings 的维度。embeddings $h$ 和 $e$ 通过预训练的 NV 和 emotion detector 提取。使用 Voicebox 中的 speech infilling task 训练 audio model，训练 conditional flow-matching model 来估计分布 $P(m \odot \hat{s}|(1 - m) \odot \hat{s}, a, h, e)$，其中 $m \in \{0, 1\}^{F \times T}$ 是 binary temporal mask，$\odot$ 表示 Hadamard product。

推理时，模型接受四个输入：文本 prompt $y^{text}$，说话人 prompt 音频 $s^{spk}$，NV prompt 音频 $s^{NV}$ 和 emotion prompt 音频 $s^{emo}$。文本 prompt 表示生成语音的内容。同时，说话人、NV 和 emotion prompt 控制生成语音中说话人、NV 和 emotion 的特征。在 speech-to-speech translation 场景中，使用源音频作为 $s^{spk}$，$s^{NV}$ 和 $s^{emo}$，翻译后的文本作为 $y_{text}$。这样可以保留源说话人的声音和情感特征。

说话人 prompt $s^{spk}$ 首先转换为 mel-filterbank 特征 $\hat{s}^{spk}$。然后 ASR 和 phoneme embedding 层转换为 phoneme embeddings $a^{spk}$。说话人 prompt 还通过 NV detector 和 emotion detector 转换为 NV embeddings $h^{spk}$ 和 emotion embeddings $e^{spk}$。

文本 prompt $y^{text}$ 转换为 text prompt embeddings $a^{text}$，然后通过 phoneme embedding 层。NV prompt embedding $h^{NV}$ 和 emotion prompt embedding $e^{emo}$ 从 NV detector 和 emotion detector 提取。如果 $h^{NV}$ 和 $h^{emo}$ 的长度与 $a^{text}$ 不同，对 $h^{NV}$ 和 $h^{emo}$ 进行线性插值，使其长度与 $a^{text}$ 匹配。

基于 flow-matching 的模型根据 $P(\tilde{s}|[\hat{s}^{spk};z^{text}],[a^{spk};a^{text}],[h^{spk};h^{NV}],[e^{spk};e^{emo}])$ 生成 mel-filterbank $\tilde{s}$，其中 $z^{text}$ 是形状为 $F \times T^{text}$ 的全零矩阵，$[;]$ 表示时间维度上的连接。生成的 $\tilde{s}$ 通过 vocoder 转换为语音信号。

### NV embeddings

在 ELaTE 中，使用 off-the-shelf laughter detection 模型获得的 embedding 来控制 zero-shot TTS 中的 laughter。

发现 laughter detector-based embedding 实际上捕捉了比 laughter 更广泛的 NV 类型。通过适当使用 laughter-detector-based embedding，可以生成各种 NVs，如 crying 和 moaning。因此，在本文中，使用 laughter detection model 的 32 维 embedding 作为 NV embedding。

### Emotion embeddings

情感可以用两种主要方式表示：
+ 一是将情感分类为不同的情感类别，如 happiness 或 sadness，反映不同的情感状态；
+ 另一种是使用两个属性描述情感，arousal 和 valence，有时还有 dominance

Arousal 表示情感的强度或激活水平，从 calm 到 highly stimulated。Valence 表示情感的愉悦程度，从 very positive 到 very negative。Dominance 表示一个人对情况的控制程度。

最终，使用预训练的 arousal-valence-dominance extractor 预测的 arousal 和 valence 值作为 emotion embedding。extractor 使用 wav2vec 2 模型初始化，并在 MSP-PODCAST 数据 fine-tune，预测 arousal、valence 和 dominance values。使用 0.5 秒的窗口大小和 0.25 秒的 hop size 提取 chunk-wise arousal-valence values。因为 extractor 输出的值在 0.0 到 1.0 范围内，将估计值减去 0.5 调整范围为 -0.5 到 0.5。通过线性插值将提取的值的长度与 phoneme embedding 对齐。这样可以捕捉每个 utterance 中更微妙的情感变化。
> 作者发现，dominance value 的使用会降低音频质量，因此省略了 dominance value。

### 大规模情感数据获取

训练数据的数量和质量是实现高质量 TTS 的关键因素。然而，情感语音的录制或录音的手动标注成本高，使得难以将数据规模扩展到超过 100 小时。

本文从 200k 小时的 in-house unlabeled anonymized English audio 中筛选出 27k 小时的 highly emotional data，称为 In-house Emotion Data（IH-EMO）。数据筛选过程如下：
+ 使用 emotion2vec model 获得 predicted emotion confidence scores
+ 如果 predicted emotion 是 {angry, disgusted, fearful, sad, surprised} 或 predicted emotion 是 {neutral, happy} 且 confidence score 为 1.0，则保留样本
+ 应用 DNSMOS，保留 OVLR score 大于 3.0 的样本
+ 应用 in-house speaker change detection model，检测到 speaker change 时丢弃样本
最终收集了 27k 小时的 emotional audio。使用 off-the-shelf speech recognition model 获得 transcription。

## 实验

数据集：
+ Libri-light：用于 pre-training audio model，不包含 NV 和 emotion embeddings
+ LAUGH 和 IH-EMO：用于 fine-tuning audio model，包含 NV 和 emotion embeddings


