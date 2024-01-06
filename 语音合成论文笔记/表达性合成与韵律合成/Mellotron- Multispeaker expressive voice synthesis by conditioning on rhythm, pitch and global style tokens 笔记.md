> NVIDIA，ICASSP 2020

1. Mellotron 是一个基于 Tacotron 2 GST 的多说话人 语音合成模型，可以在没有情感数据的情况下，使得合成的语音带有情感
2. 通过显式地以来自音频或音乐的 rhythm and continuous pitch contours 为条件，Mellotron 可以生成多种风格的语音


## Introduction

1. 之前的论文只能实现 coarse control ，Mellotron 可以在 expressive characteristics 上实现 fine grained control 
2. Mellotron 在考虑了 melodic information 如 pitch 和 rhythm 时，expressive speech synthesis 可以拓展到  singing voice synthesis，且不需要任何 singing voice 数据集

## 方法

Mellotron 同时使用  explicit 和 latent variables。

将 mel 谱 分解为 explicit variables，如 text, speaker identity, a fundamental frequency contour augmented with voiced/unvoiced decisions，和 两个 latent variables，通过模型在训练时学习这些 variable。

第一个 latent variables 为 dictionary of vectors，可以通过 audio input 进行 query，和 [Style Tokens- Unsupervised Style Modeling, Control and Transfer inEnd-to-End Speech Synthesis 笔记](Style%20Tokens-%20Unsupervised%20Style%20Modeling,%20Control%20and%20Transfer%20inEnd-to-End%20Speech%20Synthesis%20笔记.md) 相似。第二个 latent variables 为 mel 谱 和 text 之间的 attention map，和 [Tacotron 2- Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions 笔记](../Tacotron%202-%20Natural%20TTS%20Synthesis%20by%20Conditioning%20WaveNet%20on%20Mel%20Spectrogram%20Predictions%20笔记.md) 相似。

下面把 augmented fundamental frequency contour 当作 pitch contour，把 第一个 latent variable 称为 GST，把第二个 latent variable 称为 rhythm。

将 $M$ 分解为 $M=[T,S,P,R,Z]$，其中 $T$ 表示文本，$S$ 表示speaker identity，$P$ 表示 pitch contour，$R$ 表示 rhythm，$Z$ 表示 GST。那么，训练的时候就需要最大化下式：
$$P(mel^{(i)}|T^{(i)},S^{(i)},P^{(i)},R^{(i)},Z_{mel^{(i)}};\theta)$$
上标 $i$ 表示第 $i$ 个 mel 谱，$Z_{mel^{(i)}}$ 表示基于 $mel^{(i)}$ 的 GST，$\theta$ 表示模型参数。

explicit factors 有两个好处：
1. 通过向模型提供文本和说话人信息，可以防止 文本和说话人信息之间的 entanglement
2. 通过为模型提供 pitch contour 和 voicing information，可以在推理过程中直接控制 pitch 和 voicing decisions。

latent factors 也有两个好处：
1. 通过学习 alignment map，不需要提取 phoneme alignments，在推理的过程中提供 alignment map 即可控制 rhythm
2. latent variables 使得模型学习很难表达或者很难显示提取的 latent factor，从而可以充分利用 latent variable 的潜力

通过上面的公式可以从 源音频中迁移 text、rhythm 和 pitch contour 到 target speaker，只需简单替换对应的值即可。具体来说，对于 source 中的 text、pitch 和 rhythm $T_s,P_s,R_s$，从 GST 中采样一个 $Z_{query}$，然后选择 target speaker $S_t$，则：
$$P(mel_{out}|T_s,P_s,S_t,R_s,Z_{query};\theta)$$
从而生成的 $mel_{out}$ 有和 source 相同的 text、pitch 和 rhythm，来自 GST 中的 latent characteristics，来自 target speaker 的 voice。
> This allows us to train a model that makes a voice emote and sing without using any singing voice in the training dataset, without any manual labelling of emotions nor pitch, and without any manual alignments between words and audio, nor between pitch and audio.

## 实现

### 架构

Mellotron 基于 speaker embeddings 和 pitch countour 拓展 Tacotron 2，采用 单个 speaker embedding，然后在  channel 维度对每个 encoder 的输出进行拼接，pitch contour 通过单层 CNN+ReLU。同样也是在 channel 维度进行拼接。

### 训练

只需要带有 speaker id 的 (text, audio) 对即可训练。采用 Yin 算法提取 pitch contours。

采用 L2 损失进行训练。

### 推理

推理时，需要提供 text, rhythm and pitch information （来自音频或音乐）、global style token 和 speaker id。

获取 text, rhythm and pitch information 包含三个部分：
1. 从音频中获取文本，有的话就直接用 transcribe，没有就用 ASR 提取，然后从 grapheme 转为 phoneme
2. 使用 forced-alignment tool 或 Mellotron 提取 rhythm information
3. 采用 Yin 算法提取 pitch 

## 实验（略）
