> interspeech 2024，Korea University

1. 现有研究主要是模仿情感的 average style，所以只能控制几个预定义的 label，无法反映情感的 nuanced variations
2. 本文提出 EmoSphere-TTS，使用 spherical emotion vector 控制合成语音的 emotional style 和 intensity
3. 可以在不需要 label 的情况下，使用 arousal, valence, 和 dominance pseudo-labels 来模拟 emotion
4. 提出一种 dual conditional adversarial network 来提高生成语音的质量

## Introduction

1. emotion 控制很难，因为同样的 emotion 可能有不同的表现
2. 一种常见的方法是通过 emotion labels 和 reference audio 控制 diverse emotional expressions
3. 一种方法是使用 emotional dimensions 来控制 emotional expression，可以实现更连续和精细的控制，但是很少有数据集提供这些 annotations
4. 本文提出 EmoSphere-TTS，使用 spherical emotion vector space 控制 emotional style 和 intensity：
    1. 使用从 SER 中的 pseudo-labeling 得到的 AVD
    2. 使用 Cartesian-spherical transformation 得到 spherical emotion vector space
    3. 提出 dual conditional adversarial training 来提高生成语音的质量

## EmoSphere-TTS

模型结构：
![](image/Pasted%20image%2020240918213832.png)

## Emotional Style and Intensity Modeling

包含两个部分：
+ AVD encoder
+ Cartesian-spherical transformation

### AVD encoder

采用 wav2vec 2.0-based SER 提取 continuous 和 detailed representations。得到 $e_{ki}\:=\:(d_a,d_v,d_d)$，其中 $d_a$ 代表 arousal，$d_v$ 代表 valence，$d_d$ 代表 dominance，每个值在 0 到 1 之间。$e_{ki}$ 代表第 $k$ 个 emotion 的第 $i$ 个坐标。
> 本质是一个三维的坐标。

### 笛卡尔-球面变换

将 AVD pseudo-labels 转换为 spherical coordinates，做了两个假设：
1. emotional intensity 随着离 neutral emotion center 越远而增加
2. 和 neutral emotion center 的角度决定 emotional style

首先，将 neutral emotion center M 设置为原点，得到转换后的 Cartesian coordinates $e'_{ki} = (d'_a, d'_v, d'_d)$：
$$e_{ki}'=e_{ki}-M\text{ where }M=\frac{1}{N_n}\sum_{i=1}^{N_n}e_{ni},$$
其中 $N_n$ 代表 neutral coordinates $e_{ni}$ 的总数。然后，将 Cartesian coordinates 转换为 spherical coordinates $(r, \theta, \phi)$：
$$r=\sqrt{{d'_a}^2+{d'_v}^2+{d'_d}^2},\\\vartheta=\arccos\left(\frac{d'_d}{r}\right),\varphi=\arctan\left(\frac{d'_v}{d'_a}\right).$$

在 Cartesian-spherical transformation 后，通过将 $r$ 缩放到 0-1 来 normalize emotion 的 intensity。使用 interquartile range technique 确定 scale 的最小和最大值。此外，通过将方向角 $\theta$ 和 $\phi$ 分成八个 octants 来量化 emotion style，每个 octant 由 A, V, D 轴的正负方向定义。
> 得到的 spherical emotion vector 也是一个三维的坐标。

### Spherical Emotion Encoder

使用 spherical emotion encoder 将 emotion ID 和 spherical emotion embedding 结合起来。首先，使用 projection 层将 emotion style vector 和 emotion class embedding 的维度对齐，然后 concatenate 这些 projections，通过 softplus activation 和 Layer Normalization 得到 spherical emotion embedding $h_{emo}$：
$$\mathbf{h}_{emo}=\text{LN}\left(\text{softplus}\left(\text{concat}\left(\mathbf{h}_{sty},\mathbf{h}_{cls}\right)\right)\right)\boldsymbol{+}\mathbf{h}_{int}.$$
其中 $\mathbf{h}_{sty}$，$\mathbf{h}_{int}$，$\mathbf{h}_{cls}$ 分别代表 emotional style vector，emotional intensity vector，emotion class embedding 的 projection layer 的输出。
> 注意，还是有 emotion class embedding 的，所以还是需要 emotion labels。

## Dual Conditional Adversarial Training

使用多个 discriminators 来提高 TTS 模型的质量。包含 stacked 2D-convolutional layers 和 全连接层。输入是 random Mel-spectrogram clip (Mel clip)。使用 emotion 和 speaker embeddings 来捕捉多方面的特征。一个 Conv2D stack 只接收 Mel clip，而其他的接收 condition embedding 和 Mel clip 的组合。将 condition embedding 扩展到与 Mel clip 的长度相匹配以进行 concatenation。损失函数如下：
$$\mathcal{L}_D=\sum_{c\in\{spk,emo\}}\sum_t\mathbb{E}[(1-D_t(y_t,c))^2+D_t(\hat{y}_t,c)^2],\\\mathcal{L}_G=\sum_{c\in\{spk,emo\}}\sum_t\mathbb{E}[(1-D_t(\hat{y}_t,c))^2],$$
其中 $y_t$ 和 $\hat{y}_t$ 分别代表 GT 和 生成的 mel 谱，$c$ 代表 condition。

### TTS Model

保留 FastSpeech 2 的结构和目标函数，使用 spherical emotion vector 提供 emotional style 和 intensity 信息。speaker ID 映射到 embedding $h_{spk}$，然后和 emotion embedding 拼接输入到 variance adaptor。在 inference 时，使用 manual style 和 intensity vectors 控制情感。

## 实验和结果

数据集：ESD

Vocoder：BigVGAN

模型：基于 FastSpeech 2 的 acoustic model，AVD encoder 使用 wav2vec 2.0 和 linear predictor，discriminator 使用 projection layers。