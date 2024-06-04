> ICASSP 2022，Ubisoft

1. 本文重点在于用于 voice conversion 的自监督表征学习
2. 比较了 soft 和 discrete speech unit 作为输入特征，发现：
    1. discrete representations 能有效去除说话者信息，但会丢失一些内容，导致发音错误
3. 提出预测 discrete unit 上的分布来学习的 soft speech units，通过建模不确定性，soft units 可以捕获更多内容信息

> 用 soft unit 来预测 discrete unit，然后用 soft unit 通过 vocoder 来合成波形，原因在于，discrete unit 会丢失内容信息，soft 则可以保留一部分。
> 因为 soft 建模的相当于是分布，其包含的信息容量比离散的点多得多。

## Introduction

1. 典型的 VC 需要学习包含内容特征但是去除说话人相关的细节的特征，然后就可以替换说话人信息来合成目标语音
2. 最近的工作研究了用于 VC 的自监督表征学习，大多数研究集中在 discrete speech units 上
    1. 离散化可以将内容和说话者信息分开，但是也会丢失一些内容，导致转换后的语音发音错误
    2. 例如，单词 fin，摩擦音 /f/ 中的模糊帧可能被分配到附近的错误单元，导致发音错误 thin
3. 提出 soft speech units，通过训练网络预测 discrete speech units 上的分布。建模不确定性可以保留更多内容信息
4. 本文重点在于 any-to-one VC，在自监督 CPC 和 HuBERT 上比较了 discrete 和 soft speech units

## VC 系统

VC 系统包括三个组件，如图：
![](image/Pasted%20image%2020240604172010.png)

+ content encoder：从输入音频中提取 discrete 或 soft speech units
+ acoustic model：将 speech units 转换为 spectrogram
+ vocoder：将 spectrogram 转换为音频波形

### Content Encoder

discrete content encoder 包含特征提取 + k-means 聚类：
    1. 第一步可以用不同的特征，从 MFCC 到 CPC 或 HuBERT
    2. 第二步，聚类特征构建 discrete speech units 字典
    3. 从自监督模型聚类特征可以提高 unit 质量和 VC 效果
    4. 离散 content encoder 将 utterance 映射到离散 speech units 序列 $\langle d_1,\ldots,d_T\rangle$

soft content encoder 直接用特征，但是这些特征表示包含大量说话人信息，不适合 VC。因此，训练 soft content encoder 预测 discrete units 上的分布。

discrete units 强制去除说话人信息，为了准确预测 discrete units，soft content encoder 需要学习说话人无关的表征。而 speech sounds 空间不是离散的，离散化会导致内容信息的丢失。通过建模 discrete units 上的分布，保留更多内容信息。

训练过程如图 b，给定输入 utterance，提取 discrete speech units 作为 label，backbone 网络处理 utterance，线性层将输出投影到 soft speech units 序列。每个 soft unit 参数化一个 discrete units 字典上的分布：
$$p(d_t=i\mid\mathbf{s}_t)=\frac{\exp(\sin(\mathbf{s}_t,\mathbf{e}_i)/\tau)}{\sum_{k=1}^K\exp(\sin(\mathbf{s}_t,\mathbf{e}_k)/\tau)},$$

其中 $i$ 是第 $i$ 个离散单元的聚类索引，$e_i$ 是对应的可训练嵌入向量，$\sin(\cdot,\cdot)$ 计算 soft 和 discrete units 之间的余弦相似度，$\tau$ 是温度参数。最后，最小化分布和离散 targets 之间的平均交叉熵来更新 encoder。测试时，soft content encoder 将输入音频映射到 soft speech units 序列，然后送给 acoustic model。

### Acoustic Model 和 Vocoder

acoustic model 的输入是 speech units 而不是 graphemes 或 phonemes。acoustic model 将 speech units（discrete 或 soft）转换为目标说话人的 spectrogram，vocoder 将预测的 spectrogram 转换为音频。

## 实验设置

数据集：LJSpeech
任务：intra- 和 cross-lingual VC
    1. intra-lingual：LibriSpeech dev-clean set 作为 source speech
    2. cross-lingual：将英语系统应用到法语和南非荷兰语数据

比较 discrete 和 soft speech units，实现不同版本的 VC 系统：
    1. discrete content encoder：CPC 和 HuBERT 作为特征提取器
    2. soft content encoder：CPC 和 HuBERT 作为 backbone

CPC：用在 LibriLight unlab-6k set 上预训练的 CPC-big3
HuBERT：HuBERT-Base4，在 LibriSpeech-960 上预训练

+ discrete content encoder：对 CPC-big 或 HuBERT-base 得到的表征进行 k-means 聚类，100 个 clusters
+ soft content encoder：用 CPC-big 和 HuBERT-Base 作为 backbone，在 LibriSpeech-960 上微调，25k steps，学习率 2e-5
+ acoustic model 和 vocoder：基于 Tacotron 2，包括 encoder 和 autoregressive decoder

acoustic model 和 vocoder 在 LJSpeech 上训练。

baseline：
    1. AutoVC
    2. Cascaded ASR-TTS

## 实验结果

PER 和 WER 比较：
![](image/Pasted%20image%2020240604175339.png)
结论：soft speech units 相比 discrete units 的 WER 和 PER 更低。HuBERT- 和 CPC-Soft 都优于 baseline，接近 GT

说话人相似度和自然度：
![](image/Pasted%20image%2020240604175519.png)

说话人相似度：EER 50% 表示 ASV 无法区分真实和转换的 utterances。discrete units 得分接近完美，soft units 有小幅下降，但是 raw features 更差。

自然度：soft units 优于 discrete units。HuBERT-Soft 表现最好，自然度接近 GT。作者认为是 soft units 中编码的额外信息导致更自然的 prosody。

跨语言 VC：soft units 在未知语言上表现更好。但是，soft units 在跨语言设置中导致更大的说话人相似度下降，因为保留了更多的口音信息。