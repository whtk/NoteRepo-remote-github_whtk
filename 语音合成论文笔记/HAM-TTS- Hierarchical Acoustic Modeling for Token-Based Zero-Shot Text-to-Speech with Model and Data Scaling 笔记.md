> preprint 2024.09，吉利、NII、上交

1. Token-based TTS 存在发音准确性低、说话风格和音色不一致、训练数据需求大等问题
2. 本文引入 hierarchical acoustic modeling 方法 + 数据增强策略，将真实和合成数据用于训练，数据规模 650k 小时，参数量 0.8B，实现 zero-shot TTS
    1. 基于 SSL 提取 discrete units，将其作为 latent variable sequence 引入 TTS 模型
    2. 训练时，通过替换和复制数据片段增强音色一致性
    3. 采用预训练的 few-shot voice conversion 模型生成多种音色的声音，学习 utterance-level one-to-many mappings，丰富语音多样性并确保音色一致性
3. 实验结果表明，模型在发音准确性、说话风格和音色连续性方面优于 VALL-E

## Introduction

1. 之前的 TTS 采用 MFCC 等连续 acoustic features 作为中间表征，由于语义和声学信息混合、难以解耦，导致 zero-shot 场景下音色生成质量低
2. token-based TTS 很火，一般采用 codec 的特征作为中间表征，但是很难保持发音准确性、说话风格和音色的连续性，而且要大量的数据
3. 本文提出 HAM-TTS，采用 hierarchical acoustic modeling 方法 + 数据增强策略
    1. 引入 LVS（latent variable sequence），包含 HuBERT 特征，来提供 acoustic 信息，减少发音错误
    2. 训练时，同时优化 Text-to-LVS predictor 和 TTS 模型
    3. 推理时，通过 predictor 将文本转为 LVS，提供重要的 acoustic 信息
4. 但是，HuBERT 特征中包含个性化信息，会影响说话风格的一致性，所以用 K-Means 聚类方法去除个性化信息，保持一致的说话风格
5. 还设计 音色一致性数据增强策略：
    1. 用来自其他训练 utterances 的小片段替换训练样本片段，或者复制训练样本的连续片段，强制模型预测原始 utterance
6. 还采用预训练的 few-shot VC 模型生成相同内容不同音色的声音，作为补充数据集，提高语音的多样性和音色一致性
7. 在大规模中文数据集上训练多个模型，用 AISHELL1 数据集评估，HAM-TTS 在发音准确性、说话风格一致性和音色连续性方面优于 VALL-E

## 相关工作（略）

## HAM-TTS

结构如图：
![](image/Pasted%20image%2020240926113414.png)

模型包含：
+ phoneme conversion
+ codec encoder
+ predictor：将文本 prompt 转为 LVS，提供额外 acoustic 信息
    + 训练的时候，和 TTS 模型一起优化

### Hierarchical Acoustic Modeling

AudioLM 和 VALL-E 有时会产生发音错误的语音
> 作者认为，是由于直接从文本到 codec sequence 的映射缺乏 acoustic 信息。

提出 Text-to-LVS predictor，如图：
![](image/Pasted%20image%2020240926150319.png)

从 phoneme sequence 生成 acoustic 信息：
$$\boldsymbol{L}_{1:T_1}^{\prime}=f_{pred}(\boldsymbol{X}_{1:T_1}),$$
其中 $\boldsymbol{X}_{1:T_1}$ 表示长度为 $T_1$ 的 phoneme sequence，$f_{pred}(·)$ 是 predictor，$\boldsymbol{L}_{1:T_1}^{\prime}$ 是生成的 LVS，长度与 phoneme sequence 相同。LVS 和 phoneme sequence 拼接后，通过卷积层：
$$\boldsymbol{S}_{1:T_1}=\text{Conv}1\text{d}(\text{Concat}(\boldsymbol{X}_{1:T_1},\boldsymbol{L}_{1:T_1}^{\prime})),$$
其中 $\boldsymbol{S}_{1:T_1}$ 的维度和 audio codecs 一致。

Text-to-LVS predictor 和 neural codec LM 通过 Text-HuBERT aligner 的输出来进行训练。aligner 包含 $N$ 个 block，每个 block 包含 $M$ 个 ResNet Block，RMSNorm 层和 multi-head attention 层，用于将 RMSNorm 输出和经 K-Means 聚类的 HuBERT 特征对齐。LVS 的长度和 phoneme sequence 相同，计算为：
$$\boldsymbol{L}_{1:T_1}=f_{aligner}(\boldsymbol{X}_{1:T_1},\boldsymbol{H}_{1:T_2}),$$
其中 $\boldsymbol{H}_{1:T_2}$ 是经过 K-Means 聚类的 HuBERT 特征序列，得到的$\boldsymbol{L}_{1:T_1}$ 就是 supervising LVS。
> K-Means 聚类用于去除 HuBERT 特征中的个性化信息，保持 zero-shot 场景下说话风格的一致性。

用 L1 loss 计算 $\boldsymbol{L}_{1:T_1}$ 和 $\boldsymbol{L}_{1:T_1}^{\prime}$ 差异：
$$\mathcal{L}_{LVS}=\sum_{t=1}^{T_1}|\boldsymbol{L}_t'-\boldsymbol{L}_t|,$$


### Timbre Consistency Data Augmentation

采用数据增强来保证音色一致性：
+ 在加载一个 batch 的数据时，对 10% 的数据：
    1. 从另一个样本中随机选择连续片段替换当前样本的片段
    2. 随机复制当前样本的片段并连接到末尾
+ 在 loss 计算中，把没有数据增强的 codecs 作为 ground truth，计算交叉熵损失
> 从而使得模型可以抵抗音色的扰动，防止短期音色变化影响整个生成的语音片段的音色，确保生成语音的音色一致性。

### Supplmentary Synthetic Dataset

采用预训练的 UNet-based few-shot VC 模型生成大量 长语音数据，提高训练数据的多样性：
+ 从 1,000 个真实的 speaker 中随机选择几分钟的语音，每个人生成 10-20 秒的语音，得到 500 小时的数据
> 这样的大量合成数据提高了训练数据的多样性，为长语音提供了一对多的映射，不同于之前只考虑 phoneme-level diversity。

### 损失函数

采用 VALL-E 的训练策略，将 TTS 视为条件 codec 语言建模任务，训练两个 Transformer decoder-only codec 语言模型，分别用于自回归（AR）和非自回归（NAR）建模：
$$\mathcal{L}_{codecs}=\sum_{t=1}^{T_3}\mathrm{CE}(\boldsymbol{A}_t,\boldsymbol{A}_t^{\prime}),$$
其中 $\boldsymbol{A}$ 和 $\boldsymbol{A}^{\prime}$ 分别表示 GT 和合成的 codec 序列，$T_3$ 表示 codec 序列的长度，$\mathcal{L}_{codecs}$ 是 codec 生成的损失。

为了提高 HAM-TTS 处理语义信息的能力，计算 teacher forcing loss：
$$\mathcal{L}_{phoneme}=\sum_{t=1}^{T_1}\mathrm{CE}(\boldsymbol{X}_t,\boldsymbol{X}_t^{\prime}),$$
其中 $\boldsymbol{X}^{\prime}$ 表示合成的 phoneme 序列，$\mathcal{L}_{phoneme}$ 是文本生成的损失。

总损失为三个损失项之和：
$$\mathcal{L}=\mathcal{L}_{LVS}+\mathcal{L}_{phoneme}+\mathcal{L}_{codecs}$$

## 实验

### 实验设置


数据集：
+ 训练集：内部中文语音数据集，包含 150k 小时真实语音和 500k 小时合成语音。
+ 测试集：在 AISHELL1 数据集中选择 50 个说话者，每个说话者有 5 个句子，持续时间在 5-20 秒之间。

Baseline：VALL-E：
+ VALL-E 是 token-based TTS 系统的代表性 SOTA 工作，所以作为 baseline
+ 没有官方实现，所以在内部数据集上复现和训练

评估指标：
+ 发音准确性：CER
+ 说话风格一致性：NMOS
+ 音色一致性：SMOS
+ 总体质量：MOS

### 结果

总体结果如图：
![](image/Pasted%20image%2020240926151657.png)


结论：
+ HAM-TTS-S 的 CER 为 4.0%，NMOS 为 3.79，SMOS 为 4.12，优于 VALL-E；
+ HAM-TTS-L 的 CER 为 3.2%，NMOS 和 SMOS 与 GT 相当，说明 HAM-TTS 在发音准确性和说话风格一致性方面的优势。

消融实验见论文。
