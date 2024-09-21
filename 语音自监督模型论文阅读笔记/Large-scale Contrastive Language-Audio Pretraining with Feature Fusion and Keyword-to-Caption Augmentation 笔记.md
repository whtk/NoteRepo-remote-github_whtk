> ICASSP 2023，Universite ́ de Montre ́al

1. 提出 contrastive language-audio pretraining，结合 audio data 和 natural language descriptions，得到音频 representation
    1. 发布 LAION-Audio-630K，633,526 个 audio-text pairs
    2. 构建 contrastive language-audio pretraining 模型，考虑不同的 audio encoders 和 text encoders
    3. 使用 feature fusion 和 keyword-to-caption augmentation，处理不同长度的 audio inputs，提高性能
2. 在 text-to-audio retrieval、zero-shot 音频分类和有监督音频分类三个任务上进行实验
3. 模型和代码开源

## Introduction

1. CLIP 学习 text 和 image 的 correspondence，不需要 data annotation，zero-shot 效果好；audio 和 natural languages 也有 overlapping information
2. 之前的研究的问题
    1. 训练数据集小
    2. 缺乏 audio/text encoders 的选择和超参数设置的研究
    3. transformer-based audio encoder 难以处理变长的 audio inputs
    4. 大多数研究只关注 text-to-audio retrieval，没有评估 audio representations 在 downstream tasks 的表现
3. 本文贡献：
    1. 发布 LAION-Audio-630K，633,526 个 audio-text pairs 数据集
    2. 构建 contrastive language-audio pretraining pipeline，选择两个 audio encoders 和三个 text encoders，使用 feature fusion 机制增强性能，处理变长输入
    3. 在 text-to-audio retrieval、zero-shot 和 supervised 音频分类任务达到 SOTA

## LAION-Audio-630K 和训练数据集

LAION-Audio-630K 包含 633,526 个 audio-text pairs，总时长 4,325.39 小时，包含人类活动、自然声音和音频效果。

使用三个训练设置：
1. AudioCaps+Clotho (AC+CL)：约 55K 个 audio-text pairs
2. LAION-Audio-630K (LA.)：约 630K 个 audio-text pairs
3. Audioset：1.9M 个 audio samples，每个 sample 有 label

所有 audio 文件都被预处理为 48kHz 的 mono channel。
> 对于只有 tags 或 labels 的数据集，使用模板 “The sound of label-1, label-2, ..., and label-n” 或 keyword-to-caption model 扩展 labels 为 captions，得到总共 2.5M 个 audio samples

## 模型架构

如图：
![](image/Pasted%20image%2020240920102158.png)

包含 audio encoder $f_{audio}(\cdot)$ 和 text encoder $f_{text}(\cdot)$，分别处理 audio data $X_{i}^{a}$ 和 text data $X_{i}^{t}$，得到 audio embedding $E_{i}^a$ 和 text embedding $E_{i}^{t}$：
$$\begin{aligned}&E_{i}^{a}=MLP_{audio}(f_{audio}(X_i^a))\\&E_{i}^{t}=MLP_{text}(f_{text}(X_i^t))\end{aligned}$$
其中 audio/text projection layer 是一个 2 层 MLP，使用 ReLU 作为激活函数，将 encoder 输出映射到维度 $D$。

模型使用 contrastive learning paradigm 训练：
$$L=\frac1{2N}\sum_{i=1}^N(\log\frac{\exp(E_i^a\cdot E_i^t/\tau)}{\sum_{j=1}^N\exp(E_i^a\cdot E_j^t/\tau)}+\log\frac{\exp(E_i^t\cdot E_i^a/\tau)}{\sum_{j=1}^N\exp(E_i^t\cdot E_j^a/\tau)})$$
其中 $\tau$ 是可学习 temperature 参数，用于缩放 loss。两个对数项分别考虑 audio-to-text logits 和 text-to-audio logits。训练时，$N$ 为 batch size。

训练后，embeddings $E^a$ 和 $E^t$ 可用于不同任务。

### 下游任务的推理

Text-to-Audio Retrieval：通过余弦相似度函数，根据 audio embedding $E^a_p$ 找到 $M$ 个 text embeddings $E^t=\{E_1^t,\cdots,E_M^t\}$ 中最近的 $E^t_q$。

Zero-shot Audio Classification：对于 $M$ 个 audio classes $C=\{C_1,\cdots,C_M\}$，构建 $M$ 个 prompt texts $X^t=\{X_1^t,\cdots,X_M^t\}$（例如，“the sound of class-name”）。对于给定的 audio $X^a_p$，通过余弦相似度函数，找到最近的 $X^t_q$。

Supervised Audio Classification：训练后，对于给定的 audio $X^a_p$，通过添加 projection layer 和 finetuning，将其 embedding $E^a_p$ 映射到分类任务。

### Encoder 选择

选择两个 audio encoder 模型：
+ PANN：CNN-based 音频分类模型，7 个 downsampling CNN blocks 和 7 个 upsampling blocks
+ HTSAT：transformer-based 模型，4 个 swin transformer blocks，在三个音频分类数据集上达到 SOTA

都是用两个模型的倒数第二层的输出，一个 $L$ 维向量，作为 projection MLP layer 的输入，其中 $L_{PANN}=2048$，$L_{HTSAT}=768$。

选择三个 text encoder 模型：
+ CLIP transformer：CLIP 的 text encoder，输出维度为 $L_{CLIP}=512$
+ BERT：输出维度为 $L_{BERT}=768$
+ RoBERTa：输出维度为 $L_{RoBERTa}=768$

使用 2 层 MLPs，将 audio 和 text 输出映射到 512 维。

### 变长音频的特征融合

音频长度不固定，传统方法是将整个音频输入到 audio encoder，取每帧或每块音频 embeddings 的平均值。但是，这种方法在长音频上计算效率低。

本文结合全局和局部信息，对于音频长度为 $T$ 秒和固定块长度 $d=10$ 秒：
+ $T\leq d$：重复输入，然后用零值填充（例如，3 秒输入重复为 9 秒，再填充 1 秒零值）
+ $T>d$：首先将输入从 $T$ 降采样到 $d$ 秒作为全局输入。然后，随机切割三个 $d$ 秒的片段，分别从输入的前 1/3、中间 1/3 和后 1/3，作为局部输入。将这 4 个 $d$ 秒输入到 audio encoder 的第一层，得到初始特征，然后三个局部特征通过另一个 2D-Convolution layer 转换为一个特征。最后，局部特征 $X^a_{local}$ 和全局特征 $X^a_{global}$ 融合为：$X_{fusion}^a=\alpha X_{global}^a+(1-\alpha)X_{local}^a$。其中 $\alpha=f_{AFF}(X^a_{global},X^a_{local})$ 是通过 attention feature fusion (AFF) 得到的因子，AFF 是一个用于学习两个输入融合因子的双分支 CNN 模型。

### Keyword-to-Caption 增强

一些数据集包含合理的 labels 或 tags 作为 audios 的 keywords。使用预训练的语言模型 T5，生成 captions。也对输出句子进行去偏置（de-bias）处理。
> 例如，替换 “woman” 和 “man” 为 ‘person’ 作为 gender de-biasing

## 实验

根据下表：
![](image/Pasted%20image%2020240920105619.png)
最好的模型是 HTSAT + RoBERTa，CLIP transformer 最差。

选择其中最好的模型，实验结果如下：
![](image/Pasted%20image%2020240920105208.png)
逐渐增加数据集规模，发现从 “AudioCaps + Clotho” 到 “LA.” 数据集规模的增加，对 AudioCaps 评估集的结果没有改善，但在 Clotho 评估集上表现更好。

同时，Feature fusion 机制和 keyword-to-caption 增强都提高了性能。
