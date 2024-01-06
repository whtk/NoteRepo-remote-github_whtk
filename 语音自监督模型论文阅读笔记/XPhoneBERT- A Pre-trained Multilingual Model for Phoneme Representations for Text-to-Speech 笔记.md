> InterSpeech 2023，越南，VinAI Research

1. 提出 XPhoneBERT，是第一个预训练的用于学习下游 TTS 任务的音素表征的多语言模型
2. 和 BERT 架构相同，使用RoBERTa预训练方法来自近100种语言和地区的330M个音素级句子进行训练
3. 实验表明，使用XPhoneBERT作为 音素编码器（phoneme encoder）提高了 TTS 模型在自然度和韵律方面的性能

> 没有任何创新点，大力出奇迹，堆数据集和语言数量就完事了

## Introduction

1. 已经有些 TTS 模型将预训练的 BERT 生成的 contextualized word embeddings 用在了 encoder 中，phoneme sequence 通过 encoder 来产生 phoneme 表征；文本通过 BERT 得到 word embedding，然后将两者 concate 得到 decoder 的输入
2. 这其中 BERT 的作用可以帮助提高合成语音的质量，预训练的 BERT 用于为 phoneme 表征提供额外的上下文信息
3. 因此，如果 contextualized phoneme representations 是直接是直接通过一个 BERT 模型产生的，而且这个模型是用无标签的 phoneme-level 数据训练的，效果应该会更好
4. 最近的模型 PnG BERT、Mixed-Phoneme BERT、Phoneme-level BERT 都已经证明可以提高性能了，这些模型可以直接作为 TTS 模型的 input encoder；但是都是在英语上的
5. 于是为 phoneme 表征训练了一个大规模、多语种的 语言模型，在近 100 种语言中 330M phonemic description sentences 的数据集上预训练，采用 RoBERTa 预训练方法，用 BERT-based 模型

## XPhoneBERT

### 模型架构

用的是 BERT-based 架构，12 层的 transformer，768 维的 hidden size，12 heads。采用 MLM 的预训练方法，和 RoBERTa 一样，即用 dynamic masking 策略而非 next sentence prediction 的目标函数。

### 多语言预训练数据集

构造数据包含三个步骤：
+ 收集文本，分割+归一化：用的是 wiki40b 和 wikipedia 多语言数据集，用 spaCy 做分割，然后都转为小写，然后用 NeMo 做归一化（从书面语转为口头语）
+ 将文本转为 phoneme（用的是 CharsiuG2P 工具包）：当输入的 word 在 CharsiuG2P 工具包的发音词典时，采用辞典中的 phoneme，否则使用这个包中的预训练的模型来生成，这个过程保持标点符号不变
+ 做 phoneme 分割：上面转换的时候是没有 phoneme 边界的，于是采用 segments 工具包，在俩俩 phoneme 之间插入 空格，例如 “model” 被转为 “"m A d @ ë”，同时用 _ 来区分不同 word 的边界，如 “a multilingual model” 被转为 “"e I _ "m @ ë t i "ë I N w @ ë _ "m A d @ ë”

详细的数据如下：
![](image/Pasted%20image%2020231127214545.png)

### 优化

采用 white-space tokenizer，最终得到 1960 个 phoneme，模型参数量，为 87.6M，用 fairseq 库中的 RoBERTa 的实现，最大序列长度 512，Adam 优化器，batch size 1024，8 张 A100 训练 20 epoch，共计 18 天。

## 实验
