> ACL 2021，Favebook

1. 提出 Generative Spoken Language Modeling，从没有 text 的 raw audio 中学习声学和语言特征
2. baseline 系统包含 离散的 speech encoder、生成式的语言模型、speech decoder，在无监督的条件下进行训练

## Introduction

1. 本文主要贡献：
	1. 分别在声学和语言层面给出了两种 spoken language modeling 的评估指标
	2. 通过和人类评估进行比较来验证提出的指标，表明了两种的相关性很高
	3.  we show that these metrics can be predicted by simpler ones geared to evaluate the encoding mode of the spoken LM
	4. 系统性分析了最近的三个 speech-to-unit encoder，CPC Wave2vec 2.0 和 HuBERT
	5. 开源

## 