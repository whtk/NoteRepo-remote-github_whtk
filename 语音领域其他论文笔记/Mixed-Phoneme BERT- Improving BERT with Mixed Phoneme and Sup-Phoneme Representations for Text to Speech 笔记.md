> InterSpeech 2022，港科大、MSRA

1. 之前都是采用 character-based unit 来预训练，以增强 TTS phoneme encoder，但是 fine tune 的时候输入是 phoneme ，从而导致不一致
2. 本文提出 Mixed-Phoneme BERT，一种新的 BERT 变体，采用 mixed phoneme 和 sup- phoneme 表征来增强学习能力：
	1. 将邻近的 phoneme 合并为 sup- phoneme，然后将 spu- phoneme 和 phoneme 作为输入
3. 在 FastSpeech 2 的 baseline 上，可以提高 TTS 的性能

## Introduction

1. 之前已有工作把 BERT 作为一个辅助 encoder 用于 TTS，可以提取额外的文本特征，从而生成更具表达性的语音，但是会面临一个挑战：
	1. 预训练和 fine tune 的时候输入不一致
2. 从而导致一些问题：
	1. phoneme 和 character 之间的对齐不稳定
	2. 带来了更多的训练和推理时间、更大的参数量
3. 于是，预训练 phoneme encoder 的时候直接采用 phoneme 作为输入：
	1. 由于 phoneme 只有大概 200 个，直接用的话太少了，不能有效地传递语义信息
4. 于是提出 Mixed-Phoneme BERT，采用 BPE 获得 sup- phoneme 来增大字典大小，然后采用 MLM 目标函数在大规模的无标签的文本数据中训练，还引入了 预处理数据对齐和一致掩码策略 来防止信息泄漏

> 用 BPE 求 sup- phoneme，然后计算 loss 的时候，phoneme 和sup- phoneme 一起算。

## Mixed-Phoneme BERT

### 概览

提出的 Mixed-Phoneme BERT 专为 TTS 设计，包含两个stage：
+ 预训练：在大规模无标签文本上预训练
+ TTS fine tune：speech-text 对做 fine tune

![](image/Pasted%20image%2020231127113801.png)

预训练的时候，sup-phoneme tokens 和 对应的 phoneme tokens 随机 mask 来做预测；fine tune 的时候，输入 token 不做任何 mask，预训练的 BERT 作为 phoneme encoder。

### Mixed Phoneme and Sup-Phoneme Representations

sup-phoneme 指一组邻近的 phoneme，但是不特别对应某个词表中的单词。采用 BPE 编码每个单词为一个或多个 sup-phoneme token。

在 BPE 中，每个 word 都可以表征为 character 序列，本文每个 word 都可以表征为 phoneme 序列。

最终得到的 sup-phoneme 的大小比 phoneme 大，从而可以获得更高的表征能力。对于文本输入，由于 sup-phoneme  序列通常比 phoneme 序列短，于是基于其对应的 phoneme 的数量进行上采样，那么整个混合的表征为：上采样后的 sup-phoneme 序列、phoneme 序列 和 position embedding 的求和。

其中的 phoneme 序列用于增强发音，sup- phoneme 序列为模型带来语义和上下文信息。

### 预训练

和原始的 BERT 类似，随机 mask 一部分 token 然后预测被 mask 的。

如果 phoneme 和 sup- phoneme 序列是独立被随机 mask 的，可能会存在信息泄漏，也就是可以会从没被 mask 的来预测对应位置 被 mask 的。

于是采用一致性的 mask 策略，如果一个 sup- phoneme 序列被 mask 了，那么其对应的 phoneme 也被 mask，然后和 RoBERTa 一样：
+ mask 15% 的 sup- phoneme，这其中：
	+ 80% 变为 mask token
	+ 10% 替换为随机的 token
	+ 10% 保持不变

采用 MLM 目标函数预测 token，即交叉熵损失。会有两个 level 的 损失：
+ sup-phoneme level
+ phoneme level

最终得到的是预测的 vector，其一方面用于预测 phoneme，另一方面，通过平均池化来预测 sup- phoneme。

## 实验（略）