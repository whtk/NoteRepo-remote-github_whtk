> preprint，微软亚研院

1. 语音包含比文本更多的信息
2. 采用文本描述更 user-friendly（因为语音 prompt 可能都不存在），基于文本 prompt 的 TTS 有两个主要的挑战：
	1. text prompt 很难描述语音中所有的 variability
	2. 数据集太少
3. 本文提出 PromptTTS 2，采用 variation network 来提供语音中没有被文本捕获的 variability；采用 large language models 来生成高质量的 text prompt：
	1. variation network 基于 text prompt representation 从参考语音中预测 representation
	2. prompt generation pipeline 则采用 speech language understanding 模型+LLM 从语音中生成 text prompt

## Introduction

1. 为了解决 one-to-many 挑战，采用 variation network 来预测 text prompt 中丢失的信息：
	1. PromptTTS 2 包含 text prompt encoder、reference speech encoder 和一个用于合成语音的 TTS 模块
	2. variation network 基于来自 text prompt encoder 的 prompt representation，训练用于预测来自 reference speech encoder 的 reference representation
	3. 在 variation network 中采用了 diffusion，可以基于 text prompts  从高斯噪声中得到不同的信息
2. 为了解决数据集缺失的问题：
	1. 采用 speech language understanding (SLU)  模型从多种属性（情感、性别等）中描述语音
	2. 然后采用 LLM 基于属性给出描述的句子，然后组合这些句子得到 text prompt

## 背景（略）

## PromptTTS 2

### 整体架构

![](image/Pasted%20image%2020231025103800.png)

a 是 TTS 模块，其特征由 style module 控制，

### Variation Network

### 基于 LLM 的 text prompt 生成

## 实验