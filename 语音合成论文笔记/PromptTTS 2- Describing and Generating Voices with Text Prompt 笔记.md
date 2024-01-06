> preprint，微软亚研院

1. 语音包含比文本更多的信息
2. 采用文本描述更 user-friendly（因为语音 prompt 可能都不存在），基于文本 prompt 的 TTS 有两个主要的挑战：
	1. text prompt 很难描述语音中所有的 variability
	2. 数据集太少
3. 本文提出 PromptTTS 2，采用 variation network 来提供语音中没有被文本捕获的 variability；采用 large language models 来生成高质量的 text prompt：
	1. variation network 基于 text prompt representation 从参考语音中预测 representation
	2. prompt generation pipeline 则采用 speech language understanding 模型+LLM 从语音中生成 text prompt

> 在 PromptTTS 1 的基础上，增强了 style encoder，即 用一个 diffusion 模型从文本中生成 speech representation，然后采用 SLU+LLM 来自动生成数据集。

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

a 是 TTS 模块，其特征由 style module 控制。TTS backbone 用的是 [NaturalSpeech 2- Latent Diffusion Models are Natural and Zero-Shot Speech and Singing Synthesizers 笔记](NaturalSpeech%202-%20Latent%20Diffusion%20Models%20are%20Natural%20and%20Zero-Shot%20Speech%20and%20Singing%20Synthesizers%20笔记.md) 中的。

style 模块 如图 b， 和 [PromptTTS- Controllable Text-to-Speech with Text Descriptions 笔记](PromptTTS-%20Controllable%20Text-to-Speech%20with%20Text%20Descriptions%20笔记.md) 相同，采用 BERT-based 模型作为 text prompt encoder 来提取 prompt hidden。同时还采用了 reference speech encoder 来建模没有被文本捕获的信息。最终分别得到一个固定长度的 representation，即 text prompt representation $(P_1,...,P_M)$ 通过可学习的 query $(Q_{P_1},...,Q_{P_M})$ 提取，reference speech representations $(R_1,...,R_N)$ 则通过可学习的 query $(Q_{R_1},...,Q_{R_N})$ 获得。

推理的时候，只有 text prompt，于是训练一个 variation network 来基于 $(P_1,...,P_M)$ 预测 reference speech representations $(R_1,...,R_N)$ 。

### Variation Network

variation network 采用 diffusion 模型，模型训练用于估计噪声数据的对数密度的梯度 $\nabla\log p_t(z_t)$，实际就是预测原始的 reference speech representation $z_0$
。输入条件则是 prompt representation, noised reference representation 和 diffusion step $t$。

上图 c 为模型结构，基于 transformer  encoder，输入为 prompt representation $(P_1,...,P_M)$，noised reference representation $(R_1^t,...,P_M^t)$ 和 diffusion step $t$，输入为预测的 hidden representation，采用 L1 loss 进行优化，采用  FiLM 来提高模型对 step 的感知能力。

### 基于 LLM 的 text prompt 生成

![](image/Pasted%20image%2020231025203948.png)

如图，包含 SLU 和 LLM 两个部分，给定语音，SLU 对语音打标签来识别语音的属性，LLM 指导语言模型基于标签来写 text prompt。

为了提高 prompt 的质量，LLM 一步一步地构造来得到 text prompt，细节如下：
![](image/Pasted%20image%2020231025204246.png)
+ 关键词构造：根据不同的属性来选取不同的关键词
+ 语句构造：对每个属性都生成多个句子
+ 语句组合：因为包含不止一个属性，需要将这些语句组合（通过指令来实现）
+ 数据集实例化：第三阶段生成的文本作为最终的 prompt

## 实验

