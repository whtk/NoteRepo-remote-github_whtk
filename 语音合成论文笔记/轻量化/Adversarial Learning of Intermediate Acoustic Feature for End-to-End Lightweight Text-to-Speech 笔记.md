> InterSpeech 2023，Yonsei University

1. 有些文章采用 latent representation 而非 mel 谱 作为中间特征，但是生成不行
2. 本文提出在 latent representation 添加 prosody embedding 来提高性能；训练的时候从mel 谱中提取 prosody embedding，推理时则采用 GAN 从文本中预测

## Introduction

1. 提出 AILTTS，single stage 的 轻量化 TTS 模型，可以提供语音合成中的 speech variance 信息：
	1. 采用 prosody encoder，输入 mel 谱，提取 prosody-related 特征，称为 prosody embedding
	2. 把 prosody embedding 作为训练过程中的条件
	3. 还有一个 prosody predictor 从文本中预测 embedding
	4. 在 prosody predictor 中采用 GAN 来增强估计能力

## 相关工作（略）

