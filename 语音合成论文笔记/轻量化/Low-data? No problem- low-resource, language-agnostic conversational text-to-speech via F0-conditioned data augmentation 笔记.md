> InterSpeech 2022，Amazon Alexa, TTS Research

1. 提出，在一个小时的对话语音下，构建低资源的 TTS
2. 提出一个三步的技术：
	1. 训练一个 F0-conditioned voice conversion (VC) 模型进行数据增强
	2. 训练 F0 predictor 来控制合成数据的风格
	3. 训练 TTS 来用上增强后的数据
3. 实验表明，可以实现可控的 F0，且可以跨说话人和跨语言

## Introduction

1. 低资源的 TTS 的两个主流趋势：
	1. 采用多说话人的数据集做迁移学习
	2. 采用多说话人数据集+VC+TTS 做数据增强
2. 