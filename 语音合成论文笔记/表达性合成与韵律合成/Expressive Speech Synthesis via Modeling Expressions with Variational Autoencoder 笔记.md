> Interspeech 2018，东京大学

1. 提出将 VoiceLoop 和 VAE 组合，采用 VAE 显式建模 global characteristics，从而可以以无监督的方式控制合成语音的表达性

## Introduction

1. 本文关注的问题是，不依赖于 speech expression labels 的情况下合成 expressive speech，称为 unsupervised expressive speech synthesis (UESS)
2. 另一个关注的问题是，neural autoregressive models，如 WaveNet 等
3. 提出 VAE-Loop，将 VoiceLoop 和 VAE 组合起来，采用 VAE 将 global characteristics 引入语音，从而可以生成高质量的音频和可控的 expressions

## 相关工作（略）

