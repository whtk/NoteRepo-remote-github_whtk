> TASLP，2022， 腾讯 AI Lab，北大

1. 提出基于文本 text prompt 来生成 sound effects，包含 text encoder，Vector Quantized Variational Autoencoder (VQ-VAE)，token-decoder, 和 vocoder
2. 采用  token-decoder  将 text encoder 得到的 text features 转换 mel 谱，然后用 vocoder 生成波形
3. 发现 token-decoder 可以极大地影响性能：
	1. 先用 AR-token-decoder，发现性能就已经可以超过之前的工作
	2. 但是 AR 存在 unidirectional bias 和 accumulation of errors 问题
	3. 提出基于 discrete diffusion model 的 non-autoregressive token-decoder，称为 Diffsound
	4. Diffsound 效果好速度快

 ## Introduction

1. 本文是第一个基于文本描述生成声音的工作，其可以为合成语音添加背景、影视合成
2. 之前的模型通常是 two-stage 的：
	1. 先用 autoregressive (AR) decoder 基于 one-hot label 或者 video 生成 mel 谱
	2. 使用 vocoder 生成波形
3. 本文提出从 VQ-VAE 的 codebook 中先学习一个先验，用于将 mel 谱 压缩到 token，此时 mel 谱 生成问题就可以看成是使用 decoder 基于文本来预测一系列的 token 序列
4. 本文的专注点就在 token-decoder 的设计
	1. 探索了 AR 模型
	2. 提出了 Diffsound 模型
5. 为了解决数据集缺失的问题，采用 mask-based text generation strategy (MBTG) 从 event label 中生成文本描述
6. 采用 curriculum learning，先学习只有一个 event 的生成，再逐渐增加 event
7. 采用 FID、KL 散度和 audio caption loss 来评估模型，也评估了 MOS

![](image/Pasted%20image%2020231006151219.png)

## 相关工作（略）

## 框架

## 基于 diffusion 的 decoder

## 数据集和数据预处理（略）

## 评估指标（略）

## 实验（略）