> 高丽大学，2023.7.31 发布

1. 传统的韵律建模依赖于自回归方法来预测量化的 prosody vector，但是会出现长时依赖问题，而且推理很慢
2. 提出 prosody vector，采用 diffusion-based latent prosody generator 和 prosody conditional adversarial training 来合成 expressive speech
3. 提出的  prosody generator 很有效，且 prosody conditional discriminator 可以通过模仿韵律提高生成的质量
4. 采用 denoising diffusion generative adversarial networks 来提高韵律的生成速度，比传统的生成模型快 16 倍

## Introduction

1. 提出 DiffProsody，采用 diffusion-based latent prosody generator（DLPG）和 prosody conditional adversarial training 来生成 expressive speech
2. 贡献包含：
	1. 提出 diffusion-based latent prosody 建模方法，能够生成高质量的 latent prosody representation，从而增强合成语音的表达性，采用 denoising diffusion generative adversarial networks (DDGANs) 来减少 time step 数量，从而加速 AR 和 DDPM
	2. 提出 prosody conditional adversarial training，来确保在 TTS 中的 prosody 能够被正确反映

## 相关工作（略）

## DiffProsody

