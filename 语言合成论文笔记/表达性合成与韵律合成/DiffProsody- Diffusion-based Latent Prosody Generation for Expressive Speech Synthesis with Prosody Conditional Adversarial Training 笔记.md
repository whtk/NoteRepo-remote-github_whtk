> 高丽大学，2023.7.31 发布

1. 传统的韵律建模依赖于自回归方法来预测量化的 prosody vector，但是会出现长时依赖问题，而且推理很慢
2. 提出 prosody vector，采用 diffusion-based latent prosody generator 和 prosody conditional adversarial training 来合成 expressive speech
3. 提出的  prosody generator 很有效，且 prosody conditional discriminator 可以通过模仿韵律提高生成的质量
4. 采用 denoising diffusion generative adversarial networks 来提高韵律的生成速度，比传统的生成模型快 16 倍

## Introduction
