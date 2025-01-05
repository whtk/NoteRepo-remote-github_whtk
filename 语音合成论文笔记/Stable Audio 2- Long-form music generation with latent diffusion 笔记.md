> preprint 2024.4，Stability AI

1. 目前没有可以根据文本提示生成连贯完整音乐的音频生成模型
2. 提出了一个基于 long temporal contexts 的生成模型，可以生成长达 4 分 45 秒的音乐，由 diffusion-transformer 和连续的 latent 表征组成（latent rate 为 21.5 Hz）
3. 模型实现了质量和对齐的 SOTA

> 模型层面来看，其实就是在 Stable Audio 1 的基础上把 Unet 替换成了 DiT。

## Introduction

1. 现有的长音乐生成只有 local coherence，没有 long-term musical structure
2. semantic tokens 可以提供 long-term structural coherence 和 high-quality audio synthesis，其常用于自回归模型
3. 一些端到端的方法不需要 semantic tokens，可以直接生成整个音乐片段，将 pipeline 从 text→text-embedding→acoustic-token→waveform 简化为 text→waveform
4. 大多数音乐生成工作使用 autoencoders 将长波形压缩为紧凑的 latent 表征，本文使用 latent diffusion modeling 生成音乐
5. 本文将生成模型扩展到 285s 的时间，使用高度压缩的连续 latent 和 latent diffusion，生成的音乐质量和 text-prompt 一致性达到 SOTA

## Latent Diffusion 架构

模型包含三个模块：
+ autoencoder
+ 基于 CLAP 的 contrastive text-audio embedding model
+ 基于 transformer 的 diffusion model

### Autoencoder

如图：
![](image/Pasted%20image%2020241231105138.png)

+ 输入为原始波形
+ encoder 由一系列卷积块组成，每个 block 通过 strided convolutions 进行下采样和通道扩展
+ 下采样 block 前使用 dilated convolutions 和 Snake activation functions
+ decoder 结构与 encoder 类似，但使用 transposed strided convolutions 进行上采样和通道收缩
+ 结构和 DAC 类似，但在 Snake activation 中添加了 trainable $\beta$ 参数

训练目标：
+ reconstruction loss：多分辨率 STFT
+ adversarial loss：feature matching，5 个卷积判别器
+ KL 散度损失

### DiT

采用 [DiT- Scalable Diffusion Models with Transformers 笔记](../图像合成系列论文阅读笔记/DiT-%20Scalable%20Diffusion%20Models%20with%20Transformers%20笔记.md) 替代之前用的 U-Net，结构如图：
![](image/Pasted%20image%2020241231105815.png)

+ 由多个 block 组成，每个 block 包含 attention layers 和 gated MLPs
+ key 和 query 使用 rotary positional embedding
+ 每个 block 包含 cross-attention layer 用于 conditioning
+ 使用 efficient block-wise attention 和 gradient checkpointing 减少 transformer 在长序列上的计算和内存消耗

DiT 条件有三个：
+ text：自然语言控制，通过 cross-attention 引入
+ timing：变长生成，通过 sinusoidal embeddings 和 cross-attention 引入
+ timestep：diffusion 过程的时间步，进行 sinusoidal embedding 
> 然后将 timing conditioning 和当前时间步的 sinusoidal embedding 添加到 transformer 的输入

### 变长音乐生成

模型允许变长音乐生成，通过 timing condition 填充到指定长度，训练模型以填充剩余部分为 silence，生成短于窗口长度的音频时，可以裁剪 silence。

### CLAP text encoder

基于 CLAP 的对比模型，包含 HTSAT-based 音频编码器和 RoBERTa-based 文本编码器，使用 language-audio contrastive loss 训练。

## 训练设置

训练过程分为两个阶段：
1. 训练 autoencoder 和 CLAP model
2. 训练 diffusion model，首先在 3m 10s 音乐上预训练 70k GPU hours，然后在 4m 45s 音乐上微调 15k GPU hours

所有模型使用 AdamW 优化器，基础学习率为 1e-5，使用 exponential ramp-up 和 decay scheduler，使用 exponential moving average of the weights，weight decay 系数为 0.001。

DiT 使用 v-objective 训练，采样使用 DPM-Solver++（100 steps），classifier-free guidance（scale 为 7.0）。

### 数据集和 prompt 准备（略）

## 实验（略）
