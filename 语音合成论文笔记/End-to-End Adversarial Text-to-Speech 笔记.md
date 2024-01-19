> ICLR 2021，google

1. 现有的 TTS 系统通常有多个处理阶段
2. 本文提出从 phoneme 输入开始，直接端到端合成原始波形
3. 提出的 feed-forward generator 可以同时适用于训练和推理，且基于 token length prediction 采用可微的对齐方法
4. 在 mel 谱预测中采用 soft dynamic time warping 来捕获合成音频中的  temporal variation
5. 可以和 multi-stage 的 SOTA 模型相比较

## Introduction

1. 提出 EATS： End-to-end Adversarial Text-to-Speech，输入为  pure text 或 raw phoneme，输出 raw waveform
2. 模型包含两个 high-level 的子模块：
	1. aligner 用于处理 raw input sequence 产生低频的对齐后的特征
	2. decoder 通过 1D 卷积上采样得到 24K 波形
3. 贡献如下：
	1. 完全可微的 feed-forward aligner 结构，可以预测每个 token 的 duration 然后产生对齐后的表征
	2. 用 dynamic time warping-based 损失来强迫对齐，同时使得模型可以捕获语音中的时间变化
	3. MOS 为 4.083，达到了 SOTA

## 方法

目标是将输入的 characters 或 phonemes 映射到 24K raw audio。最大的挑战在于，输入和输出没有对齐，于是将 generator 分为两个模块：
+ aligner 将输入序列映射到 200Hz 下的对齐后的表征中
+ decoder 将 aligner 的输出上采样到 24K
整个结构都是可微的，整个结构如下：
![](image/Pasted%20image%2020240119172828.png)

用的是 GAN-TTS 的 generator 作为这里的 decoder，但是其输入不是预先计算好的 linguistic feature 而是 aligner 的输出。同时在 latent $z$ 中加入 speaker embedding $s$，也用了 GAN-TTS 中的  multiple random window discriminators (RWDs)；输入的 raw audio 用的是 mu 律压缩，generator 用于产生 mu 律压缩下的音频。

损失函数为：
$$\mathcal{L}_G=\mathcal{L}_{G,\mathrm{adv}}+\lambda_{\mathrm{pred}}\cdot\mathcal{L}_{\mathrm{pred}}^{\prime\prime}+\lambda_{\mathrm{length}}\cdot\mathcal{L}_{\mathrm{length}}$$
其中 $\mathcal{L}_{G,\mathrm{adv}}$ 为对抗损失。

### aligner
 
给定长为 $N$ 的 token 序列 $\textbf{x}=(x_1,\ldots,x_N)$，首先计算 token 表征 $\mathbf{h}=f(\mathbf{x},\mathbf{z},\mathbf{s})$，$f$ 为 一堆  dilated convolutions，
