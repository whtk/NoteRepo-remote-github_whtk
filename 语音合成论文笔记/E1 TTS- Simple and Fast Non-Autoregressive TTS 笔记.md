> preprint 2024.09，CUHK、网易伏羲，李海洲

1. 提出 Easy One-Step TTS（E1 TTS），非自回归 zero-shot TTS，基于 denoising diffusion 预训练和 distribution matching distillation
2. 训练简单，不需要文本和音频对之间的显式单调对齐
3. 推理高效，每个 utterance 只要一次 evaluation

## Introduction

1. 大部分 NAR TTS 都包含 duration predictor，并依赖于 alignment supervision
2. 一些研究提出了 ID-NAR TTS 模型，不需要 alignment supervision 或 explicit duration prediction，采用 attention 机制端到端学习对齐
3. 基于 diffusion 的 ID-NAR TTS 可以实现 zero-shot TTS，在自然度和说话人相似度上达到 SOTA，但是需要多次的 evaluation，diffusion distillation 可以减少次数，且大多数 distillation 基于 ODE sampling trajectories
4. 本文将基于 diffusion 的 ID-NAR TTS 模型 distill 成 one-step generator，采用 distribution matching distillation，实现和 AR/NAR baseline 系统相当的性能

## 背景

### Distribution Matching Distillation

考虑位于 $\mathbb{R}^d$ 的数据分布 $p(x)$，将 $p(x)$ 与 Gaussian perturbation kernel $q_t(x_t|x) = N(x_t;\alpha_tx, \sigma_t^2I_d)$ 卷积，得到 perturbed density $p_t(x_t) := \int p(x)q_t(x_t|x)dx$，其中 $\alpha_t, \sigma_t >0$ 是每个时间 $t \in [0,1]$ 的 SNR。大部分 diffusion 模型等价于学习一个神经网络，逼近每个时间 $t$ 的 score function $s_p(x_t, t) := \nabla_{x_t} \log p_t(x_t)$。

考虑 generator 函数 $g_{\theta}(z) : \mathbb{R}^d \rightarrow \mathbb{R}^d$，输入随机噪声 $Z \sim \mathcal{N}(0, I_d)$，输出 fake samples $\widehat{X} := g_{\theta}(Z)$，分布为 $q_{\theta}(x)$。如果可以得到两个 score functions $s_p(x) := \nabla_x \log p(x)$ 和 $s_q(x) := \nabla_x \log q_{\theta}(x)$，计算以下 KL 散度的梯度：
$$\nabla_\theta D_{\mathrm{KL}}\left(q_\theta(x)\|p(x)\right)=E\left[\left(s_q(\widehat{X})-s_p(\widehat{X})\right)\frac{\partial g_\theta(Z)}{\partial\theta}\right].$$

但是直接得到 $s_p(x)$ 和 $s_q(x)$ 很困难。可以训练 diffusion 模型估计 perturbed distributions $p_t(x_t)$ 和 $q_{\theta, t}(x_t) := R q_{\theta}(x)q_t(x_t|x)dx$ 的 score functions $s_p(x_t, t)$ 和 $s_{q, t}(x_t, t)$。考虑所有 noise scales 的 KL 散度的加权平均：
$$D_\theta:=E_{t\thicksim p(t)}\left[w_tD_{\mathrm{KL}}\left(q_{\theta,t}(x_t)\|p_t(x_t)\right)\right],$$
其中 $w_t \geq 0$ 是时间相关的权重因子，$p(t)$ 是时间分布。$W \sim \mathcal{N}(0, I_d)$ 是独立的高斯噪声，定义 $\widehat{X}_t\::=\:\alpha_t\widehat{X}\:+\:\sigma_tW$。加权 KL 散度的梯度可以计算为：
$$\nabla_\theta D_\theta=E_{t\thicksim p(t)}\left[w_t\alpha_t\left(s_q(\widehat{X}_t,t)-s_p(\widehat{X}_t,t)\right)\frac{\partial g_\theta(Z)}{\partial\theta}\right].$$

给定预训练的 score estimator $s_{\phi}(x_t, t) \approx s_p(x_t, t)$，将其 distill 成 one-step generator $g_{\theta}$ 的过程如下：
![](image/Pasted%20image%2020241001114131.png)

虽然 generator $g_{\theta}$ 理论上可以随机初始化，但是用 $s_{\phi}$ 初始化会更快收敛，性能更好。已有研究发现，预训练的 diffusion 模型已经具有 latent one-step generation 能力，可以通过调整部分参数（如 normalization layers）将其转换为 one-step generator。

distribution matching distillation 与 GAN 相似，需要交替优化，但是实验证明更稳定，避免了 GAN 训练中常见的 mode collapse 问题。


### Rectified Flow

Rectified Flow 构建 ODE：
$$\mathrm{d}Y_t=v(Y_t,t)\mathrm{d}t,\quad t\in[0,1],$$

实现两个随机分布 $X_0 \sim \pi_0$ 和 $X_1 \sim \pi_1$ 的映射，通过优化：
$$v(x_t,t):=\underset{v}{\operatorname*{\arg\min}}E\|v\left(\alpha_tX_1+\sigma_tX_0,t\right)-(X_0-X_1)\|_2^2,$$
其中 $\alpha_t = t$，$\sigma_t = (1-t)$。特殊情况下，$X_0 \sim \mathcal{N}(0, I_d)$ 且 $X_0 \perp X_1$，drift $v(x_t, t)$ 是 score function $s(x_t, t) = \nabla_{x_t} \log p_t(x_t)$ 和 $x_t$ 的线性组合，其中 $X_t := \alpha_tX_1 + \sigma_tX_0$：
$$s(x_t,t)=-\frac{1-t}{t}v(x_t,t)-\frac{1}{t}x_t.$$

实验中，所有的 diffusion 模型使用 Rectified Flow loss。

## E1 TTS

E1 TTS 是一个级联条件生成模型，输入完整文本和部分 mask 的语音，输出完整语音。整体架构如下图：
![](image/Pasted%20image%2020241001115221.png)

模型类似于[Autoregressive Diffusion Transformer forText-to-Speech Synthesis 笔记](Autoregressive%20Diffusion%20Transformer%20forText-to-Speech%20Synthesis%20笔记.md)，
但是所有 speech tokens 在第一阶段同时生成。采用 DMD 将两个 DiTs 转换为 one-step generators。

### mel 谱 Autoencoder

构建 mel 谱 autoencoder，包含 Transformer encoder 和 Diffusion Transformer decoder。Encoder 输入 log Mel spectrograms，输出 $\mathbb{R}^{32}$ 的 continuous tokens，大约 24Hz。Decoder 是 Rectified Flow 模型，输入 speech tokens，输出 Mel spectrograms。Encoder 和 decoder 一起训练，使用 diffusion loss 和 KL loss 平衡 rate 和 distortion。Mel spectrogram autoencoder 在合成时当部分 spectrogram 已知的情况下进行 fine-tuning，以增强 speech inpainting 性能。Decoder 在 transformer blocks 后添加 2D convolutions 层，提高在 spectrograms 上的性能。

### Text-to-Token Diffusion Transformer

Text-to-Token DiT 给定完整文本，估计 mask 部分的 speech tokens。训练时，speech tokens 随机分为三部分：prefix、masked middle 和 suffix。首先均匀采样 middle 部分的长度，然后均匀采样 middle 部分的开始位置。10% 的概率 mask 整个 speech token 序列。

在所有 Transformer blocks 中采用 RoPE。对于文本 token，为递增的整数位置索引。对于 speech token，为分数位置索引，增量为 $\frac{n_{\text{text}}}{n_{\text{speech}}}$。
> 这种设计类似于 attention，在其他 ID-NAR TTS 模型中也被证明有效。


### Duration 建模

训练类似于 [Voicebox- Text-Guided Multilingual Universal Speech Generation at Scale 笔记](Voicebox-%20Text-Guided%20Multilingual%20Universal%20Speech%20Generation%20at%20Scale%20笔记.md) 的 duration predictor。首先通过基于 RAD-TTS 的 aligner 获得 text 和 speech token 的粗略对齐，然后训练 duration model 估计部分 mask 的 durations。duration model 输入完整文本（phoneme sequence），部分 mask 的 durations，预测未知 durations。
> 最小化总 duration 的 L1 loss 比直接最小化 phoneme-level durations 效果更好。

### 推理

E1 TTS 的推理如下：
+ 将原始的文本和语音强制对齐，获得原始 phoneme durations
+ 将 target phoneme sequence 和未编辑 phonemes 的原始 durations 输入 duration predictor，估计 target speech 的总 duration
+ 将原始 speech 编码为 speech tokens，删除 要编辑部分的 tokens，插入新的 noise tokens，使得总 duration 匹配
+ 将 target text 和部分 mask 的 target speech tokens 输入 Text-to-Token DiT，获得重建 speech tokens，输出编辑后的 speech

对于单说话人 TTS，只有文本输入，可以移除 force-alignment 和 duration predictor 步骤。可以假设 synthesized speech 的总 duration 是输入文本长度的固定倍数，或者训练一个 total-duration predictor，不需要 text unit durations。

## 实验和结果（略）
