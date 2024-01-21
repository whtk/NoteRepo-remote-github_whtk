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
 
给定长为 $N$ 的 token 序列 $\textbf{x}=(x_1,\ldots,x_N)$，首先计算 token 表征 $\mathbf{h}=f(\mathbf{x},\mathbf{z},\mathbf{s})$，$f$ 为 一堆 dilated convolutions，然后用 $z,s$ 来调制 batch norm 层的 缩放和偏移参数。然后独立预测每个输入 token 的长度 $\begin{aligned}l_n=g(h_n,\mathbf{z},\mathbf{s})\end{aligned}$，$g$ 是 MLP，用 ReLU 激活确保输出非负。然后把累积和作为 token 的终点位置： $\begin{aligned}e_n=\sum_{m=1}^nl_m\end{aligned}$，此时 token  center 为 $\begin{aligned}c_n=e_n-\frac{1}{2}l_n\end{aligned}$，然后可以将 token 表征插值到 对齐后的 200Hz 的表征，$\mathbf{a}=(a_1,\ldots,a_S),\text{where }S=\lceil e_N\rceil$，这里的 $S$ 为总的输出 time step。为了计算 $a_t$，采用 softmax 获得 token representations $h_t$ 的插值权重，计算为：
$$w_t^n=\frac{\exp\left(-\sigma^{-2}(t-c_n)^2\right)}{\sum_{m=1}^N\exp\left(-\sigma^{-2}(t-c_m)^2\right)}$$
此时 $a_t$ 计算为 $\begin{aligned}a_t\:=\:\sum_{n=1}^Nw_t^nh_n\end{aligned}$（每一帧都有一个长为 $N$ 的权重向量，一共 $S$ 帧），通过预测 token length 然后采用累积求和来得到位置相比于直接求位置，可以迫使单调对齐。
> ！秒啊，还可以这样对齐。相比于之前的简单重复，这里用高斯核可以更好地捕获不同的 phoneme 对于每一帧的贡献。

### Windowed Generator Training

训练样本长度从 1-20s 可变，训练的时候无法 pad 到最大长度，浪费很大。于是从每个样本中随机提取 2s 用于训练，aligner 也是用于产生这 200Hz 的特征对应这段 2s 的音频。

### Adversarial Discriminator

Random window discriminators：用的是 GAN-TTS 中的 RWD，每个 RWD 都在音频不同长度的段上进行，这些 discriminator 是以 speaker 为 condition 的。

Spectrogram discriminator：提取对数 mel 谱然后用 BigGAN-deep 架构，把 spectrogram 看作图像。

### Spectrogram 预测损失

实验发现，对抗学习并足以满足对齐训练，于是提出采用显示的预测损失来引导学习。其实就是，最小化啊对数坐标下的生成的 mel 谱 和 GT mel 谱 的 L1 损失：
$$\mathcal{L}_{\mathrm{pred}}=\frac1F\sum_{t=1}^T\sum_{f=1}^F|S_{\mathrm{gen}}[t,f]-S_{\mathrm{gt}}[t,f]|$$
> mel 谱 是通过 generator 输出的波形转换而来的。

### Dynamic Time Wrapping

spectrogram 预测损失需要 token 的长度的固定的。但是生成的 和 GT mel 谱 不一定完全对齐，于是引入 dynamic time warping (DTW) ，通过 迭代查找最小成本的对齐路径 $p$，从第一个 time step 开始 $p_{\text{gen},1}=p_{\text{gt},1}=1$，在第 $k$ 次迭代中，从下面三个可能的操作中选择其一：
$$\begin{aligned}
&1.\text{ go to the next time step in both }S_{\mathrm{gen}},S_{\mathrm{gt}}:p_{\mathrm{gen},k+1}=p_{\mathrm{gen},k}+1,\mathrm{~}p_{\mathrm{gt},k+1}=p_{\mathrm{gt},k}+1; \\
&\text{2. go to the next time step in }S_{\mathrm{gt}}\text{ only. }p_{\mathrm{gen},k+1}=p_{\mathrm{gen},k},\mathrm{~}p_{\mathrm{gt},k+1}=p_{\mathrm{gt},k}+1; \\
&\text{3. go to the next time step in }S_{\mathrm{gen}}\text{ only: }p_{\mathrm{gen},k+1}=p_{\mathrm{gen},k}+1,\mathrm{~}p_{\mathrm{gt},k+1}=p_{\mathrm{gt},k}.
\end{aligned}$$
得到的路径为 $p\:=\:\langle(p_{\text{gen},1},p_{\text{gt},1}),\ldots,(p_{\text{gen},K_p},p_{\text{gt},K_p})\rangle$，其中 $K_p$ 为长度。
> 详细的原理略（见论文）。

### Aligner Length 损失

用一个损失来鼓励预测的总长度尽可能接近训练中的 step 数量 $L$：
$$\mathcal{L}_\mathrm{length}=\frac12\left(L-\sum_{n=1}^Nl_n\right)^2$$

### 文本预处理

发现采用 phoneme 作为输入时效果更好，采用 phonemizer 作为转换工具，然后在序列的最前端和最后端添加一个 silence token。

## 相关工作（略）

## 实验（略）