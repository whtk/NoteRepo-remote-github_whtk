> InterSpeech 2021，Yonsei University

1. 提出一种轻量化的端到端 TTS 模型，在一个框架中将 特征预测模块 和 波形合成模块 结合：
	1. 特征预测模块包含两个独立的子模块：从文本和韵律信息中估计 latent space embedding
	2. 波形生成模块基于 latent space embedding 生成波形
2. 采用域迁移技术，联合训练韵律 embedding 网络和波形生成任务
3. 13.4M 个参数，比 FastSpeech 快 1.6 倍

> 用域迁移学习来从文本中提取 韵律信息，其他的思路基本遵循 encoder + alignment + decoder 的范式。

## Introduction

1. 本文提出全端到端模型，不生成中间的语音特征，而是直接合成波形
2. 采用域迁移技术直接从文本中提取 韵律信息

## 相关工作（略）

## LiteTTS

![](image/Pasted%20image%2020240107111055.png)

包含：
+ prosody encoder $E^p$
+ text encoder $E^t$ 
+ domain transfer encoder $E^f$
+ alignment 模块 $A$
+ duration predictor $P$
+ wave generator $G$
+ discriminator 模块 $D$

长为 $m$ 的 phoneme 序列为 $\boldsymbol{x}=[x_1,x_2,...,x_m]$，长为 $n$ 的 mel 谱 序列为 $Z=[z_1,z_2,...,z_n]$。phoneme 序列来自 encoder，prosody 信息要么来自 prosody encoder 要么来自 domain transfer encoder（推理时）。

### 架构

text encoder 输入文本，产生 text embedding $H^t=\mathbf{E}^t(\mathbf{x}),H^t=[\boldsymbol{h}_1^t,\boldsymbol{h}_2^t,...,\boldsymbol{h}_m^t]$，结构包含一个 embedding 层和 $N$ 层的 Lite-FFT 模块。

Lite-FFT 如图 b(2)，采用 long-short range attention（LSRA）替换原始的 attention 模块，采用两个分支来并行处理信息，分别可以用于获取全局和局部信息。

prosody encoder 从 acoustic 特征中采用单个网络提取多重韵律因子，输入 $Z$，输出 韵律 embedding $\boldsymbol{H}^{p}=[\boldsymbol{h}_{1}^{p},\boldsymbol{h}_{2}^{p},...,\boldsymbol{h}_{n}^{p}]$，即 $H^p=\mathbf{E}^p(\mathbf{Z})$。然后还把 $\boldsymbol{H}^{p}$ 进行 pitch 和 energy 预测任务，来确保其包含这两种信息，这部分的损失为：
$$\mathcal{L}_p=\frac1n\sum_{i=1}^n\lVert p_i-\bar{p}_i\rVert_1\quad\mathrm{and}\quad\mathcal{L}_e=\frac1n\sum_{i=1}^n\lVert e_i-\bar{e}_i\rVert_1$$

对齐模块用于将 $n$ 帧的 $\boldsymbol{H}^{p}$ 对齐到 $m$ 帧：$\tilde{H}^{p}\:=\:\mathbf{A}(H^{p})$，训练的时候，再和 phonetic embedding 相加 $H^c=H^t+\tilde{H}^p$，从而同时包含了文本和韵律信息。

duration predictor：给定 text embedding $H^t$ ，向量 $\bar{\boldsymbol{d}}=[\bar{d}_1,\bar{d}_2,...,\bar{d}_m]$ 表示 $m$ 个 phoneme 的 duration $\bar{d}=\mathbf{P}(H^t)$，损失函数定义为：
$$\mathcal{L}_{dur}=\frac1m\sum_{i=1}^m\lVert\boldsymbol{d}_i-\bar{\boldsymbol{d}}_i\rVert_1$$
然后 $H^c$ 基于这个 duration 拓展得到 $\boldsymbol{H}^e=[\boldsymbol{h}_1^e,\boldsymbol{h}_2^e,...,\boldsymbol{h}_n^e]$，然后通过一个投影层之后输入到 generator $G$ 中。
> 论文中好像没说 GT duration 怎么来的？？

Domain transfer encoder：phoneme 序列的输入 $\mathbf{x}$ 也会输入到这个 encoder 中，产生 $m$ 帧的 embedding $H^{f}=\mathbf{E}^{f}(\boldsymbol{x})$，通过一个损失函数 $\mathcal{L}_c$ ，目的是为了让其输出和 $\tilde{H}^p$ 尽可能相近（即迁移韵律信息），试了多种损失，发现简单的 L1 效果最好。然后推理的时候，这个模块的输出 $H^{f}$ 就会包含 pitch、energy 等韵律信息，从而可以直接替代训练时的 $\tilde{H}^p$。

Waveform generator and discriminator：只把 $\boldsymbol{H}^e$ 的一个固定长度的 segment 作为输入，然后 $G$ 合成波形，$D$ 用来判别真假。

### 训练损失

Discriminator 包含 MPD 和 MSD，则 LSGAN 损失包含：
$$\begin{array}{rcl}\mathcal{L}_{GAN}(D;G)&=&\mathbb{E}_{n,s}\left[\sum_{k=1}^K(D_k(v)-1)^2+(D_k(G(s)))^2\right],\\\mathcal{L}_{GAN}(G;D)&=&\mathbb{E}_s\left[\sum_{k=1}^K(D_k(G(s))-1)^2\right]\end{array}$$
这里的 $v$ 表示不同尺度下的波形，$K$ 为 sub- discriminator 的数量。

feature mapping loss 定义为：
$$\mathcal{L}_{feat}(G;D)=\mathbb{E}_{x,s}\left[\sum_{k=1}^K\sum_{i=1}^T\frac1{N_i}\|D_k^i(x)-D_k^i(G(s))\|_1\right]$$
其中 $T$ 为 layer 数。

同时还采用了 STFT loss，最终总的损失为：
$$\begin{gathered}\mathcal{L}_G=\mathcal{L}_{GAN}\left(D;G\right)+\lambda_f\mathcal{L}_{feat}(G;D)+\mathcal{L}_{dur}\\+\lambda_m\mathcal{L}_{mrstft}+\mathcal{L}_p+\mathcal{L}_e+\lambda_c\mathcal{L}_c,\\\mathcal{L}_D=\mathcal{L}_{GAN}(G;D)\end{gathered}$$
其中 $\lambda_f=2,\lambda_m=30,\lambda_c=5$。

## 实验结果（略）