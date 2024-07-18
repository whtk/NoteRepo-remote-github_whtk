> Meta，2022

1. 提出一种 audio codec，包含 streaming encoder-decoder 架构，latent space 是量化的，训练方式是端到端的
2. 使用一个 multiscale spectrogram adversary 来加速训练，减少伪影，提高生产质量
3. 引入一个新的 loss balancer mechanism 来稳定训练
4. 研究用 lightweight Transformer 来进一步压缩得到的表征

## Introduction

1. 有损压缩的目标是，最小化比特率的同时最小化失真
2. 有损神经压缩模型存在两个问题：
	1. 模型需要表征很高范围的信号，可以用大规模数据来解决
	2. 压缩效率，包括时间和大小
3. 提出 encodec，可以在 语音和音乐上实现 SOTA 的 MUSHRA score

## 相关工作（略）

## 模型

时间长度为 $d$ 的语音信号表示为序列 $x\in[-1,1]^{C_{\mathrm{a}}\times T}$，其中 $C_{\mathrm{a}}$ 为通道数，$T=d\cdot f_{\mathrm{sr}}$ 为样本点数。

EnCodec 包含三个组成部分：
+ encoder $E$ 输入语音，输出表征 $z$
+ 量化层 $Q$ 输出压缩后的表征 $z_q$
+ decoder $G$ 从 $z_q$ 中重构时域信号 $\hat{x}$

整个架构使用 重构损失和感知损失 端到端训练：
![](image/Pasted%20image%2020231126222359.png)

### Encoder 和 Decoder 架构

encoder $E$ 包含 1D 卷积 + $B$ 个卷积块，每个卷积块包含 residual 和 下采样层。卷积块后面是两层的 LSTM 来进行序列建模，最后通过 1D 卷积。

取决于低延迟（streamable）还是高质量，模型有两个变体。

模型在 24KHz 的音频下，输出 75 个 latent（48KHz 就是 150 个）。

decoder 镜像 encoder 结构。

### RVQ

结构和 [SoundStream- An End-to-End Neural Audio Codec 笔记](SoundStream-%20An%20End-to-End%20Neural%20Audio%20Codec%20笔记.md) 一致，每 10 bit 分一个 codebook。

### 语言模型和熵编码

训练一个基于 small transformer 的语言模型，包含 5 层，8 heads，200 channels，维度为 800。

在 time step $t$，$t-1$ 时刻得到的离散表征转为连续表征，然后进行相加。

transformer 的输出送到 $N_q$ 个线性层中（$N_q$ 为 RVQ 的 quantizer 的数量），输出的 channel 等于 codebook 的大小（也就是向量个数），从而输出在时间 $t$ 下的 logits（也就是每个 向量 对应的概率）。

熵编码：
采用  range based arithmetic coder，从而可以用上前面 transformer 得到的概率。

### 目标函数

包含：
+ 重构损失
+ 感知损失（判别器损失）
+ RVQ commitment loss

重构损失：
包含时域和频域损失。时域上，最小化 target 和 压缩后的音频 之间的 L1 距离：$\ell_t(\boldsymbol{x},\hat{\boldsymbol{x}})=\|\boldsymbol{x}-\hat{\boldsymbol{x}}\|_1$。对于频域，采用不同时间尺度下的 mel 谱 的 L1 和 L2 损失的线性组合：
$$\ell_f(\boldsymbol{x},\boldsymbol{\hat{x}})=\frac1{|\alpha|\cdot|s|}\sum_{\alpha_i\in\alpha}\sum_{i\in e}\|\mathcal{S}_i(\boldsymbol{x})-\mathcal{S}_i(\hat{\boldsymbol{x}})\|_1+\alpha_i\|\mathcal{S}_i(\boldsymbol{x})-\mathcal{S}_i(\hat{\boldsymbol{x}})\|_2$$
这里的 $\mathcal{S}_i$ 是一个 64-bin 的 mel 谱，其 STFT 的窗口为 $2^i$。

判别器损失：
引入基于  multi-scale STFT-base 判别器来计算感知损失，如图：
![](image/Pasted%20image%2020231127100412.png)

MS-STFT 判别器由在多尺度 复值 STFT 上操作的结构相同的网络组成，其中实部和虚部被连接起来。同样也是不同的 STFT 的窗口长度 $[2048,\:1024,\:512,\:256,\:128]$（24KHz）。

此时生成器的对抗损失为：
$$\ell_g(\hat{\boldsymbol{x}})=\frac1K\sum_k\max(0,1-D_k(\hat{\boldsymbol{x}})))$$
其中 $K$ 为 discriminator 的数量。同时还和之前的工作一样，额外加上了一个特征匹配损失来训练生成器：
$$\ell_{feat}(\boldsymbol{x},\hat{\boldsymbol{x}})=\frac1{KL}\sum_{k=1}^K\sum_{l=1}^L\frac{\|D_k^l(\boldsymbol{x})-D_k^l(\hat{\boldsymbol{x}})\|_1}{\mathrm{mean}\left(\|D_k^l(\boldsymbol{x})\|_1\right)}$$

Discriminator 的损失为 Hinge Loss：
$$L_d(\boldsymbol{x},\hat{\boldsymbol{x}})=\frac1K\sum_{k=1}^K\max(0,1-D_k(\boldsymbol{x}))+\max(0,1+D_k(\hat{\boldsymbol{x}}))$$
由于判别器要比生成器要强，以 2/3 的概率更新判别器（24KHz，对于 48KHz 这个概率为 0.5）。

VQ commitment loss：
计算 encoder 的输出和量化后的值之间的 loss（但是梯度直通），对于每个 residual step $c\in\{1,\ldots C\}$，$z_c$ 为 当前的 residual，$q_c(z_c)$ 为其在 codebook 中最临近的离散值，则损失定义为：
$$l_w=\sum_{c=1}^C\|\boldsymbol{z}_c-q_c(\boldsymbol{z}_c)\|_2^2$$
此时 生成器 的总损失为：
$$L_G=\lambda_t\cdot\ell_t(\boldsymbol{x},\hat{\boldsymbol{x}})+\lambda_f\cdot\ell_f(\boldsymbol{x},\hat{\boldsymbol{x}})+\lambda_g\cdot\ell_g(\hat{\boldsymbol{x}})+\lambda_{feat}\cdot\ell_{feat}(\boldsymbol{x},\hat{\boldsymbol{x}})+\lambda_w\cdot\ell_w(w)$$

Balancer：
引入一个 loss balance 来稳定训练。定义 $g_i=\frac{\partial\ell_i}{\partial\hat{x}}$，而 $\langle\|g_i\|_2\rangle_\beta$ 为 $g_i$ 在上一个 training batch 上的指数滑动平均，给定一系列权重 $\lambda_i$ 和 reference norm $R$，定义：
$$\tilde{g}_i=R\frac{\lambda_i}{\sum_j\lambda_j}\cdot\frac{g_i}{\langle\|g_i\|_2\rangle_\beta}$$
然后反向传播的时候，用的是 $\sum_i\tilde{g}_i$ 而非原始的 $\sum_i\lambda_ig_i$。

## 实验（略）