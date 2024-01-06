> ASRU 2023，西工大、腾讯

1. Singing Voice Conversion Challenge 2023 比赛的工作
2. 基于 VITS，用了四个关键模块：
	1. prior encoder
	2. posterior encoder
	3. decoder
	4.  parallel bank of transposed convolutions (PBTC) 模块
3. 采用 Whisper 提取 bottleneck features (BNF) 作为 prior encoder 的输入
4. 对 source signal 采用 pitch perturbation 来移除 speaker timbre，从而避免 timbre 泄漏到 target speaker 中
5. PBTC 模块提取 multi-scale F0 作为 prior encoder 的辅助输入，用于捕获歌声中的韵律变化
6. 设计了一个三阶段的训练策略
7. 在 Task 1 和 2 上分别排名第一和第二

## Introduction

1. VC 的重点在于，将语音解耦为多个因子，包含 speaker timbre、linguistic content、speaking style，然后 linguistic content 和 speaking style 再和目标说话人的 timbre 结合实现转换
2. SVC 的关键在于建模 expressive singing styles 例如 temporal pitch variation；同时实现高的合成自然度和说话人相似度是很困难的
3. SVCC 2023 有两个任务：
	1. in domain SVC
	2. cross domain SVC
4. 本文基于 VITS，从 Whisper 的浅层 encoder 中提取 BNF，其中不仅包含 linguistic content 而且还有丰富的 style-related 信息（如韵律），对于 source speech 的 BNF 提取，提取之前先进行了 speech perturbation 来避免 timbre 泄漏；然后采用 PBTC 模块来提取多尺度的 F0；最后引入三阶段的训练策略

## 方法

### 概况

![](image/Pasted%20image%2020231226160326.png)

encoder 和 decoder 是一种 self-reconstruction 的结构，用于重构原始输入。prior encoder 融合 singer timbre、singing style 和 linguistic content，flow 用于连接先验和后验。

Posterior Encoder：采用 non-causal WaveNet residual blocks，输入为线性谱，posterior encoder 用于建模后验 $p(z|y)$，其中 $y$ 为 waveform。

Prior Encoder：结构为多层的 Transformer，给定 BNF 和 F0 作为输入，记为 $c_{bnf},c_{f0}$，则 prior encoder 和 后面的 flow 用于估计先验分布 $p(z|c_{bnf},c_{f0},singer)$（图中的 speaker ID 用于区分不同的说话人）。

Decoder：用的是 neural source filter 方案，包含 source module 和 filter module，source module 将 F0 转为 sine-based 激励信号，定义为：
$$e_t=\begin{cases}\alpha sin(\sum\limits_{k=1}^t2\pi\frac{f_k}{Ns}+\phi)+n_t,&f_t>0\\100n_t,&f_t=0\end{cases}$$
其中 $N_s$ 为采样率，然后 filter module 将激励信号和表征 $z$ 融合。

### BNF 提取

用的是 shallow encoder layer 的 BNF 特征，且在这之前采用 random pitch perturbations 。

### Multi-scale F0 建模

直接采用 F0 有时候会产生如 unnatural singing and out-of-tone 等问题，于是采用 PBTC 来提取 multi-scale F0。

如图：
![](image/Pasted%20image%2020231226194728.png)

PBTC 模块包含 VQ 模块、投影层 和一组 1D 转置卷积层，每个卷积的 dilation 都不同，后面还接了一个线性层。F0 序列首先量化为 $L$ bins，然后通过 one-hot 编码，然后通过线性投影层（增加特征维度），$K$ 个带有 $F$ 个滤波器的转置卷积层分别处理量化后的 F0，从而可以从不同的时间粒度处理 F0 信息，卷积得到的输出时间维度各不相同，但是都投影到 $T$，最后把所有的结果进行相加。

### 训练策略

一般是先在多说话人的歌声数据中做预训练，然后 adapt 到 目标说话人。

这里为了提高合成质量，还采用了 warmup 策略，总的训练如下：
+ Warm-up：在语音数据中初始化 SVC 模型
+ Pre-training：在歌声数据下预训练 SVC 模型
+ Adaption：adapt 到 目标说话人

adapt 的过程中可能会出现过拟合，于是增强目标说话人的数据，采用了四种数据增强函数：formant shifting, pitch randomization, random, frequency, and speed adjustment.

损失函数包括：

重构损失和 KL loss（CVAE 中的）：
$$\begin{gathered}\mathcal{L}_{KL}=\mathcal{D}_{KL}(q(z|y)\|p(z|c)),\\\mathcal{L}_{cvae}=\mathcal{L}_{recon}+\mathcal{L}_{KL}\end{gathered}$$
其中 $c$ 为  BNFs, F0 和 speaker ID 条件。

GAN loss 包含：
$$\begin{gathered}
\begin{aligned}L_{adv}(G)=\mathbb{E}_{(z)}[(D(G(z))-1)^2],\end{aligned} \\
L_{adv}(D)=\mathbb{E}_{(y,z)}[(D(y)-1)^2+(D(G(z)))^2]
\end{gathered}$$

总的在 warm-up 和 pre-training 阶段的 loss 为：
$$\begin{gathered}
L(G)=L_{adv}(G)+L_{cvae}, \\
L(D)=L_{adv}(D)
\end{gathered}$$

在 adapt 阶段，为了进一步避免过拟合，还采用了 weight regulation，定义为：
$$L_{wReg}=\|\theta-\hat{\theta}\|^2$$
目的是为了避免模型的参数变化过大。