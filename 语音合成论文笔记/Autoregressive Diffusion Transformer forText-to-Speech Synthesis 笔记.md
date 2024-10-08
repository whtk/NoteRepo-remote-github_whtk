> preprint 2024,07，CUHK，李海洲

1. Audio tokenization 通常需要在 code bitrate 和 reconstruction accuracy 之间权衡
2. 提出在连续空间 $\mathbb{R}^d$ 中编码音频，并使用 decoder-only diffusion transformer（ARDiT）自回归生成这些序列：
    1. ARDiT 在 zero-shot TTS 上性能甚至超过 SOTA
    2. 高码率连续表征可以实现几乎无损重构
    3. 采用 IKL 散度在每个自回归 step 上进行蒸馏可以提高质量，将 diffusion 迭代采样过程压缩为一步
    4. 模型可以生成 170 ms 的 24 kHz 语音

## Introduction

1. 离散 audio tokenization 有一些问题：
    1. 需要在 bitrate 和重构质量之间权衡
    2. 离散分布下的 gradient-based optimization，现在通常需要使用辅助 loss 和 codebook re-initialization
2. 可以通过将 speech 表示为 $\mathbb{R}^d$ 中的向量序列（continuous tokens）来避免这些问题，一些方法如下：
    1. 使用 finite compositions 的 flows 来建模 $p_{\theta}(x_n|x_{<n})$
    2. 使用 Mixture Density Network（MDN）来建模 $p_{\theta}(x_n|x_{<n})$
    3. 使用 GAN 来建模 $p_{\theta}(x_n|x_{<n})$
    4. 使用 Diffusion probabilistic models（DPMs）来建模 continuous densities
3. diffusion 的蒸馏可以将其转换为 single-step 模型：
    1. SSD-LM 将 autoregressive sequence modeling 与 diffusion models 结合，使用 decoder-only transformer
    2. SSD-2 设计了 attention mask，提高了 ARDiTs 的训练效率
    3. 本文提出将 Distribution Matching Distillation（DMD）用于音频合成中的自回归 DiT 蒸馏
4. 贡献如下：
	+ 提出 ARDiT，一个 decoder-only 的 DiT，不需要离散 tokenization
	+ 采用 FIM 训练，ARDiT 在 zero-shot TTS 和 speech editing 上表现很好
	+ 使用 DMD 蒸馏 ARDiT TTS 模型，蒸馏后的模型只需要一次 evaluation 就可以生成一个或多个 continuous tokens
	+ 通过控制 RoPE 的旋转角度，控制 ARDiT TTS 生成语音总时长

## 相关工作（略）

## 方法

### 背景

假设 $X,Z$ 是独立的 $\mathbb{R}^d$ 值随机变量，其 PDF 为 $p(x)$，高斯密度为 $p(z) = \mathcal{N}(0, I_d)$。对于 $t \in [0,1]$，定义 $\alpha_t = (1-t)$，$\sigma_t = t$。令 $X_t = \alpha_tX + \sigma_tZ$，定义速度场 $v(x_t, t) : \mathbb{R}^d \times [0,1] \rightarrow \mathbb{R}^d$ 为：
$$v(x_t,t):=\underset{v}{\operatorname*{\arg\min}}E\left\|v(X_t,t)-(Z-X)\right\|_2^2=E[Z-X\mid X_t=x_t].$$
可以通过求解以下 ODE 来采样 $p(x)$：
$$\mathrm{d}Y_t=v(Y_t,t)\mathrm{d}t,\quad Y_1\thicksim\mathcal{N}(0,I_d),\quad t\in[0,1].$$

通过估计 $v(x_t, t)$ 为 $v_{\theta}(x_t, t)$，通过最小化 $E_{t\thicksim U[0,1]}\left\|v_{\theta}(X_t,t)-(Z-X)\right\|_2^2$ 来获得生成模型，其采 PDF 为 $p_{\theta}(x) \approx p(x)$。

假设 $X_t \thicksim p_t(x_t)$，可以证明 score function $s(x_t, t) = \nabla_{x_t} \log p_t(x_t)$ 可以从速度场 $v(x_t, t)$ 中得到：
$$v(x_t,t)=-\sigma_ts(x_t,t)-\frac{x_t+\sigma_t^2s(x_t,t)}{\alpha_t}=\frac{-1}{1-t}x_t+\frac{-t}{1-t}s(x_t,t).$$

这表明，Diffusion Probabilistic Models（DPMs），Flow Matching 模型都是估计 score function。

给定一个在 $p(x)$ 上训练的 Flow Matching 模型 $v_{\theta}(x_t, t)$。通过 ODE 建立了一个映射 $f_{\theta}(w) : \mathbb{R}^d \rightarrow \mathbb{R}^d$，将高斯噪声转换为数据样本。$f_{\theta}$ 计算涉及解 ODE，所以很慢。

DMD 可以将 $v_{\theta}$ 蒸馏为单步生成器 $g_{\xi} : \mathbb{R}^d \rightarrow \mathbb{R}^d$，将随机噪声 $W \sim \mathcal{N}(0, I_d)$ 映射到 $\widehat{X} = g_{\xi}(W)$，其密度为 $p_{\xi}(x)$。定义 $\widehat{X}_t := \alpha_t \widehat{X} + \sigma_t W$，其中 $W$ 是独立的高斯随机变量。假设 $p(x_t, t)$ 是 $X_t$ 的密度，$p_{\xi}(x_t, t)$ 是 $\widehat{X}_t$ 的密度。它们的 Integral Kullback–Leibler（IKL）散度定义为：
$$D_\xi:=D_{\mathrm{IKL}}\left(p_\xi(x_t,t)\|p(x_t,t)\right):=E_{t\thicksim\mathcal{U}[0,1]}\left[w_tD_{\mathrm{KL}}\left(p_\xi(x_t,t)\|p(x_t,t)\right)\right],$$

其中 $w_t \geq 0$ 是时间 $t$ 的权重因子。假设 $s(x_t, t) = \nabla_{x_t} \log p(x_t, t)$ 和 $s_{\xi}(x_t, t) = \nabla_{x_t} \log p_{\xi}(x_t, t)$。有：
$$\nabla_\xi D_\xi=E_{t\thicksim\mathcal{U}[0,1]}\left[w_t\alpha_t\left(s_\xi(\widehat{X}_t,t)-s(\widehat{X}_t,t)\right)\frac{\partial g_\xi(W)}{\partial\xi}\right].$$

$s_{\xi}(x_t, t)$ 和 $s(x_t, t)$ 是未知的，但是可以用 Flow Matching 模型 $v_{\eta}(x_t, t)$ 和 $v_{\theta}(x_t, t)$ 近似：
$$\mathcal{L}_\eta:=E_{t\thicksim\mathcal{U}[0,1]}\left\|v_\eta(\widehat{X}_t,t)-(\widehat{Z}-\widehat{X})\right\|_2^2.$$

假设 $v_{\eta}$ 和 $v_{\theta}$ 训练好了，有：
$$v_\eta(x_t,t)-v_\theta(x_t,t)\approx\frac{-t}{1-t}\cdot\left(s_\xi(x_t,t)-s(x_t,t)\right).$$

DMD 训练中，固定 $v_{\theta}$，采用以下 loss 函数交替更新 $g_{\xi}$ 和 $v_{\eta}$：
$$\mathcal{L}_{\xi}:=\underbrace{E_{t\sim\mathcal{U}[0,1]}\left\|\widehat{X}+\mathrm{sg}\left(v_\theta(\widehat{X}_t,t)-v_\eta(\widehat{X}_t,t)-\widehat{X}\right)\right\|_2^2}_{\mathcal{L}_{\mathrm{IKL}}}+\beta_{\mathrm{reg}}\cdot\underbrace{E\left\|g_\xi(W)-f_\theta(W)\right\|_2^2.}_{\mathcal{L}_{\mathrm{reg}}}$$

其中 $\beta_{\mathrm{reg}} > 0$ 是 L2 回归 loss 的权重，$\mathrm{sg}$ 是 stop gradient 操作符。为简单起见，设置 $w_t\alpha_t = \frac{2t}{1-t}$。此时，$\nabla_{\xi}\mathcal{L}_{\mathrm{IKL}} \approx \nabla_{\xi}D_{\xi}$。

### Contextual mel 谱 Autoencoder

结构如下：
![](image/Pasted%20image%2020241003161150.png)

将 log Mel spectrogram 用 autoencoder 压缩为连续 token 序列。给定随机 Mel spectrogram $Y \in \mathbb{R}^{N_{\text{frame}} \times D_{\text{mel}}}$，其中 $N_{\text{frame}}$ 是帧数，$D_{\text{mel}}$ 是 Mel 滤波器数，将 $Y$ 编码为连续 token 序列 $Z \in \mathbb{R}^{N_{\text{latent}} \times D_{\text{latent}}}$，其中 $N_{\text{latent}} = \left\lfloor\frac{N_{\text{frame}}}{4}\right\rfloor$，$D_{\text{latent}} = 16$。编码器是 transformer，输入 $Y$，输出 $\mu, \log\sigma \in \mathbb{R}^{N_{\text{latent}} \times D_{\text{latent}}}$。编码器定义了给定 $Y$ 的 $Z$ 的条件密度为 $q_{\phi}(z|y) = \mathcal{N}(z; \mu, \sigma^2)$。解码器是一个基于 DiT 的条件 Flow Matching 模型 $v_{\psi}(y_t; t, z)$，给定 $Z$，恢复 $Y$。定义潜在先验密度 $p(z) = \mathcal{N}(z; 0, 1)$，encoder 和 decoder 优化如下：
$$\mathcal{L}(\phi,\psi):=\beta_{\mathrm{M}}\cdot E\left[D_{\mathrm{KL}}\left(q_\phi(z|Y)\|p(z)\right)\right]+E_{W\sim\mathcal{N}(0,I)}\left\|v_\psi\left((1-t)Y+tW;t,Z\right)-(W-Y)\right\|_2^2.$$

第一项 $E[D_{\mathrm{KL}}(q_{\phi}(z|Y)\|p(z))]$ 是互信息 $I(Y; Z)$ 的变分上界。所以权重 $\beta_{\mathrm{MI}} > 0$ 控制编码速率和重构精度之间的权衡。在实验中，Mel spectrogram encoder 每秒发出 23.5 个 token，其理论比特率为 1.7 kbps。

为了在 Mel spectrogram 目标部分已知时进行条件解码，解码器在 Mel spectrogram masked reconstruction 上进行微调。在 speech editing 和 zero-shot TTS 的推断阶段，提供已知的 Mel spectrogram 帧给 decoder。

> 这部分本质是 autoencoder，将 Mel spectrogram 压缩为连续 token 序列，encoder 是 transformer，decoder 是 DiT。

### 基于 Autoregressive Diffusion Transformers 的 TTS
给定已经被 encode 的 Mel spectrogram $Y$，得到连续 token 序列 $Z = [Z_0; \cdots; Z_{N_{\text{latent}}-1}] \in \mathbb{R}^{N_{\text{latent}} \times D_{\text{latent}}}$。假设 $C = [C_0, \cdots, C_{N_{\text{phone}}-1}] \in \Sigma^{N_{\text{phone}}}$ 是 $Y$ 的 phoneme 序列，其中 $\Sigma$ 是所有音素的集合。

ARDiT 是半自回归的，通过估计条件向量场 $v_{\theta}({z_t^{i:i+B}};t, c, z^{<i})$ 来从 $p_{\theta}(z^{i:i+B}|c, z^{<i})$ 中采样。这里 $B \in \mathbb{N}_+$ 是 block 大小，$i \in \mathbb{N}_+$ 是 block 中第一个 token 的索引。假设 $W \in \mathbb{R}^{N_{\text{latent}} \times D_{\text{latent}}}$ 是独立随机变量，其密度为 $p(w) = \Pi_{n,d}\mathcal{N}(w_{n,d}; 0, 1)$。令 $Z_t = (1-t)Z + tW$。ARDiT 的训练 loss 为：
$$\mathcal{L}(\theta):=E_{i,t\sim\mathcal{U}[0,1]}\left\|v_\theta\left(Z_t^{i:i+B};t,C,Z^{<i}\right)-\left(W^{i:i+B}-Z^{i:i+B}\right)\right\|_2^2.$$

一种简单的方法是，将 $(C, Z^{<i}, Z^{i:i+B}_t)$ 输入 transformer，然后将输出的最后 $B$ 个向量组合在一起得到 $v_{\theta}(Z^{i:i+B}_t; t, C, Z^{<i})$。
> 这种实现在训练和采样中都不如离散 token 上的 LMs 高效。在训练中，不支持 teacher-forcing。每个 batch 的 $\nabla_{\theta}\mathcal{L}(\theta)$ 只依赖于模型输出中的一个长度为 $B$ 的小段。推理时，不支持 KV-cache，导致不必要的重计算。

对 transformer 的输入序列和 attention mask 进行特殊设计。首先，将 $Z$ 分成大小为 $B$ 的 block。定义第 $i$ 个 token 的 block 索引为 $\#i := \left\lfloor\frac{i+S}{B}\right\rfloor$，其中 $S \in \{0, \cdots, B-1\}$ 是整数，表示 block shift。假设有 $M$ 个 block。对于每个 block $m$，$Z^{b_m:e_m}$ 表示该 block 中的 token，其中 $b_m$ 和 $e_m$ 分别表示 block 的开始和结束索引。对于每个 block $m$，选择时间 $t_m \in [0, 1]$。定义 $t := [t_0, \cdots, t_{M-1}]$。定义 $Z_t$，其中 $Z^{b_m:e_m}_t := (1-t_m)Z_{b_m:e_m} + t_mW_{b_m:e_m}$。

ARDiT 训练过程如下：
![](image/Pasted%20image%2020241007120812.png)

将 $(C, Z, Z_t)$ 拼接后输入 transformer。attention mask 规则如下：$C$ 中的 token 可以 attend 到 $C$ 中的 token；$Z$ 中的 token 可以 attend 到 $C$ 和 block 索引小于等于自己的 $Z$ 中的 token；$Z_t$ 中的 token 可以 attend 到同一 block 索引的 $Z_t$ 中的 token 和 block 索引小于自己的 $Z$ 中的 token。


ARDiT 推理过程如下：
![](image/Pasted%20image%2020241007121607.png)

假设正在生成 block $m$。ARDiT 的输入是 $(C, Z^{<e_m-1}, Z^{b_m:e_m}_t)$。attention mask 规则如下：$C$ 中的 token 可以 attend 到 $C$ 中的 token；$(Z^{<e_m-1}, Z^{b_m:e_m}_t)$ 中的 token 可以 attend 到 $C$ 和其他 block 索引小于等于自己的 token。

比较 ARDiTs 和相同模型大小的 decoder-only transformers（LMs）的计算复杂度。对于 LMs，用相同数量的离散 token 替换连续 token。在训练期间，ARDiT 每个 utterance 处理 $N_{\text{phone}} + 2 \cdot N_{\text{latent}}$ 个 token，而 LM 处理 $N_{\text{phone}} + N_{\text{latent}}$ 个 token。在推理期间，具有 KV cache 的 ARDiT 需要大约比 LM 多 $N_{\text{FE}} + 1$ 倍的计算量和 $N_{\text{FE}}/B$ 倍的网络评估次数，其中 $N_{\text{FE}}$ 是 ODE solver 的平均函数评估次数。

由于 ARDiT 推理计算和网络评估次数都随 $N_{\text{FE}}$ 线性增长，因此减少 $N_{\text{FE}}$ 对于实际应用至关重要。本文应用 Distribution Matching Distillation（DMD）来将 $N_{\text{FE}}$ 减少到 1。ARDiT 的 DMD 训练略。

> 用 fill-in-the-middle（FIM）训练技术，使得 ARDiT 支持 speech editing。细节见论文。

### Position Embeddings 和总时长控制

为了控制 ARDiT 生成的语音总时长，使用 position embeddings 来 inform ARDiT 总语音时长 $N_{\text{latent}}$。ARDiT 模型基于 Rotary Position Embedding（RoPE）。RoPE 通过旋转 self-attention 中的 key 和 value 向量来编码相对位置信息。具体来说，对于任意位置索引 $n \in \mathbb{R}$，定义旋转矩阵 $R_n$ 如下：
$$U_\theta:=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix},\quad R_n:=\begin{bmatrix}U_{n\theta_0}&0&\cdots&0\\0&U_{n\theta_1}&\cdots&0\\\varvdots&\varvdots&\ddots&\varvdots\\0&0&\cdots&U_{n\theta_{d-1}}\end{bmatrix}\in\mathbb{R}^{2d\times2d}.$$

类似于 VALL-E，phoneme token 和 speech token 使用不同的 position embeddings。对于 phoneme token $C = [C_0, \cdots, C_{N_{\text{phone}}-1}]$，每个 token $C_i$ 被赋予位置索引 $i$。对于 speech token $(Z, Z_t, Z_t)$，token $Z^i, Z^{i}_t, Z^{i}_t$ 被赋予分数位置索引 $i \cdot \eta$，其中 $\eta = \frac{N_{\text{phone}}}{N_{\text{latent}}}$ 是 speech rate。例如，如果 $(N_{\text{phone}}, N_{\text{latent}}) = (10, 20)$，则 speech rate 是 $\eta = 0.5$，第 $i$ 个 speech token 的分数位置索引是 $0.5i$。

这种设计类似于 ParaNet，发现可以加速 ARDiT 模型的训练。有效地缓解了生成过长序列的问题。在推理阶段，ARDiT 从参考语音中估计语音时长 $N_{\text{latent}}$。

## 实验（略）
