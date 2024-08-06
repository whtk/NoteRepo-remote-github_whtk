> KRAFTON，ICLR 2024

1. audio tokenization 使得长序列和多序列建模很复杂
2. 本文提出 CLaM-TTS，采用 probabilistic residual vector quantization，实现 token 长度压缩，使得 LM 一次生成多个 token
3. 实验结果表明 CLaM-TTS 在自然度、可懂性、说话者相似性和推理速度上优于或与最先进的基于 neural codec 的 TTS 模型相当

> 很神奇，LM 自回归用的是连续的 latent，合成 mel 谱那块反而用的是 code。

## Introduction

1. 在图像处理中，通过减少输入长度可以缓解训练和推理中的挑战
2. 通过在数万小时的多样音频数据上训练 LLM，可以实现 zero-shot TTS
3. 提出 CLaM-TTS，将 speech 编码为多个 token 序列，每个 time step 的多个 token 通过单个 LM 自回归生成：
    + 核心在于 probabilistic discrete representation learning，确保所有离散 code 都参与训练
    + 提出 principled framework，采用 latent language 方法，使 latent LM 一次生成一堆 token
    + 训练数据：100K 小时
    + 实验结果表明 CLaM-TTS 在自然度、可懂性、说话者相似性和推理速度上优于或与最先进的 zero-shot neural codec-based TTS 模型相当

## 相关工作（略）

## 预备知识

基于 [VALL-E- Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers 笔记](VALL-E-%20Neural%20Codec%20Language%20Models%20are%20Zero-Shot%20Text%20to%20Speech%20Synthesizers%20笔记.md)，考虑两种数据：文本和其对应的语音的 mel-spectrogram，分别用 $\boldsymbol{x}$ 和 $\boldsymbol{y}$ 表示。通过 VAE 和 RVQ 模型，从 mel-spectrogram 的 latent representation $\boldsymbol{z}_{1:T}$ 建模离散 code 序列 $\boldsymbol{c}_{1:T} = \{c_1, \ldots, c_T\}$，其中 $c_t$ 表示深度为 $D$ 的量化离散 code。然后，使用 LM $p_{\theta}(\boldsymbol{c}_{1:T} | \boldsymbol{x})$ 预测 $\boldsymbol{c}_{1:T}$。推理时，语言模型为给定文本 $\boldsymbol{x}$ 生成 $\boldsymbol{c}_{1:T}$，通过 VAE decoder 和预训练 vocoder 转换为语音。

### RQ-VAE

RQ-VAE 包含三个部分：
+ encoder ${\phi}$ 将数据 $\boldsymbol{y}$ 映射为 latent representations 序列 $\boldsymbol{z}_{1:T}$
+ residual vector quantizer $RQ_{\psi}(\cdot)$ 将每个时间步 $t$ 的 latent vector $\boldsymbol{z}_t$ 转为离散 code  $\boldsymbol{c}_{t,1:D} = RQ_{\psi}(\boldsymbol{z}_t)$ 或对应的量化 embedding $\hat{\boldsymbol{z}}_t$
+ decoder ${\omega}$ 从一系列量化的 latent representations $\hat{\boldsymbol{z}}_{1:T}$ 重构数据 $\boldsymbol{y}$

其中 $\boldsymbol{c}_{t,1:D}$ 表示集合 $\{c_{t,1}, \ldots, c_{t,D}\}$，$D$ 表示量化器的深度。通过多阶段最近邻查找 codebook embeddings 近似量化 encoder 的 latent representation，codebook 的 vocab size 为 $V$。该过程定义为在每个深度 $d$ 最小化残差误差的 code：
$$c_{t,d}=\underset{c^{\prime}\in\{1,...,V\}}{\operatorname*{\arg\min}}\left\|\boldsymbol{r}_{t,d-1}-e_\psi(c';d)\right\|^2,\quad\boldsymbol{r}_{t,d}=\boldsymbol{r}_{t,d-1}-e_\psi(c_{t,d};d)\quad\text{for all }d\in[1,D],$$
其中 $\boldsymbol{r}_{t,0} = \boldsymbol{z}_t$，$e_{\psi}(c; d)$ 为深度 $d$ 的 codebook 中第 $c$ 个 embedding vector。embedding 的和 $\sum_{d=1}^{D} e_{\psi}(c_{t,d}; d)$ 为量化的 latent representation $\hat{\boldsymbol{z}}_t$。通过指数移动平均更新 codebook embeddings。

### Mean-Field Variational Inference

考虑一个 latent variable model，其联合分布 $p_{\psi}(\boldsymbol{z}_t, \boldsymbol{c}_{t,1:D})$ 由 $\psi$ 参数化。其中 $\boldsymbol{z}_t$ 表示观测到的随机变量，$\boldsymbol{c}_{t,1:D}$ 表示 latent 随机变量 $\{c_{t,1}, \ldots, c_{t,D}\}$。本文通过变分推断，求分布 $q(\boldsymbol{c}_{t,1:D} | \boldsymbol{z}_t)$ 的参数，来近似不可求的分布 $p_{\psi}(\boldsymbol{c}_{t,1:D} | \boldsymbol{z}_t)$。ELBO 为：
$$\log p_\psi(\boldsymbol{z}_t)=\log\sum_{\boldsymbol{c}_{t,1:D}}p_\psi(\boldsymbol{z}_t|\boldsymbol{c}_{t,1:D})p(\boldsymbol{c}_{t,1:D})\geq\mathbb{E}_{q(\boldsymbol{c}_{t,1:D}|\boldsymbol{z}_t)}\bigg[\log\frac{p_\psi(\boldsymbol{z}_t|\boldsymbol{c}_{t,1:D})p(\boldsymbol{c}_{t,1:D})}{q(\boldsymbol{c}_{t,1:D}|\boldsymbol{z}_t)}\bigg].$$

mean-field 变分推断是一种特定的变分推断，假设在观测变量条件下，latent 变量之间相互独立：$q(\boldsymbol{c}_{t,1:D} | \boldsymbol{z}_t) = \prod_{d=1}^{D} q(c_{t,d} | \boldsymbol{z}_t)$。每个最优 variational posterior 分布 $q^*(c_{t,d} | \boldsymbol{z}_t)$ 满足：
$$q^*(\boldsymbol{c}_{t,d}|\boldsymbol{z}_t)\propto\exp\left(\mathbb{E}_{q(\boldsymbol{c}_{t,-d}|\boldsymbol{z}_t)}[\log p_\psi(\boldsymbol{z}_t|\boldsymbol{z}_{t,d},\boldsymbol{z}_{t,-d})p(\boldsymbol{c}_{t,d},\boldsymbol{c}_{t,-d})]\right),$$
其中 $\boldsymbol{c}_{t,-d}$ 表示除了 $c_{t,d}$ 之外的所有深度 $c_{t,1:D}$ 的 latent 变量。基于上式的 iterative coordinate ascent 算法用于更新分布 $q$，算法的复杂度主要在于计算 $q(\boldsymbol{c}_{t,-d} | \boldsymbol{z}_t)$ 上的期望。

## 方法

### Mel-VAE

目标是 neural codec，可以生成较短长度的离散 codes。采用 RQ-VAE 压缩音频的 mel-spectrograms。引入基于 variational inference 的方法学习 residual codewords，解决传统 vector quantization 方法中的 codeword collapse 问题。

Mel-VAE 与 RQ-VAE 类似，encoder 将 mel-spectrogram $\boldsymbol{y}$ 映射为 latent representations 序列 $\boldsymbol{z}_{1:T}$，residual vector quantizer $RQ_{\psi}(\cdot)$ 将每个时间步 $t$ 的 latent vector $\boldsymbol{z}_t$ 转为离散 code $\boldsymbol{c}_t$ 或对应的量化 embedding $\hat{\boldsymbol{z}}_t$。decoder 从一系列量化的 latent representations $\hat{\boldsymbol{z}}_{1:T}$ 重构 mel-spectrogram $\boldsymbol{y}$。

假设 $q(\boldsymbol{c}_{t,1:D} | \boldsymbol{z}_t) = \prod_{d=1}^{D} q(c_{t,d} | \boldsymbol{z}_t)$，$p(\boldsymbol{c}_{t,d}, \boldsymbol{c}_{t,-d})$ 是均匀分布，mean-field variational inference 得到分布的条件：
$$q^*(\boldsymbol{c}_{t,d}|\boldsymbol{z}_t)\propto\exp(\mathbb{E}_{q(\boldsymbol{c}_{t,-d}|\boldsymbol{z}_t)}\left[\log p_\psi(\boldsymbol{z}_t|\boldsymbol{c}_{t,d},\boldsymbol{c}_{t,-d})\right]),$$
其中 latent 服从正态分布：$p_{\psi}(\boldsymbol{z}_t | \boldsymbol{c}_t) = \mathcal{N}(\boldsymbol{z}_t ; \sum_{d} e_{\psi}(c_{t,d}; d), \sigma_{\psi}^2 \mathbf{I})$。

但是不同深度的 code 之间的存在相互依赖性，难以不使用迭代方法解决。于是近似 $\mathbb{E}_{q(\boldsymbol{c}_{t,-d}|\boldsymbol{z}_t)}[\log p_{\psi}(\boldsymbol{z}_t | \boldsymbol{c}_{t,d}, \boldsymbol{c}_{t,-d})]$ 为 $\log p_{\psi}(\boldsymbol{z}_t | \boldsymbol{c}_{t,d}, \boldsymbol{c}^*_{t,-d})$，其中 $\boldsymbol{c}^*_{t,1:D} = RQ_{\psi}(\boldsymbol{z}_t)$。后验分布为：$q^*(\boldsymbol{c}_{t,d} | \boldsymbol{z}_t) \propto p_{\psi}(\boldsymbol{z}_t | \boldsymbol{c}_{t,d}, \boldsymbol{c}^*_{t,-d})$。

在 variational inference 框架中，独立优化每个深度 $d$ 的 codebook embeddings：
$$\begin{aligned}&\mathcal{L}(\psi_d;\boldsymbol{z}_t,\boldsymbol{c}_{t,-d}^*)=\mathbb{E}_{q^*(\boldsymbol{c}_{t,d}|\boldsymbol{z}_t)}\left[\boldsymbol{-}\log p_\psi(\boldsymbol{z}_t|\boldsymbol{c}_{t,d},\boldsymbol{c}_{t,-d}^*)\right],\\&\mathcal{L}(\psi;\boldsymbol{z}_t,\boldsymbol{c}_{t,1:D}^*)=\sum_{d=1}^D\mathcal{L}(\psi_d;\boldsymbol{z}_t,\boldsymbol{c}_{t,-d}^*).\end{aligned}$$

Mel-VAE 的其他模块，包括 encoder 参数 ${\phi}$ 和 decoder 参数 ${\omega}$，通过 commitment loss、reconstruction loss 和 adversarial losses 进行训练：
$$\mathcal{L}(\omega,\phi;\boldsymbol{y},\boldsymbol{c}_{t,1:D})=\lambda_r|\boldsymbol{y}-\boldsymbol{\hat{y}}|+\lambda_c\|\boldsymbol{z}-\sum_de_\psi(\boldsymbol{c}_{t,d};d)\|^2+\lambda_a\mathcal{L}_{adv},$$

其中 $\lambda_r$、$\lambda_c$ 和 $\lambda_a$ 分别对应重构损失、commitment loss 和 adversarial losses 的系数。对于 adversarial training，采用 multi-length discriminator 和 modified multi-resolution spectrogram discriminator。将 least squares GAN objective 和 L1 feature matching loss 结合为 loss function $L_{adv}$。

### Latent Language Model

给定文本 $\boldsymbol{x}$ 作为条件来生成语音 code。speech codes 通过 VQ 生成，但是预测的是 vector 本身，然后通过 RVQ 将其转换为每层的多个 token，从而与之前逐个预测 speech code tokens 的方法不同。
> 啊？

具体来说，不是直接从文本预测 token，而是考虑 speech 的连续 latent representation $\boldsymbol{z}_t$，通过 RVQ 转换为 speech code：
$$p_\theta(\boldsymbol{c}_{1:T}|\boldsymbol{x})=\prod_{t=1}^Tp_\theta(\boldsymbol{c}_t|\boldsymbol{x},\boldsymbol{c}_{<t})=\prod_{t=1}^T\int p_\theta(\boldsymbol{c}_t,\boldsymbol{z}_t|\boldsymbol{x},\boldsymbol{c}_{<t})d\boldsymbol{z}_t=\prod_{t=1}^T\int p_\theta(\boldsymbol{z}_t|\boldsymbol{x},\boldsymbol{c}_{<t})p_\psi(\boldsymbol{c}_t|\boldsymbol{z}_t)d\boldsymbol{z}_t,$$
其中 $\boldsymbol{c}_{<t}$ 表示 $\boldsymbol{c}_{1:t-1}$，$p_{\psi}(\boldsymbol{c}_t | \boldsymbol{z}_t)$ 作为 $p_{\theta}(\boldsymbol{c}_t | \boldsymbol{z}_t, \boldsymbol{x}, \boldsymbol{c}_{<t})$ 的替代。定义条件分布 $p_{\theta}(\boldsymbol{z}_t | \boldsymbol{x}, \boldsymbol{c}_{<t})$ 为高斯混合模型：
$$p_\theta(\boldsymbol{z}_t|\boldsymbol{x},\boldsymbol{c}_{<t})=\sum_{k=1}^Kp_\theta(k|\boldsymbol{x},\boldsymbol{c}_{<t})\mathcal{N}(\boldsymbol{z}_t;\mu_\theta(k,\boldsymbol{x},\boldsymbol{c}_{<t}),\sigma_\psi^2I).$$

在该模型中，可以得到对数似然的变分下界：
$$\begin{aligned}
\log p_\theta(\boldsymbol{c}|\boldsymbol{x})& \Large\geq\sum_{\large t=1}\mathbb{E}_{q(k|\boldsymbol{x},\boldsymbol{c}_{\leq t})}\left[\boldsymbol{-}D_{KL}(p_\psi(\boldsymbol{z}_t|\boldsymbol{c}_t)||p_\theta(\boldsymbol{z}_t|\boldsymbol{x},\boldsymbol{c}_{<t},k))\boldsymbol{+}\log p_\theta(k|\boldsymbol{x},\boldsymbol{c}_{<t})\boldsymbol{+}\mathcal{B}(\psi,\boldsymbol{c}_t)\right]  \\
&=-\mathcal{L}_{\mathrm{VB}}(\theta)+\mathcal{B}(\psi,\boldsymbol{c}_t),
\end{aligned}$$

对于任何 $q(k|\boldsymbol{x}, \boldsymbol{c}_{\leq t})$，可以得到上界的推导和 $\mathcal{B}(\psi, \boldsymbol{c}_t)$ 的定义。令 $q(k|\boldsymbol{x}, \boldsymbol{c}_{\leq t}) \propto \exp(-D_{KL}(p_{\psi}(\boldsymbol{z}_t | \boldsymbol{c}_t) || p_{\theta}(\boldsymbol{z}_t | \boldsymbol{x}, \boldsymbol{c}_{<t}, k)))$。

第二个损失 $L_{EOS}(\theta)$ 与训练二元分类器以识别 speech 结束 (EOS) 有关，训练 latent language model 的总损失为上述两个损失之和：$L(\theta) = L_{VB}(\theta) + L_{EOS}(\theta)$。


如图：
![](image/Pasted%20image%2020240806171353.png)

自回归 latent model 有三个不同的输出：
+ 高斯混合分布的权重
+ 高斯混合分布的均值
+ 生成 EOS 的概率

包括 transformer decoder 和三个并行模块：
+ 一个 softmax 激活的 prediction layer 用于 $p_{\theta}(k | \boldsymbol{x}, \boldsymbol{c}_{<t})$；
+ 一个 prediction layer 用于 $\mu_{\theta}(k, \boldsymbol{x}, \boldsymbol{c}_{<t})$；
+ 一个预测 EOS 的二元分类器层。

此外，使用 Mel-VAE 的预训练 quantizer $RQ_{\psi}(\cdot)$。

### 模型架构和推理

模型架构：
+ Mel-VAE：采用 causal 1d convolutional U-Net，移除 skip connections 和 attention layers，将 1-d ConvNeXt blocks 附加到 decoder 的最后一层。使用 32-stage residual vector quantization，每个深度的 codebook 大小为 1,024。
+ text-to-code latent language model：采用 transformer-based encoder-decoder LM，特别是预训练的 ByT5-large2，保持 text encoder 冻结。

推理：
1. 在时间步 $t$，从分布 $p_{\theta}(k | \boldsymbol{x}, \boldsymbol{c}_{<t})$ 随机选择 $k$；
2. 随后，从 $p_{\theta}(\boldsymbol{z}_t | \boldsymbol{x}, \boldsymbol{c}_{<t}, k)$ 随机采样 latent vector $\boldsymbol{z}_t$。因此，在时间步 $t$，通过学习的 quantizer 得到离散 code：$c_t = RQ_{\psi}(\boldsymbol{z}_t)$；
3. 如果 EOS 的概率超过 0.5，则结束生成，否则继续。随后，通过 Mel-VAE 的 decoder 将生成的 codes 解码为 mel-spectrograms，然后通过预训练 vocoder BigVGAN 将其转换为原始波形。

## 实验(略)
