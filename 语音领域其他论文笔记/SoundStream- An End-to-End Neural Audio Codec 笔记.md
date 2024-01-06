> TASLP 2021，Google

1. 提出 SoundStream，可以高效压缩语音、音频、音乐等
2. 基于全卷积的 encoder/decoder 网络和  residual vector quantizer，然后进行端到端联合训练
3. 引入对抗和重构损失，从而可以从量化的 embedding 中生成高质量的音频
4. 通过在量化层中引入 structured dropout ，可以实现从 3kbps 到 18kbps 的比特率，且性能损失几乎没有
5. 还可以实现低延迟，从而在实时 GPU 上支持流式的推理

## Introduction

1. audio codec 可以分为两类：
	1. waveform codecs：在 decoder 端重构，高比特率可以生成高质量音频，但是低比特率会有伪影
	2. parametric codecs：需要对音频做某种假设引入强先验，encoder 用于估计参数，然后量化，decoder 基于参数合成波形
2. 提出 SoundStream，全卷积编码器输入波形，生成低采样率下的 embedding 序列，然后通过 residual vector quantizer 量化，通过 全卷积解码器来重构波形
	1. 使用重建和对抗性损失进行端到端训练
	2. 联合训练一个（或多个）鉴别器，目的是将解码后的音频与原始音频区分开来
	3. 编码器和解码器都只使用因果卷积，latency 取决于 temporal resampling ratio
3. 贡献：
	1. 提出 SoundStream
	2. 引入一种新的 residual vector quantizer
	3. 效果优于 Opus 和 EVS
	4. 支持流式推理

结构如下：
![](image/Pasted%20image%2020230924212630.png)

## 相关工作（略）

## 模型

考虑单通道信号 $x\in \mathbb{R}^T$，采样率 $f_s$。SoundStream 有三个模块组成：
+ encoder，将 $x$ 映射到 embedding
+ RVQ：用一组有限的 codebooks 中的向量的和来替换每个 embedding，从而压缩表征
+ decoder：从量化后的 embedding 中产生重构波形 $\tilde{x}$

模型和一个 discriminator 端到端训练，还可以额外引入一个 conditioning signal，决定是在编码器还是解码器端应用去噪。

![](image/Pasted%20image%2020230924213856.png)

### encoder 架构

架构如上图，和 SEANet encoder 相同，但没有 skip connections。

包含一个 1D 卷积和 $B_{enc}$ 个卷积块，每个块又包含三个 residual unit，residual unit 又包含 dilation 卷积。

最后一个 1D 卷积将 embedding 维度变为 $D$。

为了保证 real-time inference，所有的卷积都是 causal 的，encoder 的输出维度 $\mathrm{enc}(x)\in\mathbb{R}^{S\times D},\text{with }{S}=T/M$，其中 $M$ 为所有的 stride 乘积。

### decoder 架构

架构和 encoder 相似，decoder block 和 encoder block 是镜像的，卷积层的 stride 也是镜像的，最后一个 1D 卷积后生成波形 $\hat{x}$，而且 encoder 和 decoder 中，通道相同的层参数共享，$C_{enc}=C_{dec}=C$。

### Residual Vector Quantizer

量化器的目标是将编码器的输出 $\mathrm{enc}(x)$ 压缩到目标比特率 $R$，单位为 bps。为了实现端到端的训练，需要通过 BP 和 encoder、decoder 联合训练。

vector quantizer 学习包含 $N$ 个 vector 的 codebook 来编码 $\mathrm{enc}(x)$ 的每帧（每帧的维度假设为 $D$ ），然后 $\text{enc}(x)\in\mathbb{R}^{S\times D}$ 就变成了长为 $S\times N$ 的 one-hot 向量，从而可以使用 $S\log_2N$ 比特来表示。

VQ 的缺点是，对于采样率较高的音频，需要很大的 codebook 。

![](image/Pasted%20image%2020230925102239.png)

于是提出 RVQ，级联 $N_q$ 个 VQ，未量化的输入通过第一个 VQ 并计算量化残差，残差由剩下的 $N_q-1$ 个 VQ 迭代量化。每个 VQ 需要量化的比特数为 $r_i=r/N_q=\log_2N$ 。

同时采用 VQ-VAE-2 中的 指数移动平均更新对每个量化器的  codebook 的训练。

训练时，没有对 codebook vector 进行随机初始化，而是在第一个 batch 上运行 k-means 算法， 将 k-means 得到的聚类中心作为初始化。当 codebook 中 的 vector 在好几个 batch 中都没被用上时，将其替换为当前 batch 中随机采样的帧。

但是这里有一个问题，不同的目标比特率就需要训练不同的 RVQ，于是为了实现在多个目标比特率下都可以运行，对每个样本，在 $[1,N_q]$ 中均匀采样一个 $n_q$ ，只使用 $[1,n_q]$ 之间的的 VQ，这可以看成是一种 structured dropout。因此，模型可以适用于于 $1$ 到 $N_q$ 的所有目标比特率编码和解码音频。 在推理过程中，$n_q$ 的值是根据所需的比特率选择的。

###  discriminator 架构

有两个 discriminator：
+ wave-based discriminator，输入为波形，采用 SEANet 中的 multi-resolution convolutional discriminator
+ STFT-based discriminator：输入为波形的 STFT，结构如图，输入是二维，输出是一维时域信号

![](image/Pasted%20image%2020230925110242.png)

### 目标函数

记 $\mathcal{G}(x)=\det(Q(\mathsf{enc}(x))$ 为 generator。

adversarial loss 用来提高感知质量，定义为 hinge loss。令 $k\in\{0,\dots,K\}$ 表示不同的 discriminator，$k=0$ 为 STF -based discriminator，其他表示 不同 resolution 的 wave-based discriminator，$T_k$ 为输出的 logits 个数（长度，时间），discriminator 的损失为：
$$\begin{gathered}\mathcal{L}_{\mathcal{D}}=E_x\left[\frac1K\sum_k\frac1{T_k}\sum_t\max\left(0,1-\mathcal{D}_{k,t}(x)\right)\right]+\\E_x\left[\frac1K\sum_k\frac1{T_k}\sum_t\max\left(0,1+\mathcal{D}_{k,t}(\mathcal{G}(x))\right)\right],\end{gathered}$$
generator 的对抗损失为：
$$\mathcal{L}_\mathcal{G}^{\mathrm{adv}}=E_x\left[\frac1K\sum_{k,t}\frac1{T_k}\max\left(0,1-\mathcal{D}_{k,t}(\mathcal{G}(x))\right)\right]$$

为了提高保真度，还有两个额外的损失：
“feature” loss，其实就是 generator 在 discriminator 特征空间下的相似度：
$$\mathcal{L}_{\mathcal{G}}^{\mathrm{feat}}=E_x\left[\frac1{KL}\sum_{k,l}\frac1{T_{k,l}}\sum_{t}\left|\mathcal{D}_{k,t}^{(l)}(x)-\mathcal{D}_{k,t}^{(l)}(\mathcal{G}(x))\right|\right]$$
其中，$L$ 为 internal layers 的层数。

multi-scale spectral reconstruction loss：
$$\begin{gathered}\mathcal{L}_{\mathcal{G}}^{\mathrm{rec}}=\sum_{s\in2^6,\ldots,2^{11}}\sum_t\|\mathcal{S}_t^s(x)-\mathcal{S}_t^s(\mathcal{G}(x))\|_1+\\\alpha_s\sum_t\|\log\mathcal{S}_t^s(x)-\log\mathcal{S}_t^s(\mathcal{G}(x))\|_2,\end{gathered}$$
其中 $\mathcal{S}_t^s(x)$ 为 64-bin 的mel 谱的第 $t$ 帧。

generator 总的 loss 就是前面的加权：
$$\mathcal{L}_G=\lambda_{\mathrm{adv}}\mathcal{L}_G^{\mathrm{adv}}+\lambda_{\mathrm{feat}}\cdot\mathcal{L}_G^{\mathrm{feat}}+\lambda_{\mathrm{rec}}\cdot\mathcal{L}_G^{\mathrm{rec}}.$$
实验中设置，$\lambda_{\mathrm{adv}}=1,\lambda_{\mathrm{feat}}=100,\lambda_{\mathrm{rec}}=1$。

### Joint compression and enhancement（略）


## 实验（略）