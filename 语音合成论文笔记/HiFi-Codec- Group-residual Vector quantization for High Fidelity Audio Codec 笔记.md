> prerprint，北大、腾讯 AI Lab、浙大

1. 把 audio codec 用于音频生成领域的两个挑战有：
	1. 缺乏可用的训练过程、需要大量数据和 GPU
	2. 要实现好的重构性能，需要很多 codebook
2. 本文提出一种 group-residual vector quantization（GRVQ）技巧，同时开发了一个 High Fidelity Audio Codec model, HiFi-Codec，只需 4 个 codebook
3. 用已有的数据集如 LibriTTS, VCTK, AISHELL 等训练（总时间超过 1000h），8个 GPU 
4. 4 个 codebook 超过 Encodec
5. 提出 AcademiCodec，开源的工具包，开源训练  Encodec, SoundStream, 和 HiFi-Codec

> GRVQ 其实就是把表征 split 成两部分，然后分别做 RVQ。。。
> 其他创新点几乎没有，最大的贡献是 开源 AcademiCodec

## Introduction

1. 作者实验发现，大部分的信息都在 RVQ 的第一个 codebook 中，其他的 codebook 也包含一些细节，但是是稀疏的
2. 本文关注于设计用于生成任务的 audio codec，目标是：
	1. 好的重构性能
	2. codebook 尽可能少
3. 提出 GRVQ 来实现上面的功能

## 相关工作（略）

## 方法

### 概览

考虑时间长度为 $d$ 的单通道语音信号 $\boldsymbol{x}\in\mathcal{R}^T$，且 $T=d\times sr$，sr 为采样率。

HiFi-Codec 包含三个组件：
+ encoder $E$ 输入 audio 输出 latent representation $\boldsymbol{z}$
+ GRVQ 层 产生压缩表征 $\boldsymbol{z}_q$
+ decoder $G$ 从 $\boldsymbol{z}_q$ 中重构信号 $\hat{\boldsymbol{x}}$

模型以端到端的方式进行训练，在时域和频域优化重构损失，还有一个 discriminator 的感知损失。

结构如图：
![](image/Pasted%20image%2020231118173058.png)

### encoder 和 decoder

encoder 包含 1D 卷积+ B 个 卷积块，后面接两层 LSTM 和一个最终的 1D 卷积。

decoder 镜像 encoder 的结构。

### GRVQ

RVQ 的缺点在于，第一层的 codebook 包含太多信息，于是提出在第一层中增加更多的 codebook。

具体来说，对于给定的 latent representation $\boldsymbol{z}$，首先将其平均分为几组（本文分为两组，$\boldsymbol{z}_1,\boldsymbol{z}_2$），然后采用 RVQ 来量化每个组的特征，最后将两组量化后的特征组合起来得到最终的量化表征，算法如下：
![](image/Pasted%20image%2020231118173815.png)

> 就这？？

### Discriminator

用三个 discriminator：
+ Encodec 中的 multi-scale STFT-based (MS-STFT) 
+ HiFi-GAN 中的 multi-period discriminator (MPD)
+ HiFi-GAN 中的 multi-scale discriminator (MSD)

### 训练损失

基于 GAN 的目标函数，优化 generator 和 discriminator。

联合优化：
+ 重构损失
+ 感知损失
+ GRVQ  commitment loss

#### 重构损失

包含时域损失和频域损失。

时域就是预测语音和真实的语音之间的 L1 loss。

频域类似于 encodec，用多尺度的 mel 谱 loss。

#### 判别器损失

对抗损失用于提高感知质量。

MS-STFT 用于区分 mel 谱，MPD 和 MSD 用于区分波形。

目标函数为：
$$\mathcal{L}_d=\frac1K\sum_{i=1}^Kmax(0,1-D_k(\boldsymbol{x}))+max(0,1+D_k(\boldsymbol{\hat{x}}))$$
其中 $K$ 为 discriminator 的数量。

也可以定义为 hinge loss：
$$\mathcal{L}_{adv}=\frac1K\sum_{i=1}^Kmax(0,1-D_k(\boldsymbol{\hat{x}}))$$

最后 feature loss 为中间特征的损失：
$$\mathcal{L}_{feat}=\frac{1}{KL}\sum_{k=1}^{K}\sum_{l=1}^{L}\frac{||D_k^l(\boldsymbol{x})-D_k^l(\boldsymbol{\hat{x}})||_1}{mean(||D_k^l(\boldsymbol{x})||_1)}$$

#### GRVQ Commitment Loss

对于第 $i$ 个组的第 $c$ 个 quantizer，commitment loss 计算如下：
$$\mathcal{L}_c=\sum_{i,c}||\boldsymbol{z}_{i,c}-q_{i,c}(\boldsymbol{z}_{i,c})||_2^2$$

此时 generator 的总损失为：
$$Loss_G=\lambda_{\text{adv}}\mathcal{L}_{adv}+\lambda_{\text{feat}}\cdot\mathcal{L}_{feat}+\lambda_{\text{rec}}\cdot\mathcal{L}_{rec}+\lambda_c\cdot\mathcal{L}_c$$

## 实验（略）

用的是 语音增强 的评估指标：PESQ 和 STOI。

数据集包括：LibriTTS, VCTK, AISHELL，主要是中文和英文数据。

