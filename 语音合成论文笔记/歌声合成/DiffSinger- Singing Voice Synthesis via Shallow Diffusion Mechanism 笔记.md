> AAAI，2022，ZJU

1. 歌声语音合成（SVS）从 music score（乐谱）中生成歌声
2. 之前的 singing acoustic model 都采用简单的损失或者 GAN 来重构 acoustic feature，但是分别会有 over-smoothing 和 训练不稳定的问题
3. 提出 DiffSinger，基于 music score 迭代地将噪声转换为 mel 谱，引入 shallow diffusion mechanism 来更好地利用先验知识
4. 提出 boundary prediction  方法来定位 intersection，自适应地决定 shallow step
5. 在 Chinese singing dataset 上超过了 SOTA 的工作

## Introduction

1. SVS 通常包含一个 acoustic model  基于 music score 来生成 acoustic feature（如 mel 谱），一个 vocoder 来生成 waveform
2. 提出 DiffGAN，用于 SVS  的 acoustic model，基于 music score 将噪声转为 mel 谱
3. 为了进一步提高质量和加快推理速度，引入 shallow diffusion mechanism，发现 ground-truth mel 谱 $M$ 的 diffusion trajectories 和 使用简单的 mel 谱 decoder 预测得到的 $\widetilde{M}$ 的 trajectories 之间有交叉，因此在推理阶段：
	1. 使用简单的 mel 谱 decoder 生成 $\widetilde{M}$
	2. 通过 diffusion process 计算一个 shallow step $k$ 下的样本 $\widetilde{M}_k$
	3. 从 $\widetilde{M}_k$ 开始进行 reverse process 而非从白噪声开始，然后通过 $k$ 步完成迭代
4. 训练一个 boundary prediction network 来定位 intersection，从而自适应地决定 $k$
5. 同时开发了 DiffSpeech 用于 TTS

## Diffusion

## DiffSinger

![](image/Pasted%20image%2020231002095839.png)

DiffSinger 基于 Diffusion，SVS 任务需要建模条件分布 $p_\theta(M_0|x)$，$M$ 为 mel 谱，$x$ 为 music score。

### 简单的 DiffSinger

简单的 DiffSinger 是图二中去掉虚线框的部分。训练时，输入为时刻 $t$ 的 mel 谱 $M_t$，然后预测其中的噪声 $\epsilon_\theta(\cdot)$，推理时，从高斯白噪声开始，经过 $T$ 次迭代生成最终的 mel 谱：
$$M_{t-1}=\frac1{\sqrt{\alpha_t}}\left(M_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(M_t,x,t)\right)+\sigma_t\mathbf{z},$$

### Shallow Diffusion Mechanism

![](image/Pasted%20image%2020231002101341.png)

如上图：分别是预测的 mel 谱 和 GT mel 谱，可以发现：
+ 当 $t=0$ 是，$M$ 中相邻谐波之间细节丰富，但是 $\widetilde{M}$ over-smoothing 了
+ 随着 $t$ 的增加，两者才逐渐变得不可分

解释如下：
![](image/Pasted%20image%2020231002101828.png)

来自 $\widetilde{M}$ 流型的路径和来自 $M$ 流型的路径会在某个某个较大的 step 相交。

于是提出 shallow diffusion mechanism，reverse process 从两个路径的相交点开始（而非白噪声开始），从而可以减少 reverse process 的负担，具体来说，在推理阶段：
+ 采用一个辅助的 decoder 来生成 $\widetilde{M}$（采用 L1 condition loss），如图 1 a 虚线框所示
+ 产生 step $k$ 下中间样本，但是是通过 （forward）diffusion process 来生成的：$\widetilde{M}_k(\widetilde{M},\boldsymbol{\epsilon})=\sqrt{\bar{\alpha}_k}\widetilde{M}+\sqrt{1-\bar{\alpha}_k}\boldsymbol{\epsilon}$
+ 如果 $k$ 选的好的话，那么 $\widetilde{M}_k$ 和 $M_k$ 可以认为来自相同的分布
+ 从 $\widetilde{M}_k$ 开始进行 reverse process，此时只需要完成 $k$ 次的迭代即可

### Boundary Prediction

提出使用 boundary predictor 来自适应地学习 $k$。包含：
+ 一个分类器
+ 一个在 mel 谱 中添加噪声的模块

对于 $t\in[1,T]$，定义 $M_t$ 的 label 为 1，$\widetilde{M}_t$ 的标签为 0，采用交叉熵来训练模型，来判断时刻 $t$ 的输入 mel 谱 是来自 $\widetilde{M}_t$ 还是 $M_t$，损失为：
$$\begin{gathered}\mathbb{L}_{BP}=-\mathbb{E}_{M\in\mathcal{Y},t\in[0,T]}[\log BP(M_t,t)+\\\log(1-BP(\widetilde{M}_t,t))]\end{gathered}$$
其中 $\mathcal{Y}$ 为由 mel 谱 组成的训练集。

然后怎么选呢？对于所有的 $M\in\mathcal{Y}$，找到最早的 $k^\prime$（或者说最小的），使得在 $[k^\prime,T]$ 之间的 step 中，有 95% 以上的 step 满足一下条件：
+ $\mathbf{BP}(\widetilde{M}_t,t)$ 和  $\mathbf{BP}({M}_t,t)$ 的 margin 都处于某个阈值之下
> 由于本质上 BP 为分类器，这个意思就是无法区分 95% 以上的大噪声 GT 样本和预测样本之间的差异。
> 其实这里的 $k$ 可以看成是模型的超参数，只不过这里用一个神经网络来学习罢了，因此实际上我们也可以用暴力搜索来查找 $k$。

满足此条件的 $k^\prime$ 即为 intersection 的位置。

整个训练和推理过程如下：
![](image/Pasted%20image%2020231002104947.png)
![](image/Pasted%20image%2020231002104957.png)


### 模型架构

Encoder 用于将 music score 编码到条件序列，包含：
+ lyrics encoder，将 phoneme ID 映射到 embedding（linguistic hidden sequence）
+ length regulator 根据 duration 信息将 linguistic hidden sequence 拓展到 mel 谱 长度（duration 采用 MFA 获得）
+ pitch encoder 将  pitch ID 映射到 pitch embedding sequence（pitch 通过 parsel-mouth 获得）
+ 把 linguistic sequence 和 pitch sequence 相加得到最终的 $E_m$

Step Embedding 采用 sinusoidal position embedding，然后通过两个 Linear 层得到 $E_t$。

辅助 Decoder 包含几层 feed-forward Transformer，和 FastSpeech 2 中的 mel 谱 decoder 一样。

Denoiser 采用 non-causal WaveNet 架构，如图：
![](image/Pasted%20image%2020231002105636.png)

Boundary Predictor 中的分类器为 ResNet，输入除了 mel 谱 还有 step embedding $E_t$。
