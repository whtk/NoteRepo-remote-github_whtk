> CVPR，2020

1. 好的图像转换系统需要满足：
   + 生成图片具有多样性
   + 在不同的域上有可伸缩性
2. 本文贡献：
   1. 提出 StarGAN v2
   2. 开源 AFHQ 数据集

## Introduction

1. StarGAN 可以使用一个生成器学习多域之间的转换，但是由于每个域还是会被分配一个确定的标签，导致 StarGAN 学习到的映射是域之间确定的映射，不能捕获数据分布的多模态特性
2. 由于 StarGAN 的输入标签是固定的，那么在给定源图像时，输入相同的域，每次转换都不可避免地产生相同的输出
3. 本文提出了StarGANv2，能够跨域产生多样性的图片。
   1. 基于StarGAN，把域标签（domain label）替换成特定的风格编码（specific style code，简称 sc，下同），从而能够表征一个domain 的多样的风格。
   2. 提出了两个模块：
	   1. 映射网络（mapping network）将随机高斯噪声转换成 sc
	   2. 风格编码器（style encoder）将输入一张参考图片，提取其 sc
	   3. 在多个 domain 的情况下，这两个模块可以产生多个分支，每个分支都对应一个 sc

## StarGANv2

### 框架

定义 $\mathcal{X}, \mathcal{Y}$ 分别表示为图集和域 domains 集，则给定某张图片 $\mathbf{x}$ 和某个域 $y$，目标是训练一个单一的生成器 $G$，给定 $y$，能够基于 $\mathbf{x}$ 产生多样性的图片。在每个域 domain 的 style space 中生成 domain-specific 的 sc，然后训练 $G$ 使其满足 sc。如图所示：
![](image/Pasted%20image%2020230916220505.png)

1. 生成器：将输入 $\mathbf{x}$ 转换为 $G(\mathbf{x}, s)$，$s$ 代表style code，它来自于映射网络 $F$ 或者风格编码器 $E$，同时使用 AdaIN将 $s$ 引入 $G$，不再需要输入 $y$（$s$ 对应于StarGAN论文中的 $c$）。
2. 映射网络：给定隐编码 $z$ 和 域 domain $y$，映射网络 $F$ 产生对应的sc：$s=F_y(z)$，下标 $y$ 代表对应于域 $y$ 的映射网络（这说明，每个域 domain 都对应一个独立的映射网络，更详细的说，$F$ 为一个有着多分支输出的 MLP，每个分支对应一个域）。
3. 风格编码器：风格编码器的输入为一张给定的图片和域，其输出为该图片对应的sc：${s}=E_{y}(\mathbf{x})$，其中 $E_{y}$ 表示对应于 domain $y$ 下的输出。最终使得生成器可以合成和输入 $\mathbf{x}$ 的 sc 相似的图片。
4. 判别器：$D$ 也是一个多任务的判别器，有很多个输出分支。每个分支 $D_y$ 进行二值分类，确定图像 $\mathbf{x}$ 是其域 $y$ 的real图像还是由 $G$ 生成的fake图像 $G(\mathbf{x},s)$。

### 损失函数

**对抗损失**：

$$
\begin{aligned}
\mathcal{L}_{a d v}=& \mathbb{E}_{\mathbf{x}, y}\left[\log D_{y}(\mathbf{x})\right]+\\
& \mathbb{E}_{\mathbf{x}, \widetilde{y}, \mathbf{z}}\left[\log \left(1-D_{\widetilde{y}}(G(\mathbf{x}, \widetilde{\mathbf{s}}))\right)\right]
\end{aligned}
$$

其中，$z \in \mathcal{Z}, \tilde{y}\in \mathcal{Y}$ 为随机采样，然后通过映射网络得到sc： $\tilde{s} = F_{\tilde{y}}(z)$，这一部分和基本的GAN网络一致。

**风格重构损失**：

$$
\mathcal{L}_{s t y}=\mathbb{E}_{\mathbf{x}, \widetilde{y}, \mathbf{z}}\left[\left\|\widetilde{\mathbf{s}}-E_{\widetilde{y}}(G(\mathbf{x}, \widetilde{\mathbf{s}}))\right\|_{1}\right]
$$

训练一个单一的encoder学习从图片到latent code的转换，简单来说，将图片和某个 style code $\tilde{s}$ 输入生成器，得到的图片再输入风格编码器，过程前后的 style code 应该保持不变。

**风格多样性损失**：

$$
\mathcal{L}_{d s}=\mathbb{E}_{\mathbf{x}, \widetilde{y}, \mathbf{z}_{1}, \mathbf{z}_{2}}\left[\left\|G\left(\mathbf{x}, \widetilde{\mathbf{s}}_{1}\right)-G\left(\mathbf{x}, \widetilde{\mathbf{s}}_{2}\right)\right\|_{1}\right]
$$

目标为最大化该损失函数。通过把两个不同的 latent code 输入到映射网络，$\widetilde{\mathbf{s}}_{i}=F_{\widetilde{y}}\left(\mathbf{z}_{i}\right) \text { for } i \in\{1,2\}$，最大化上式使得生成器能够尽可能的生成特定风格但是又具有多样性的图片。

**循环重构损失**（保留源特征）：

$$
\mathcal{L}_{c y c}=\mathbb{E}_{\mathbf{x}, y, \widetilde{y}, \mathbf{z}}\left[\|\mathbf{x}-G(G(\mathbf{x}, \widetilde{\mathbf{s}}), \hat{\mathbf{s}})\|_{1}\right]
$$

一张图片（假设其对应的sc为 $\hat{\mathbf{s}}=E_y(\mathbf{x})$ ），通过两次生成器，在第二个生成器的sc为 $\hat{\mathbf{s}}$ 时，得到的生成图应该和原始图片一样。

总目标函数为：

$$
\begin{aligned}
\min _{G, F, E} \max _{D} & \mathcal{L}_{a d v}+\lambda_{s t y} \mathcal{L}_{s t y} \\
&-\lambda_{d s} \mathcal{L}_{d s}+\lambda_{c y c} \mathcal{L}_{c y c}
\end{aligned}
$$

$\lambda$ 为不同损失的权重。

## 实验（略）