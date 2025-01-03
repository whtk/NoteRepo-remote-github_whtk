> DiT 经典论文，ICCV 2023，UC Berkeley、New York University

1. 探索基于 transformer 架构的新的 diffusion 模型
2. 训练图像的 latent diffusion 模型，用 transformer 替换常用的 U-Net backbone
3. 通过 Gflops 衡量 forward pass 复杂度，分析 DiTs 的可扩展性，发现 Gflops 更高的 DiTs 有更低的 FID
4. 最大的 DiT-XL/2 模型在 class-conditional ImageNet 512×512 和 256×256 benchmarks 上超越 SOTA diffusion 模型

## Introduction

1. Transformer 之前都是用在自回归生成模型，但是没有被用于其他生成模型框架
2. 对于 diffusion models，普遍都是用 convolutional U-Net 作为 backbone
3. U-Net 最初用于 pixel-level autoregressive models 和 conditional GANs，后来被用于 diffusion models
4. 本文揭示 diffusion models 中的架构选择，表面：
    1. U-Net 的 inductive bias 对于 diffusion models 不重要
    2. 可以用 transformer 替换 U-Net
    3. diffusion models 可以从架构统一的趋势中受益
5. 本文关注基于 transformer 的 diffusion models，称为 DiTs
6. 研究 transformer 在 network complexity 和 sample quality 之间的 scaling 效果

## 相关工作（略）

## Diffusion Transformers

### 预备知识

DDPM 的原理：对于高斯 diffusion 模型，假设 forward noising 过程逐渐向真实数据 $x_0$ 添加噪声：$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1 - \bar{\alpha}_t)I)$，其中 $\bar{\alpha}_t$ 是超参数。采用 reparameterization trick，可以采样 $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon_t$，其中 $\epsilon_t \sim \mathcal{N}(0, I)$。

训练 diffusion models 学习 reverse process：$p_{\theta}(x_{t-1} | x_t) = \mathcal{N}(\mu_{\theta}(x_t), \Sigma_{\theta}(x_t))$，其中神经网络用于预测 $p_{\theta}$ 的统计量。reverse process 模型使用 对数似然的变分下界进行训练，即最小化 $L(\theta) = -p(x_0 | x_1) + \sum_t D_{KL}(q^*(x_{t-1} | x_t, x_0) || p_{\theta}(x_{t-1} | x_t))$。通过将 $\mu_{\theta}$ 重新参数化为 noise prediction network $\epsilon_{\theta}$，模型可以使用预测噪声 $\epsilon_{\theta}(x_t)$ 和真实采样的高斯噪声 $\epsilon_t$ 之间的均方误差进行训练。一旦 $p_{\theta}$ 训练完成，可以通过初始化 $x_{t_{\max}} \sim \mathcal{N}(0, I)$ 并通过 reparameterization trick 采样 $x_{t-1} \sim p_{\theta}(x_{t-1} | x_t)$。

Conditional diffusion 需要额外的信息作为输入，如类别标签 $c$。此时 reverse process 变为 $p_{\theta}(x_{t-1} | x_t, c)$，其中 $\epsilon_{\theta}$ 和 $\Sigma_{\theta}$ 与 $c$ 有关。classifier-free guidance 可以用于鼓励采样过程找到 $x$，使得 $\log p(c | x)$ 较高。通过 Bayes Rule，$\log p(c | x) \propto \log p(x | c) - \log p(x)$，因此 $\nabla_x \log p(c | x) \propto \nabla_x \log p(x | c) - \nabla_x \log p(x)$。通过将 diffusion models 的输出解释为 score function，可以引导 DDPM 采样过程，使其通过 $\hat{\epsilon}_{\theta}(x_t, c) = \epsilon_{\theta}(x_t, \emptyset) + s \cdot \nabla_x \log p(x | c) \propto \epsilon_{\theta}(x_t, \emptyset) + s \cdot (\epsilon_{\theta}(x_t, c) - \epsilon_{\theta}(x_t, \emptyset))$ 采样高概率 $p(x | c)$ 下的 $x$。其中 $s > 1$ 表示引导的尺度（$s = 1$ 为标准采样）。当 $c = \emptyset$ 时，通过在训练过程中随机丢弃 $c$ 并用学习到的“null” embedding $\emptyset$ 替换 $c$ 来评估 diffusion model。

在高分辨率像素空间直接训练 diffusion models 的计算代价过高。Latent diffusion models (LDMs) 通过两阶段方法解决这个问题：
1. 学习一个 autoencoder，将图像压缩为较小的空间表示
2. 训练基于 $z = E(x)$ 的 diffusion model，而不是基于图像 $x$ 的。然后可以通过从 diffusion model 中采样表示 $z$，然后使用学习到的 decoder 解码为图像 $x = D(z)$。

本文将 DiTs 应用于 latent space，使得图像生成 pipeline 成为一种混合方法；混合 convolutional VAEs 和基于 transformer 的 DDPMs。

### DiT Design Space

下面介绍 DiTs。DiT 基于 ViT 架构，输入为 sequence of patches。架构如图：
![](image/Pasted%20image%2020240922220526.png)

### Patchify

DiTs 的输入是 spatial representation $z$（对于 256 × 256 × 3 图像，$z$ 的形状为 32 × 32 × 4）。模型第一层是 patchify，将空间输入转换为 $T$ 个 token 的序列，每个 token 的维度为 $d$。然后加入位置编码。token 数量 $T$ 由 patch size $p$ 决定。减半 $p$ 会使 $T$ 增加 4 倍。本文的 DiT 中，$p = 2, 4, 8$。
> 也就是每 $p \times p$ 的 patch 被转换为一个 token。

如图：
![](image/Pasted%20image%2020240922222122.png)

### DiT 模块设计

在 patchify 之后，输入 token 通过一系列 transformer blocks。除了噪声输入，diffusion 还有处理额外的条件，如时间步 $t$、类别 $c$等。本文探索了四种 transformer blocks 变体，处理不同方式的条件输入。所有 blocks 的设计如上图。包含：
+ In-context conditioning：将 $t$ 和 $c$ 的 embedding 作为两个额外的 token 附加到输入序列中，与图像 token 一样（类似于 ViTs 中的 cls token）。在通过最后一个 block 之后，从序列中移除条件 token。
> 这种方法对模型的 Gflops 影响微乎其微。
+ Cross-attention block：将 $t$ 和 $c$ 的 embedding 连接为长度为 2 的序列，与图像 token 序列分开。修改 transformer block，添加一个额外的 multi-head cross-attention layer。
> cross-attention 对模型的 Gflops 影响最大，大约 15% 的额外开销。
> 个人感觉这种方法的效果最差。
+ Adaptive layer norm (adaLN) block：将 transformer block 中的标准 layer norm 替换为 adaptive layer norm。不直接学习维度缩放和偏移参数 $\gamma$ 和 $\beta$，而是从 $t$ 和 $c$ 的 embedding 中得到。
> adaLN 对模型的 Gflops 影响最小，最高效。
+ adaLN-Zero block：在 ResNets 中，将每个 residual block 初始化为 identity function。本文修改 adaLN DiT block，除计算 $\gamma$ 和 $\beta$，还计算 dimension-wise scaling 参数 $\alpha$，这个参数用于 DiT block 中的 residual connections 之前。初始化 MLP 为所有 $\alpha$ 输出零向量，将整个 DiT block 初始化为 identity function。
> 与 vanilla adaLN block 一样，adaLN-Zero 对模型的 Gflops 影响微乎其微。

### 模型大小

用 $N$ 个 DiT blocks，每个 block 的 hidden dimension 大小为 $d$。使用四种配置：DiT-S、DiT-B、DiT-L 和 DiT-XL。覆盖了从 0.3 到 118.6 Gflops 的模型大小和 flop。

### Transformer decoder

在最后一个 DiT block 之后，需要将 token 序列解码为 噪声 和 对角协方差。使用 linear decoder，将每个 token 解码为 $p \times p \times 2C$ 张量，其中 $C$ 是 DiT spatial 输入的通道数。最后，将解码的 token 重新排列为原始空间布局，得到预测的噪声和协方差。

## 实验细节

模型命名方式为 config 和 latent patch size $p$，如 DiT-XL/2 表示 XLarge config 和 $p = 2$。

训练：在 ImageNet 数据集上训练 256 × 256 和 512 × 512 图像分辨率的 class-conditional latent DiT 模型。使用 AdamW 训练，学习率为 $1 \times 10^{-4}$，无权重衰减，batch size 为 256。只使用水平翻转数据增强。训练过程中没有 warmup 和正则化。训练过程非常稳定，没有观察到常见的 transformer 训练中的 loss spikes。使用 EMA 权重，衰减为 0.9999。所有结果使用 EMA 模型。所有 DiT 模型大小和 patch 大小使用相同的训练超参数。训练超参数几乎完全来自 ADM。没有调整学习率、衰减/ warm-up schedules、Adam $\beta_1/\beta_2$ 或权重衰减。

使用 Stable Diffusion 中的预训练 VAE 模型。VAE encoder 下采样因子为 8，即输入 256 × 256 × 3 图像，$z = E(x)$ 的形状为 32 × 32 × 4。在所有实验中，diffusion models 在 Z-space 中操作。采样新的 latent 后，使用 VAE decoder 解码为像素。保留 ADM 的 diffusion 超参数：$t_{\max} = 1000$，线性方差调度从 $1 \times 10^{-4}$ 到 $2 \times 10^{-2}$。

评估指标：Fre ́chet Inception Distance (FID)、Inception Score、sFID 和 Precision/Recall。

所有模型使用 JAX 实现，在 TPU-v3 pods 上训练。DiT-XL/2 在 TPU v3-256 pod 上训练，大约 5.7 iterations/second。

## 实验

见论文。
