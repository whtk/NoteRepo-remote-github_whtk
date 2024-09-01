> ICLR 2024，CUHK，上海人工智能实验室，斯坦福

1. 将 motion dynamics 添加到现有 T2I 使其能够生成动画仍是一个的挑战
2. 提出 AnimateDiff，可以在不需要特定调整的情况下为 T2I 模型添加动画
    1. 核心是一个 plug-and-play 的 motion module，可以训练一次并无缝集成到任何 T2I 中
    2. motion module 可以从视频中学习 motion priors
    3. 训练完成后，motion module 可以插入到 T2I 模型中
3. 提出 MotionLoRA，在低的训练和数据 下使预训练的 motion module 来适应到新的 motion pattern

## Introduction

1. DreamBooth 和 LoRA 等方法可以在小数据集上对 T2I 模型进行定制微调，但是这些模型只能生成静态图像
2. 本文旨在将现有的 T2I 模型直接转换为动画生成器，而无需特定的微调
3. 提出 AnimateDiff，可以训练一个 plug-and-play 的 motion module，从视频数据集中学习 motion priors，训练分为三个阶段：
    1. 在 base T2I 上微调 domain adapter，使其与目标视频数据集的分布对齐
    2. 将 base T2I 和 domain adapter 与新初始化的 motion module 一起 inflate，然后在视频上优化这个motion 模块
    3. 定的 使用 MotionLoRA 适应预训练的 motion module 到特定的 motion patterns
4. 总结：
    1. 提出 AnimateDiff，可以使任何 personalized T2Is 具有动画生成能力，无需特定的微调
    2. 证明 Transformer 架构足以建模 motion priors
    3. 提出 MotionLoRA，一个轻量级的微调技术，可以适应新的 motion patterns

## 相关工作（略）

## 预备知识

### Stable Diffusion

选择 SD 作为 base T2I 模型。SD 在预训练的 autoencoder $\mathcal{E}(\cdot)$ 和 $\mathcal{D}(\cdot)$ 的 latent space 中进行 diffusion 过程。训练时，编码后的图像 $z_0=\mathcal{E}(x_0)$ 通过 forward diffusion 得到 $z_t$：
$$z_t=\sqrt{\bar{\alpha_t}}z_0+\sqrt{1-\bar{\alpha_t}}\epsilon,\:\epsilon\thicksim\mathcal{N}(0,I),$$
其中 $t = 1,\dots, T$，$\bar{\alpha_t}$ 决定了第 $t$ 步的噪声强度。去噪网络 $\epsilon_\theta(\cdot)$ 学习预测添加的噪声，损失为 MSE：
$$\mathcal{L}=\mathbb{E}_{\mathcal{E}(x_0),y,\epsilon\sim\mathcal{N}(0,I),t}\left[\|\epsilon-\epsilon_\theta(z_t,t,\tau_\theta(y))\|_2^2\right],$$
其中 $y$ 为 $x_0$ 的文本 prompt；$\tau_\theta(\cdot)$ 是将提示映射到向量序列的 text encoder。$\epsilon_\theta(\cdot)$ 结构为 UNet，包含四个不同分辨率的 down/upsample blocks，每个 block 包含 ResNet、spatial self-attention layers 和 cross-attention layers。

### Low-rank adaptation (LoRA)

LoRA 不重训模型的所有参数，而是添加 rank-decomposition matrices 并仅优化这些引入的权重。具体来说，rank-decomposition matrices 作为预训练模型权重 $W\in\mathbb{R}^{m\times n}$ 的残差。LoRA 中的新模型权重为：
$$\mathcal{W}^{\prime}=\mathcal{W}+\Delta\mathcal{W}=\mathcal{W}+AB^T,$$
其中 $A\in\mathbb{R}^{m\times r}$，$B\in\mathbb{R}^{n\times r}$ 是 rank-decomposition matrices，$r$ 是超参数，称为 LoRA 层的 rank。实际使用中，LoRA 只用于 attention layers，减少模型微调的成本和存储。

## ANIMATEDIFF
模型的核心在于，从视频数据中学习可迁移的 motion pairs，从而可以在不需要特点 fine tune 的情况下用于 personalized T2I。

如图：
![](image/Pasted%20image%2020240901160105.png)
在推理时，motion module（蓝色部分） 和 MotionLoRA（绿色，可选） 可以直接插入到 personalized T2I 中，构成 animation generator，通过去噪过程生成动画。

AnimateDiff 包含三个三个部分：
+ domain adapter：仅用于训练阶段，用于缓解原始的 T2I 模型和视频分布之间的 gap
+ motion module；学习 motion prior
+ MotionLoRA：可选，使预训练的 motion modules 可以 adapt 到新的 motion pattern

![](image/Pasted%20image%2020240901161150.png)

### Domain Adapter

图形和视频存在很大的 gap，如果把视频直接当成图像，每个视频的 frame 都包含 motion blur, compression artifacts 和 水印，从而使得两个模态之前存在 gap，影响性能。

于是提出在一个独立的网络中拟合 domain information，称为 domain adapter（然后在推理时丢弃）。采用 LoRA 实现，将其插入到 T2I 模型的 self-/cross-attention layer 中，如上图，以 query (Q) projection 为例，投影之后的 feature $z$ 为：
$$Q=\mathcal{W}^Qz+\text{AdapterLayer}(z)=\mathcal{W}^Qz+\alpha\cdot AB^Tz,$$
其中 $\alpha=1$，在推理时可以调整为其他的值（为 $0$ 则不用 domain adapter），从视频中随机采样静态的 frame 来训练。

### Motion Module

为了在预训练的 T2I 模型中在时间维度建模 motion dynamics，需要：
+ 将 2D diffusion 模型拓展到 3D
+ 设计一个 sub-module 实现时间轴上的信息交换

一种方法是，让这些 image layer 独立地处理视频的 frame，采用和现有工作相似的方法，模型输入为 5D 的 tensor $x\in\mathbb{R}^{b\times c\times f\times h\times w}$，其中 $b$ 为 batch，$f$ 为 frame-time，然后忽略 $f$（将其 reshape 到 $b$），也就是说，对于每个 frame 都是独立处理的，通过 image layer 之后再 reshape 回 5D tensor。
> 对于 文本 中的 attention，输入维度为 b N d，其中 b 为 batch，N 为 sequence length，d 为 hidden size。对于视频：
> + 如果是之前常用的 spatial attention，输入会被 reshape 为 (b*f) (h*w) c，分别对应文本中的 batch、sequence length 和 hidden size
> + 如果是 temporal attention，输入会被 reshape 为 (b*h*w) f c，分别对应文本中的 batch、sequence length 和 hidden size

采用 Transformer 架构作为 motion module，对其进行改进得到 temporal Transformer：
+ 包含几个沿着时间轴的 self-attention block
+ 使用 sinusoidal position encoding 来编码每个 frame 的位置
+ 其输入是经过 reshape 的 feature map（空间维度合并到 batch）。将 feature map 沿着时间轴划分，可以看作长度为 $f$ 的向量序列 $\{z_1,\dots,z_f;z_i\in\mathbb{R}^{(b\times h\times w)\times c}\}$，然后经过投影和几个 self-attention block：
$$z_{out}=\text{Attention}(Q,K,V)=\text{Softmax}(QK^T/\sqrt{c})\cdot V,$$
其中 $Q=W^Qz$，$K=W^Kz$，$V=W^Vz$。sinusoidal position encoding 是必要的，否则模块不知道 frame 的顺序。
> 为了避免额外模块引入的负面影响，作者将 temporal Transformer 的输出投影层初始化为 0，并添加残差连接，使得 motion module 在训练开始时是一个 identity mapping。

### MotionLoRA

MotionLoRA 是一个用于实现 motion personalization 的高效 fine-tuning 方法，即，在 Motion module 的 self-attention layers 中添加 LoRA 层，然后在新的 motion patterns 的 reference videos 上训练这些 LoRA 层。

通过 rule-based data augmentation 获取了几种 shot types 的 reference videos，例如，为了获取 zooming 效果的视频，逐渐减小（zoom-in）或增大（zoom-out）视频帧的裁剪区域。
> 结果：MotionLoRA 可以在 20 ∼ 50 个 reference videos，2000 个 epoch（约 1 ∼ 2 小时）以及约 30M 存储空间的情况下取得良好的结果。

## 实现（略）

## 实验（略）
