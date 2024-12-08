> ICLR 2022，Apple 公司，https://github.com/apple/ml-cvnets

1. 本文将 CNN 和 ViT 结合起来构建一个用于 mobile vision task 的轻量化、低延迟的网络
2. 提出 MobileViT，用于移动设备的轻量化的通用 ViT

## Introduction

1. 现有的将卷积+transformer 混合起来的方法仍然 heavy-weight，且对数据增强很敏感
2. 优化 FLOPs 并不足以实现 low latency
3. 本文观关注于设计轻量化、通用、低延迟的网络，将 CNN 和 ViT 结合起来，引入 MobileViT 模块在一个 tensor 中编码 local 和 global 信息
4. 在 5-6M 的参数下，可以在 ImageNet-1k 数据集中实现 top-1 的性能

## 相关工作（略）

## MobileViT：轻量化 Transformer
标准的 ViT 如下图 a：
![](image/Pasted%20image%2020241207103645.png)

输入 $\mathbf{X} \in \mathbb{R}^{H \times W \times C}$ reshap 为 patch 序列 $\mathbf{X}_f \in \mathbb{R}^{N \times PC}$，将其投影到固定的 $d$ 维空间 $\mathbf{X}_p \in \mathbb{R}^{N \times d}$，然后使用 $L$ 个 transformer block 学习 patch 之间的表征。ViT 的自注意力计算复杂度为 $O(N^2d)$。ViT 忽略了 CNN 中的 空间归纳偏差，因此需要更多参数来学习视觉表示。
本文引入 MobileViT，使用 transformer 作为卷积来学习全局表征，从而隐式地将卷积的特性（如空间偏差）融入网络。

### 模型架构
MobileViT block 如上图 b。对于输入 tensor $\mathbf{X} \in \mathbb{R}^{H \times W \times C}$，MobileViT 使用 $n \times n$ 标准卷积和一个 point-wise（或 $1 \times 1$）卷积得到 $\mathbf{X}_L \in \mathbb{R}^{H \times W \times d}$。$n \times n$ 卷积层编码 local 空间信息，而 point-wise 卷积将 tensor 投影到高维空间（$d$ 维，其中 $d > C$）。

MobileViT 旨在模拟 long-range non-local 依赖，同时有 $H \times W$ 的感受野。

为了使 MobileViT 可以学习带有空间归纳偏差的全局表征，将 $\mathbf{X}_L$ 展开为 $N$ 个不重叠的 patch $\mathbf{X}_U \in \mathbb{R}^{P \times N \times d}$。这里，$P = wh$，$N = \frac{HW}{P}$ 是 patch 的数量，$h \leq n$ 和 $w \leq n$ 分别是 patch 的高度和宽度。对于每个 $p \in \{1, \cdots, P\}$，通过 transformer 编码 inter-patch 关系得到 $\mathbf{X}_G \in \mathbb{R}^{P \times N \times d}$：
$$\mathbf{X}_G(p)=\text{Transformer}(\mathbf{X}_U(p)),1\leq p\leq P$$

由于 MobileViT 不会丢失 patch 的顺序，也不会丢失每个 patch 中像素的空间顺序。因此，可以将 $X_G \in \mathbb{R}^{P \times N \times d}$ 折叠为 $X_F \in \mathbb{R}^{H \times W \times d}$。然后使用 point-wise 卷积将 $X_F$ 投影到低维空间 $C$，然后与 $X$ 进行拼接。接着使用另一个 $n \times n$ 卷积层融合这些特征。
> 因为 $X_U(p)$ 使用卷积在 $n \times n$ 区域编码 local 信息，$X_G(p)$ 使用 transformer 从 $P$ 个 patch 中编码全局信息，因此 $X_G$ 中的每个像素都可以编码 $X$ 中所有像素的信息。因此，MobileViT 的有效感受野为 $H \times W$。

标准的卷积包含三个操作：（1）展开，（2）矩阵乘法（学习 local 表征），（3）折叠。MobileViT block 与卷积类似，使用相同的构建块。MobileViT block 将卷积中的 local 处理（矩阵乘法）替换为更深的 global 处理（一堆 transformer 层）。因此，MobileViT 具有卷积的特性（如空间偏差）。MobileViT block 可以看作是 transformers 作为卷积。
MobileViT 使用卷积和 transformer，使得 MobileViT block 具有卷积的特性，同时允许全局处理。从而可以设计浅而窄的 MobileViT 模型来实现轻量化。

MobileViT 和 ViTs 的 multi-headed self-attention 的计算复杂度分别为 $O(N^2Pd)$ 和 $O(N^2d)$。理论上，MobileViT 比 ViTs 低效。然而，在实践中，MobileViT 比 ViTs 更高效。MobileViT 的 FLOPs 比 DeIT 少 2 倍，且在 ImageNet-1K 数据集上比 DeIT 准确率高 1.8%。

### Multi-scale Sampler（略）


## 实验（略）
