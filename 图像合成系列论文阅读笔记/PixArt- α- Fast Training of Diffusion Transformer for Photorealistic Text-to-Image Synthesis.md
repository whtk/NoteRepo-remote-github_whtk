> preprint 2023，华为诺亚方舟实验室

1. 提出 PIXART-α：
    1. 基于 Transformer 的 T2I diffusion 模型，图像质量和 SOTA 相当
    2. 支持 1024 ×1024 分辨率合成，训练成本低
2. 提出了三个核心设计：
    1. 训练策略分解：三个训练步骤，分别优化 像素依赖、文本-图像对齐 和 图像质量
    2. 高效 Transformer：将 cross-attention 模块整合到 DiT 来引入文本条件，简化 class-condition 
    3. 高信息量数据：强调文本-图像对中概念密度的重要性，利用大型 Vision-Language 模型标记伪标题，学习文本-图像对齐
3. PIXART-α 训练时间为 Stable Diffusion v1.5 的 12%

## Introduction
1. SDv1.5 训练时间为 6250 GPU 天，RAPHAEL 训练时间为 60K GPU 天，训练成本分别为 320K 和 3,080K 美元
1. 提出 PIXART-α，降低训练成本，保持与 SOTA 相当的图像质量。PIXART-α 与现有 T2I 的训练成本如下：
![](image/Pasted%20image%2020241224101633.png)

## 方法

### 动机

T2I 训练缓慢的原因：
+ 训练的 pipeline，T2I 生成任务可以分解为三部分
	+ 捕获像素依赖：理解图像内部像素级依赖关系，捕获分布
	+ 文本-图像对齐：准确的对齐，生成与文本描述准确匹配的图像
	+ 高质量
+ 数据集的打标质量：
	+ 文本-图像对齐不准确
	+ 描述不足
	+ 词汇使用不多样
	+ 包含低质量数据

### 训练策略分解

模型的生成能力可以通过将训练分为三个阶段来逐步优化：
+ 阶段一，Pixel dependency 学习：训练一个 class-conditional 模型，易于训练，成本低；同时发现合适的初始化可以显著提高训练效率，于是采用 ImageNet 作为预训练模型
+ 阶段二，文本-图像对齐学习：从预训练的 class-guided 图像生成模型过渡到 T2I，难点在于文本和图像之间的对齐。于是构建了一个包含了很多 high concept density 的文本-图像对的数据集
+ 阶段三，高分辨率和高质量图像生成：使用高质量数据微调

### 高效的 T2I Transformer

模型结构如下：
![](image/Pasted%20image%2020241224153432.png)

PIXART-α 采用 [Scalable Diffusion Models with Transformers 笔记](Scalable%20Diffusion%20Models%20with%20Transformers%20笔记.md) 作为基础架构，但是修改了 Transformer blocks 来处理 T2I 任务：
+ Cross-Attention 层：在 DiT block 中加入 multi-head cross-attention（在self-attention 和 feed-forward 之间），使模型可以与文本嵌入交互
+ AdaLN-single：DiT 中 adaLN 的线性投影占了很大的参数，提出 adaLN-single，只在第一个 block 中使用 time embedding 作为输入。具体来说，对于第 $i$ 个 block，$S^{i} = [β^{i}_{1},β^{i}_{2},γ^{i}_{1},γ^{i}_{2},α^{i}_{1},α^{i}_{2}]$ 是 adaLN 中所有的 scale 和 shift 参数，通过 MLP $S^{i} = f^{i}(c+t)$ 得到，其中 $c$ 和 $t$ 分别表示 class condition 和 time embedding，而在 adaLN-single 中，只在第一个 block 中计算一组全局的 shift 和 scale，然后通过 $S^{i} = g(S,E^{i})$ 得到 $S^{i}$，其中 $g$ 是求和函数，$E^{i}$ 是一个与 $S$ 形状相同的可训练的 embedding，用于调整不同 block 中的 scale 和 shift 参数
+ Re-parameterization：为了利用预训练权重，所有 $E^{i}$ 都初始化为使得 $S^{i}$ 与 DiT 中的相同的值，这种设计有效地用全局 MLP 和 layer-specific 可训练的 embeddings 替换了 layer-specific MLPs，同时保持了与预训练权重的兼容性

实验表明，引入全局 MLP 和 layer-wise embeddings 以及 cross-attention 层，保持了模型的生成能力，同时减小模型的大小。

### 数据集构建

图像-文本对自动标注：采用 LLaVA 模型，使用 prompt “Describe this image and its style in a very detailed manner” 来生成高信息密度的标题。

然而，LAION 数据集主要由购物网站的简单产品预览组成，不适合训练 T2I 生成，因此使用 SAM 数据集，通过 LLaVA 得到了高质量的文本-图像对。

第三阶段，使用 JourneyDB 和 10M 内部数据集来提高生成图像的美学质量。

通过 LLaVA 标注，有效的名词比例从 8.5% 增加到 13.3%，平均每张图像的名词数量从 6.4 增加到 21，SAM-LLaVA 的总名词数量为 328M，每张图像 30 个名词，表明 SAM 包含更丰富的目标和更高的信息密度。

## 实验（略）

