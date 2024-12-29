> ICCV 2023，Stanford

1. 提出 ControlNet，在大型预训练的 T2I diffusion 模型中添加条件控制
    1. ControlNet 锁住训练好的 diffusion 模型，利用训练好的模型作为 backbone，学习多种条件控制
    2. 使用 zero convolutions 逐渐增加参数，确保微调过程中不会受噪声的影响
2. 测试多种条件控制，如 edges, depth, segmentation, human pose 等、单个或多个条件、有无 prompt 
3. ControlNet 的训练在小（<50K）和大（>1M）数据集上都很稳健

## Introduction

1. T2I 中，通过文本 prompt 难以表达复杂的布局、姿势、形状和形式
2. 可以通过提供额外的图像（如边缘图、人体姿势骨架、分割图、深度、法线等）来实现更精细的空间控制
3. 但大型 T2I diffusion 模型学习条件控制很难：
    1. 特定条件数据很少
    2. 直接微调或继续训练可能导致过拟合和灾难性遗忘
4. 提出 ControlNet，实现预训练 T2I diffusion 大模型的条件控制
    1. 通过 lock 模型参数但是 训练 encoder 的 copy，保留大模型的质量和能力
    2. trainable copy 和原始 lock 住的模型通过 zero convolution 连接，参数初始化为 0
    3. 可以确保训练过程中不会对大模型的深层特征添加**有害噪声**
> 这里的有害噪声应该是指特定条件下的训练数据，而不是传统意义上的噪声
5. 实验表明 ControlNet 可以通过 Canny 边缘、Hough 线、用户涂鸦、人体关键点、分割图、形状法线、深度等多种条件控制 SD

## 相关工作（略）

## 方法

### ControlNet

结构如图：
![](image/Pasted%20image%2020241228105245.png)

ControlNet 将额外的条件引入到 network block 中。假设 $\mathcal{F}(\cdot;\Theta)$ 是一个训练好的 block，参数为 $\Theta$，将输入特征图 $\boldsymbol{x}$ 转换为另一个特征图 $\boldsymbol{y}$：
$$\boldsymbol{y}=\mathcal{F}(\boldsymbol{x};\Theta).$$
其中的 $\boldsymbol{x}$ 和 $\boldsymbol{y}$ 通常是 2D 特征图，即 $\boldsymbol{x}\in\mathbb{R}^{h\times w\times c}$，其中 $h,w,c$ 分别是特征图的高、宽和通道数。

然后 lock（冻结）原始 block 的参数 $\Theta$，同时对 block 克隆一个 trainable copy，参数为 $\Theta_c$。trainable copy 输入条件向量 $\boldsymbol{c}$。lock 的参数保留了已经训练好的模型，而 trainable copy 重用大规模预训练模型。

trainable copy 通过 zero convolution layer 连接到 lock 的模型。这里的 zero convolution 是一个 1×1 卷积层，权重和偏置都初始化为 0。ControlNet 使用两个 zero convolutions，参数分别为 $\Theta_{z1}$ 和 $\Theta_{z2}$。ControlNet 计算如下：
$$\boldsymbol{y}_{\mathfrak{c}}=\mathcal{F}(\boldsymbol{x};\Theta)+\mathcal{Z}(\mathcal{F}(\boldsymbol{x}+\mathcal{Z}(\boldsymbol{c};\Theta_{\mathrm{z}1});\Theta_{\mathrm{c}});\Theta_{\mathrm{z}2}),$$
其中 $\boldsymbol{y}_c$ 是 ControlNet block 的输出。在第一次训练中，由于 zero convolution 的权重和偏置都初始化为 0，所以公式中的两个 $Z(\cdot;\cdot)$ 都为 0，因此：
$$\boldsymbol{y}_c=\boldsymbol{y}.$$

在训练开始时，有害噪声无法影响 trainable copy 中的隐藏状态。此外，由于 $Z(\boldsymbol{c};\Theta_{z1})=0$，trainable copy 也接收输入图像 $\boldsymbol{x}$，因此 trainable copy 具有完全的功能，保留了大型预训练模型的能力，可以作为进一步学习的强大 backbone。


### 用于 T2I Diffusion 的 ControlNet

SD 由 encoder、middle block 和 decoder 组成，本质上是一个 U-Net。

ControlNet 用于每个 encoder 层，创建 12 个 encoding blocks 和 1 个 middle block 的 trainable copy。12 个 encoding blocks 在 4 个分辨率（64×64,32×32,16×16,8×8）上，每个分辨率重复 3 次。输出加到 12 个 skip-connections 和 1 个 middle block 上，如图：
![](image/Pasted%20image%2020241228110800.png)

> 提问：
> 1. 为什么在 decoder 上不使用 ControlNet：decoder 可以看成是根据特征图生成图像，由于要实现条件控制，所以需要修改输入的特征图，这里的 decoder 当成一个普通的任意的图像 decoder 理解即可
> 2. 为什么 skip connection 连接到的是 decoder（按照前面的说法，不应该是连在同一个模块的输出吗）：猜测应该是把这里的输出当成一种 condition 来控制 decoder 的合成
> 所以本质上就是从 encoder 中拿了一个 $\boldsymbol{x}$，输入一个 $\boldsymbol{c}$，得到某种特征，通过这个特征控制 decoder 的合成，背后的思路就是简单的条件控制合成，但是做法很巧妙。

由于 lock 的参数被冻结，finetuning 时原始 encoder 不需要计算梯度。在单个 NVIDIA A100 PCIE 40GB 上测试，使用 ControlNet 优化 Stable Diffusion 只需要比不使用 ControlNet 多约 23% 的 GPU 内存和 34% 的时间。

为了将 ControlNet 添加到 SD，先将每个输入条件图像（如边缘、姿势、深度等）从 512×512 的输入大小转换为与 SD 匹配的 64×64 特征向量。具体来说，使用一个小网络 $\mathcal{E}(\cdot)$，包含四个卷积层，核大小为 4×4，步长为 2×2（由 ReLU 激活，分别使用 16、32、64、128 个通道，用高斯权重初始化，并与整个模型一起训练）将图像空间条件 $\boldsymbol{c}_i$ 编码为特征空间条件向量 $\boldsymbol{c}_f$：
$$\boldsymbol{c}_\mathrm{f}=\mathcal{E}(\boldsymbol{c}_\mathrm{i}).$$
特征向量 $\boldsymbol{c}_f$ 传入 ControlNet。

### 训练

给定输入图像 $\boldsymbol{z}_0$，图像扩散算法逐渐向图像添加噪声，生成带噪图像 $\boldsymbol{z}_t$，其中 $t$ 表示添加噪声的次数。给定一组条件，包括时间步 $t$、文本提示 $\boldsymbol{c}_t$，以及任务特定条件 $\boldsymbol{c}_f$，diffusion 模型学习 $\epsilon_\theta$ 预测添加到带噪图像 $\boldsymbol{z}_t$ 的噪声：
$$\mathcal{L}=\mathbb{E}_{\boldsymbol{z}_0,\boldsymbol{t},\boldsymbol{c}_t,\boldsymbol{c}_\mathrm{f},\epsilon\thicksim\mathcal{N}(0,1)}{\left[\|\epsilon-\epsilon_\theta(\boldsymbol{z}_t,\boldsymbol{t},\boldsymbol{c}_t,\boldsymbol{c}_\mathrm{f}))\|_2^2\right]},$$

这里的 $\mathcal{L}$ 是整个 diffusion 模型的学习目标，用于微调带 ControlNet 的 diffusion 模型。

训练时随机替换 50% 的文本提示 $\boldsymbol{c}_t$ 为空字符串。
> 增加 ControlNet 直接识别输入条件图像（如边缘、姿势、深度等）中的语义的能力。

由于 zero convolutions 不会向网络添加噪声，模型应该始终能够预测高质量的图像。作者观察到，模型**不会逐渐学习控制条件**，而是**突然可以合成遵循输入条件的图像**（通常在不到 10K 优化步骤内）。

### 推理

可以通过几种方式进一步控制 ControlNet 的条件如何影响 diffusion 过程：

+ Classifier-free guidance resolution weighting：SD 使用CFG 生成高质量图像 $\epsilon_{\mathrm{prd}}=\epsilon_{\mathrm{uc}}+\beta_{\mathrm{cfg}}(\epsilon_{\mathrm{c}}-\epsilon_{\mathrm{uc}})$，其中 $\epsilon_{\mathrm{prd}}$、$\epsilon_{\mathrm{uc}}$、$\epsilon_{\mathrm{c}}$、$\beta_{\mathrm{cfg}}$ 分别是模型的最终输出、无条件输出、有条件输出、权重。当通过 ControlNet 添加条件图像时，可以同时添加到 $\epsilon_{\mathrm{uc}}$ 和 $\epsilon_{\mathrm{c}}$，或者只添加到 $\epsilon_{\mathrm{c}}$。
> 在某些情况下，如没有 prompt 时，同时添加到 $\epsilon_{\mathrm{uc}}$ 和 $\epsilon_{\mathrm{c}}$ 会完全移除 CFG guidance；只加到 $\epsilon_{\mathrm{c}}$ 又会使 guidance 非常强。作者的解决方案是首先将条件图像添加到 $\epsilon_{\mathrm{c}}$，然后根据每个 block 的分辨率乘以一个权重 $w_i=64/h_i$，其中 $h_i$ 是第 $i$ 个 block 的大小，如 $h_1=8,h_2=16,\ldots,h_{13}=64$。此方法称为 CFG Resolution Weighting。
+ Composing multiple ControlNets：为了将多个条件图像（如 Canny 边缘和姿势）应用于单个 Stable Diffusion 实例，可以直接将相应 ControlNet 的输出添加到 Stable Diffusion 模型中。不需要额外的加权或线性插值。

## 实验（略）
