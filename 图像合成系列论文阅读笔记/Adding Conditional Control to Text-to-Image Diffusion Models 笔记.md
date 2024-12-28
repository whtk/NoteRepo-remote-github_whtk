> ICCV 2023，Stanford
<!-- We present ControlNet, a neural network architecture to
add spatial conditioning controls to large, pretrained text-
to-image diffusion models. ControlNet locks the production-
ready large diffusion models, and reuses their deep and ro-
bust encoding layers pretrained with billions of images as a
strong backbone to learn a diverse set of conditional controls.
The neural architecture is connected with “zero convolutions”
(zero-initialized convolution layers) that progressively grow
the parameters from zero and ensure that no harmful noise
could affect the finetuning. We test various conditioning con-
trols, e.g., edges, depth, segmentation, human pose, etc., with
Stable Diffusion, using single or multiple conditions, with
or without prompts. We show that the training of Control-
Nets is robust with small (<50k) and large (>1m) datasets.
Extensive results show that ControlNet may facilitate wider
applications to control image diffusion models. -->
1. 提出 ControlNet，在大型预训练的 T2I diffusion 模型中添加条件控制
    1. ControlNet 锁住训练好的 diffusion 模型，利用训练好的模型作为 backbone，学习多种条件控制
    2. 使用 zero convolutions 逐渐增加参数，确保微调过程中不会受噪声的影响
2. 测试多种条件控制，如 edges, depth, segmentation, human pose 等、单个或多个条件、有无 prompt 
3. ControlNet 的训练在小（<50k）和大（>1m）数据集上都很稳健

## Introduction
<!-- Many of us have experienced flashes of visual inspiration
that we wish to capture in a unique image. With the advent
of text-to-image diffusion models [54, 62, 72], we can now
create visually stunning images by typing in a text prompt.
Yet, text-to-image models are limited in the control they
provide over the spatial composition of the image; precisely
expressing complex layouts, poses, shapes and forms can be
difficult via text prompts alone. Generating an image that
accurately matches our mental imagery often requires nu-
merous trial-and-error cycles of editing a prompt, inspecting
the resulting images and then re-editing the prompt. -->
1. T2I 中，通过文本 prompt 难以表达复杂的布局、姿势、形状和形式
<!-- Can we enable finer grained spatial control by letting
users provide additional images that directly specify their
desired image composition? In computer vision and machine
learning, these additional images (e.g., edge maps, human
pose skeletons, segmentation maps, depth, normals, etc.)
are often treated as conditioning on the image generation
process. Image-to-image translation models [34, 98] learn
the mapping from conditioning images to target images. The
research community has also taken steps to control text-
to-image models with spatial masks [6, 20], image editing
instructions [10], personalization via finetuning [21, 75], etc.
While a few problems (e.g., generating image variations,
inpainting) can be resolved with training-free techniques
like constraining the denoising diffusion process or edit-
ing attention layer activations, a wider variety of problems
like depth-to-image, pose-to-image, etc., require end-to-end
learning and data-driven solutions. -->
2. 可以通过提供额外的图像（如边缘图、人体姿势骨架、分割图、深度、法线等）来实现更精细的空间控制
<!-- Learning conditional controls for large text-to-image dif-
fusion models in an end-to-end way is challenging. The
amount of training data for a specific condition may be sig-
nificantly smaller than the data available for general text-to-
image training. For instance, the largest datasets for various
specific problems (e.g., object shape/normal, human pose
extraction, etc.) are usually about 100K in size, which is
50,000 times smaller than the LAION-5B [79] dataset that
was used to train Stable Diffusion [82]. The direct finetun-
ing or continued training of a large pretrained model with
limited data may cause overfitting and catastrophic forget-
ting [31, 75]. Researchers have shown that such forgetting
can be alleviated by restricting the number or rank of train-
able parameters [14, 25, 31, 92]. For our problem, designing
deeper or more customized neural architectures might be
necessary for handling in-the-wild conditioning images with
complex shapes and diverse high-level semantics. -->
3. 但大型 T2I diffusion 模型学习条件控制很难：
    1. 特定条件数据很少
    2. 直接微调或继续训练可能导致过拟合和灾难性遗忘
<!-- This paper presents ControlNet, an end-to-end neural
network architecture that learns conditional controls for large
pretrained text-to-image diffusion models (Stable Diffusion
in our implementation). ControlNet preserves the quality
and capabilities of the large model by locking its parameters,
and also making a trainable copy of its encoding layers.
This architecture treats the large pretrained model as a strong
backbone for learning diverse conditional controls. The
trainable copy and the original, locked model are connected
with zero convolution layers, with weights initialized to zeros
so that they progressively grow during the training. This
architecture ensures that harmful noise is not added to the
deep features of the large diffusion model at the beginning
of training, and protects the large-scale pretrained backbone
in the trainable copy from being damaged by such noise. -->
4. 提出 ControlNet，实现预训练 T2I diffusion 大模型的条件控制
    1. 通过 lock 模型参数但是 训练 encoder 的 copy，保留大模型的质量和能力
    2. trainable copy 和原始 lock 住的模型通过 zero convolution 连接，参数初始化为 0
    3. 可以确保训练过程中不会对大模型的深层特征添加**有害噪声**
> 这里的有害噪声应该是指特定条件下的训练数据，而不是传统意义上的噪声
<!-- Our experiments show that ControlNet can control Sta-
ble Diffusion with various conditioning inputs, including
Canny edges, Hough lines, user scribbles, human key points,
segmentation maps, shape normals, depths, etc. (Figure 1).
We test our approach using a single conditioning image,
with or without text prompts, and we demonstrate how our
approach supports the composition of multiple conditions.
Additionally, we report that the training of ControlNet is
robust and scalable on datasets of different sizes, and that for
some tasks like depth-to-image conditioning, training Con-
trolNets on a single NVIDIA RTX 3090Ti GPU can achieve
results competitive with industrial models trained on large
computation clusters. Finally, we conduct ablative studies to
investigate the contribution of each component of our model,
and compare our models to several strong conditional image
generation baselines with user studies -->
5. 实验表明 ControlNet 可以通过 Canny 边缘、Hough 线、用户涂鸦、人体关键点、分割图、形状法线、深度等多种条件控制 SD

## 相关工作（略）

## 方法
<!-- ControlNet is a neural network architecture that can en-
hance large pretrained text-to-image diffusion models with
spatially localized, task-specific image conditions. We first
introduce the basic structure of a ControlNet in Section 3.1
and then describe how we apply a ControlNet to the image
diffusion model Stable Diffusion [72] in Section 3.2. We
elaborate on our training in Section 3.3 and detail several
extra considerations during inference such as composing
multiple ControlNets in Section 3.4. -->

### ControlNet

结构如图：
![](image/Pasted%20image%2020241228105245.png)
<!-- ControlNet injects additional conditions into the blocks of
a neural network (Figure 2). Herein, we use the term network
block to refer to a set of neural layers that are commonly
put together to form a single unit of a neural network, e.g.,
resnet block, conv-bn-relu block, multi-head attention block,
transformer block, etc. Suppose F(·; Θ) is such a trained
neural block, with parameters Θ, that transforms an input
feature map x, into another feature map y as -->
ControlNet 将额外的条件引入到 network block 中。假设 $\mathcal{F}(\cdot;\Theta)$ 是一个训练好的 block，参数为 $\Theta$，将输入特征图 $\boldsymbol{x}$ 转换为另一个特征图 $\boldsymbol{y}$：
$$\boldsymbol{y}=\mathcal{F}(\boldsymbol{x};\Theta).$$
<!-- In our setting, x and y are usually 2D feature maps, i.e., x ∈
Rh×w×c with {h,w,c}as the height, width, and number of
channels in the map, respectively (Figure 2a). -->
$\boldsymbol{x}$ 和 $\boldsymbol{y}$ 通常是 2D 特征图，即 $\boldsymbol{x}\in\mathbb{R}^{h\times w\times c}$，其中 $h,w,c$ 分别是特征图的高、宽和通道数。
<!-- To add a ControlNet to such a pre-trained neural block,
we lock (freeze) the parameters Θ of the original block and
simultaneously clone the block to a trainable copy with
parameters Θc (Figure 2b). The trainable copy takes an
external conditioning vector c as input. When this structure
is applied to large models like Stable Diffusion, the locked
parameters preserve the production-ready model trained with
billions of images, while the trainable copy reuses such large-
scale pretrained model to establish a deep, robust, and strong
backbone for handling diverse input conditions. -->
然后 lock（冻结）原始 block 的参数 $\Theta$，同时将 block 克隆一个 trainable copy，参数为 $\Theta_c$。trainable copy 输入条件向量 $\boldsymbol{c}$。lock 的参数保留了已经训练好的模型，而 trainable copy 重用大规模预训练模型。
<!-- The trainable copy is connected to the locked model with
zero convolution layers, denoted Z(·;·). Specifically, Z(·;·)
is a 1 ×1 convolution layer with both weight and bias ini-
tialized to zeros. To build up a ControlNet, we use two
instances of zero convolutions with parameters Θz1 and Θz2
respectively. The complete ControlNet then computes -->
trainable copy 通过 zero convolution layer 连接到 lock 的模型。这里的 zero convolution 是一个 1×1 卷积层，权重和偏置都初始化为 0。ControlNet 使用两个 zero convolutions，参数分别为 $\Theta_{z1}$ 和 $\Theta_{z2}$。ControlNet 计算如下：
$$\boldsymbol{y}_{\mathfrak{c}}=\mathcal{F}(\boldsymbol{x};\Theta)+\mathcal{Z}(\mathcal{F}(\boldsymbol{x}+\mathcal{Z}(\boldsymbol{c};\Theta_{\mathrm{z}1});\Theta_{\mathrm{c}});\Theta_{\mathrm{z}2}),$$
<!-- where yc is the output of the ControlNet block. In the first
training step, since both the weight and bias parameters of
a zero convolution layer are initialized to zero, both of the
Z(·;·) terms in Equation (2) evaluate to zero, and yc = y.-->
其中 $\boldsymbol{y}_c$ 是 ControlNet block 的输出。在第一次训练中，由于 zero convolution 的权重和偏置都初始化为 0，所以公式中的两个 $Z(\cdot;\cdot)$ 都为 0，因此：
$$\boldsymbol{y}_c=\boldsymbol{y}.$$
<!-- In this way, harmful noise cannot influence the hidden states
of the neural network layers in the trainable copy when the
training starts. Moreover, since Z(c; Θz1) = 0 and the train-
able copy also receives the input image x, the trainable copy
is fully functional and retains the capabilities of the large,
pretrained model allowing it to serve as a strong backbone
for further learning. Zero convolutions protect this back-
bone by eliminating random noise as gradients in the initial
training steps. We detail the gradient calculation for zero
convolutions in supplementary materials. -->
在训练开始时，有害噪声无法影响 trainable copy 中的隐藏状态。此外，由于 $Z(\boldsymbol{c};\Theta_{z1})=0$，trainable copy 也接收输入图像 $\boldsymbol{x}$，因此 trainable copy 具有完全的功能，保留了大型预训练模型的能力，可以作为进一步学习的强大 backbone。

<!-- ControlNet for Text-to-Image Diffusion -->
### 用于 T2I Diffusion 的 ControlNet
<!-- We use Stable Diffusion [72] as an example to show how
ControlNet can add conditional control to a large pretrained
diffusion model. Stable Diffusion is essentially a U-Net [73]
with an encoder, a middle block, and a skip-connected de-
coder. Both the encoder and decoder contain 12 blocks,
and the full model contains 25 blocks, including the middle
block. Of the 25 blocks, 8 blocks are down-sampling or
up-sampling convolution layers, while the other 17 blocks
are main blocks that each contain 4 resnet layers and 2 Vi-
sion Transformers (ViTs). Each ViT contains several cross-
attention and self-attention mechanisms. For example, in
Figure 3a, the “SD Encoder Block A” contains 4 resnet lay-
ers and 2 ViTs, while the “×3” indicates that this block is
repeated three times. Text prompts are encoded using the
CLIP text encoder [66], and diffusion timesteps are encoded
with a time encoder using positional encoding. -->
<!-- The ControlNet structure is applied to each encoder level
of the U-net (Figure 3b). In particular, we use ControlNet
to create a trainable copy of the 12 encoding blocks and 1
middle block of Stable Diffusion. The 12 encoding blocks
are in 4 resolutions (64 ×64,32 ×32,16 ×16,8 ×8) with
each one replicated 3 times. The outputs are added to the
12 skip-connections and 1 middle block of the U-net. Since
Stable Diffusion is a typical U-net structure, this ControlNet
architecture is likely to be applicable with other models. -->
SD 由 encoder、middle block 和 decoder 组成，本质上是一个 U-Net。

ControlNet 用于每个 encoder 层，创建 12 个 encoding blocks 和 1 个 middle block 的 trainable copy。12 个 encoding blocks 在 4 个分辨率（64×64,32×32,16×16,8×8）上，每个分辨率重复 3 次。输出加到 12 个 skip-connections 和 1 个 middle block 上，如图：
![](image/Pasted%20image%2020241228110800.png)

> 提问：
> 1. 为什么在 decoder 上不使用 ControlNet：decoder 可以看成是根据特征图生成图像，由于要实现条件控制，所以需要修改输入的特征图，这里的 decoder 当成一个普通的任意的图像 decoder 理解即可
> 2. 为什么 skip connection 连接到的是 decoder（按照前面的说法，不应该是连在同一个模块的输出吗）：猜测应该是把这里的输出当成一种 condition 来控制 decoder 的合成
> 所以本质上就是从 encoder 中拿了一个 $\boldsymbol{x}$，输入一个 $\boldsymbol{c}$，得到某种特征，通过这个特征控制 decoder 的合成，背后的思路就是简单的条件控制合成，但是做法很巧妙。
<!-- The way we connect the ControlNet is computationally
efficient — since the locked copy parameters are frozen, no
gradient computation is required in the originally locked
encoder for the finetuning. This approach speeds up train-
ing and saves GPU memory. As tested on a single NVIDIA
A100 PCIE 40GB, optimizing Stable Diffusion with Control-
Net requires only about 23% more GPU memory and 34% 
more time in each training iteration, compared to optimizing
Stable Diffusion without ControlNet.-->
由于 lock 的参数被冻结，finetuning 时原始 encoder 不需要计算梯度。在单个 NVIDIA A100 PCIE 40GB 上测试，使用 ControlNet 优化 Stable Diffusion 只需要比不使用 ControlNet 多约 23% 的 GPU 内存和 34% 的时间。
<!-- Image diffusion models learn to progressively denoise
images and generate samples from the training domain. The
denoising process can occur in pixel space or in a latent
space encoded from training data. Stable Diffusion uses
latent images as the training domain as working in this space
has been shown to stabilize the training process [72]. Specif-
ically, Stable Diffusion uses a pre-processing method similar
to VQ-GAN [19] to convert 512 ×512 pixel-space images
into smaller 64 ×64 latent images. To add ControlNet to
Stable Diffusion, we first convert each input conditioning
image (e.g., edge, pose, depth, etc.) from an input size of
512 ×512 into a 64 ×64 feature space vector that matches
the size of Stable Diffusion. In particular, we use a tiny
network E(·) of four convolution layers with 4 ×4 kernels
and 2 ×2 strides (activated by ReLU, using 16, 32, 64, 128,
channels respectively, initialized with Gaussian weights and
trained jointly with the full model) to encode an image-space
condition ci into a feature space conditioning vector cf as, -->
为了将 ControlNet 添加到 SD，先将每个输入条件图像（如边缘、姿势、深度等）从 512×512 的输入大小转换为与 SD 匹配的 64×64 特征向量。具体来说，使用一个小网络 $\mathcal{E}(\cdot)$，包含四个卷积层，核大小为 4×4，步长为 2×2（由 ReLU 激活，分别使用 16、32、64、128 个通道，用高斯权重初始化，并与整个模型一起训练）将图像空间条件 $\boldsymbol{c}_i$ 编码为特征空间条件向量 $\boldsymbol{c}_f$：
$$\boldsymbol{c}_\mathrm{f}=\mathcal{E}(\boldsymbol{c}_\mathrm{i}).$$
<!-- The conditioning vector cf is passed into the ControlNet. -->
特征向量 $\boldsymbol{c}_f$ 传入 ControlNet。

### 训练
<!-- Given an input image z0, image diffusion algorithms
progressively add noise to the image and produce a noisy
image zt, where trepresents the number of times noise is
added. Given a set of conditions including time step t, text
prompts ct, as well as a task-specific condition cf, image
diffusion algorithms learn a network ϵθ to predict the noise
added to the noisy image zt with -->
给定输入图像 $\boldsymbol{z}_0$，图像扩散算法逐渐向图像添加噪声，生成带噪图像 $\boldsymbol{z}_t$，其中 $t$ 表示添加噪声的次数。给定一组条件，包括时间步 $t$、文本提示 $\boldsymbol{c}_t$，以及任务特定条件 $\boldsymbol{c}_f$，diffusion 模型学习 $\epsilon_\theta$ 预测添加到带噪图像 $\boldsymbol{z}_t$ 的噪声：
$$\mathcal{L}=\mathbb{E}_{\boldsymbol{z}_0,\boldsymbol{t},\boldsymbol{c}_t,\boldsymbol{c}_\mathrm{f},\epsilon\thicksim\mathcal{N}(0,1)}{\left[\|\epsilon-\epsilon_\theta(\boldsymbol{z}_t,\boldsymbol{t},\boldsymbol{c}_t,\boldsymbol{c}_\mathrm{f}))\|_2^2\right]},$$
<!-- where Lis the overall learning objective of the entire dif-
fusion model. This learning objective is directly used in
finetuning diffusion models with ControlNet. -->
这里的 $\mathcal{L}$ 是整个 diffusion 模型的学习目标，用于微调带 ControlNet 的 diffusion 模型。
<!-- In the training process, we randomly replace 50% text
prompts ct with empty strings. This approach increases
ControlNet’s ability to directly recognize semantics in the
input conditioning images (e.g., edges, poses, depth, etc.) as
a replacement for the prompt. -->
训练时随机替换 50% 的文本提示 $\boldsymbol{c}_t$ 为空字符串。
> 增加 ControlNet 直接识别输入条件图像（如边缘、姿势、深度等）中的语义的能力。
<!-- During the training process, since zero convolutions do
not add noise to the network, the model should always be
able to predict high-quality images. We observe that the
model does not gradually learn the control conditions but
abruptly succeeds in following the input conditioning image;
usually in less than 10K optimization steps. As shown in Fig-
ure 4, we call this the “sudden convergence phenomenon”. -->
由于 zero convolutions 不会向网络添加噪声，模型应该始终能够预测高质量的图像。作者观察到，模型**不会逐渐学习控制条件**，而是**突然可以合成遵循输入条件的图像**（通常在不到 10K 优化步骤内）。

### 推理
<!-- We can further control how the extra conditions of Con-
trolNet affect the denoising diffusion process in several ways.
 -->
可以通过几种方式进一步控制 ControlNet 的条件如何影响 diffusion 过程：
<!-- Classifier-free guidance resolution weighting. Stable Dif-
fusion depends on a technique called Classifier-Free Guid-
ance (CFG) [29] to generate high-quality images. CFG is
formulated as ϵprd= ϵuc + βcfg(ϵc−ϵuc) where ϵprd, ϵuc,
ϵc, βcfg are the model’s final output, unconditional output,
conditional output, and a user-specified weight respectively.
When a conditioning image is added via ControlNet, it can
be added to both ϵuc and ϵc, or only to the ϵc. In challenging
cases, e.g., when no prompts are given, adding it to both ϵuc
and ϵc will completely remove CFG guidance (Figure 5b);
using only ϵc will make the guidance very strong (Figure 5c).
Our solution is to first add the conditioning image to ϵc and 
then multiply a weight wi to each connection between Stable
Diffusion and ControlNet according to the resolution of each
block wi = 64/hi, where hi is the size of ith block, e.g.,
h1 = 8,h2 = 16,...,h13 = 64. By reducing the CFG guid-
ance strength , we can achieve the result shown in Figure 5d,
and we call this CFG Resolution Weighting.-->
+ Classifier-free guidance resolution weighting：SD 使用CFG 生成高质量图像 $\epsilon_{\mathrm{prd}}=\epsilon_{\mathrm{uc}}+\beta_{\mathrm{cfg}}(\epsilon_{\mathrm{c}}-\epsilon_{\mathrm{uc}})$，其中 $\epsilon_{\mathrm{prd}}$、$\epsilon_{\mathrm{uc}}$、$\epsilon_{\mathrm{c}}$、$\beta_{\mathrm{cfg}}$ 分别是模型的最终输出、无条件输出、有条件输出、权重。当通过 ControlNet 添加条件图像时，可以同时添加到 $\epsilon_{\mathrm{uc}}$ 和 $\epsilon_{\mathrm{c}}$，或者只添加到 $\epsilon_{\mathrm{c}}$。
> 在某些情况下，如没有 prompt 时，同时添加到 $\epsilon_{\mathrm{uc}}$ 和 $\epsilon_{\mathrm{c}}$ 会完全移除 CFG guidance；只加到 $\epsilon_{\mathrm{c}}$ 又会使 guidance 非常强。作者的解决方案是首先将条件图像添加到 $\epsilon_{\mathrm{c}}$，然后根据每个 block 的分辨率乘以一个权重 $w_i=64/h_i$，其中 $h_i$ 是第 $i$ 个 block 的大小，如 $h_1=8,h_2=16,\ldots,h_{13}=64$。此方法称为 CFG Resolution Weighting。
<!-- Composing multiple ControlNets. To apply multiple con-
ditioning images (e.g., Canny edges, and pose) to a single
instance of Stable Diffusion, we can directly add the outputs
of the corresponding ControlNets to the Stable Diffusion
model (Figure 6). No extra weighting or linear interpolation
is necessary for such composition. -->
+ Composing multiple ControlNets：为了将多个条件图像（如 Canny 边缘和姿势）应用于单个 Stable Diffusion 实例，可以直接将相应 ControlNet 的输出添加到 Stable Diffusion 模型中。不需要额外的加权或线性插值。

## 实验
