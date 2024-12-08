> NIPS 2024，CASIA、清华
<!-- Recent advancements in vision backbones have significantly improved their perfor-
mance by simultaneously modeling images’ local and global contexts. However,
the bidirectional interaction between these two contexts has not been well explored
and exploited, which is important in the human visual system. This paper pro-
poses a Fully Adaptive Self-Attention (FASA) mechanism for vision transformer
to model the local and global information as well as the bidirectional interaction
between them in context-aware ways. Specifically, FASA employs self-modulated
convolutions to adaptively extract local representation while utilizing self-attention
in down-sampled space to extract global representation. Subsequently, it conducts
a bidirectional adaptation process between local and global representation to model
their interaction. In addition, we introduce a fine-grained downsampling strategy
to enhance the down-sampled self-attention mechanism for finer-grained global
perception capability. Based on FASA, we develop a family of lightweight vision
backbones, Fully Adaptive Transformer (FAT) family. Extensive experiments
on multiple vision tasks demonstrate that FAT achieves impressive performance.
Notably, FAT accomplishes a 77.6% accuracy on ImageNet-1K using only 4.5M
parameters and 0.7G FLOPs, which surpasses the most advanced ConvNets and
Transformers with similar model size and computational costs. Moreover, our
model exhibits faster speed on modern GPU compared to other models. Code will
be available at https://github.com/qhfan/FAT. -->
1. 现有的模型通过建模图像中的局部和全局信息来提高性能，但局部和全局信息之间的双向交互很少使用
2. 本文提出 Fully Adaptive Self-Attention (FASA) 机制，用于 vision transformer，以 context-aware 方式建模局部和全局信息及其之间的双向交互：
    1. 采用 self-modulated convolutions 提取局部表征
    2. 在 down-sampled 空间中使用 self-attention 提取全局表征
    3. 在局部和全局表征之间进行 bidirectional adaptation 来建模交互
    4. 引入细粒度的下采样策略，增强 down-sampled self-attention 的全局感知能力
3. 基于 FASA，提出了一系列轻量化的 vision backbone，Fully Adaptive Transformer (FAT) family

## Introduction
<!-- Vision Transformers (ViTs) have recently garnered significant attention in the computer vision
community due to their exceptional ability for long-range modeling and context-aware characteristics.
However, because of the quadratic complexity of self-attention in ViT [13], its computational cost is
extremely high. As a result, many studies have emerged to improve ViT’s computational efficiency
and performance in various ways. For instance, some methods restrict tokens that perform self-
attention to a specific region and introduce inductive bias to ViT [34; 12; 65; 55]. Further, some
methods aim to transform ViT into lightweight backbones with fewer parameters and computational
requirements [38; 40; 4; 30; 37], achieving promising results but still not matching the performance
of the most advanced ConvNets [51]. How to design an excellent lightweight Vision Transformer
remains a challenge. -->
1. ViT 的计算复杂度很高，有很多提出了改进 ViT 的计算效率和性能的方法
<!-- In current state-of-the-art Vision Transformers, some either excel in creating local feature extraction
modules [34; 12; 65] or employing efficient global information aggregation modules [57; 58], while
others incorporate both [42; 41]. For instance, LVT [65] unfolds tokens into separate windows and
applies self-attention within the windows to extract local features, while PVT [57; 58] leverages
self-attention with downsampling to extract global features and reduce computational cost. Unlike
them, LITv2 [41] relies on window self-attention and spatial reduction attention to capture local and
global features, respectively. In terms of the local-global fusion, most methods use simple local-global
sequential structures [38; 40; 37; 26], whereas others combine local and global representation with
simple linear operations through local-global parallel structures [48; 41; 43]. However, few works
have investigated the bidirectional interaction between local and global information. Considering
the human visual system where bidirectional local-global interaction plays an important role, these
simplistic mixing methods are not fully effective in uncovering the intricate relationship between
local and global contexts. -->
2. 当前的 Vision Transformers 要么擅长创建 local feature extraction 模块，要么擅长使用 efficient global information aggregation 模块，很少有研究双向 local-global 交互
<!-- In Fig. 1, we illustrate how humans observe an
object and notice its body details. Using the
example of a fox, we can observe two types
of interaction that occur when humans focus
on either the fox’s nose or the entire animal.
In the first type of interaction, known as Lo-
cal to Global, our understanding of the local
feature transforms into the "Nose of a fox." In
the second type of interaction, called Global
to Local, the way we comprehend the global
feature changes to the "A fox with nose." It
can be seen that the bidirectional interaction
between local and global features plays an es-
sential role in the human visual system. Based
on this fact, we propose that a superior visual
model should not only extract good local and
global features but also possess adequate mod-
eling capabilities for their interaction. -->
3. 人类观察物体时会注意到其局部细节，这种双向 local-global 交互在人类视觉系统中起着重要作用
<!-- In this work, our objective is to model the bidirectional interaction between local and global contexts
while also improving them separately. To achieve this goal, we introduce three types of Context-
Aware Feature Aggregation modules. Specifically, as shown in Fig. 1, we first adaptively aggregate
local and global features using context-aware manners to obtain local and global tokens, respectively.
Then, we perform point-wise cross-modulation between these two types of tokens to model their
bidirectional interaction. We streamline all three processes into a simple, concise, and straightforward
procedure. Since we use context-aware approaches to adaptively model all local, global, and local-
global bidirectional interaction, we name our novel module the Fully Adaptive Self-Attention (FASA).
In FASA, we also further utilize a fine-grained downsampling strategy to enhance the self-attention
mechanism, which results in its ability to perceive global features with finer granularity. In summary,
FASA introduces only a small number of additional parameters and FLOPs, yet it significantly
improves the model’s performance. -->
4. 本文旨在建模局部和全局上下文之间的双向交互，提出了三种 Context-Aware Feature Aggregation 模块：
    1. 使用 context-aware 方式自适应聚合 local 和 global 特征，得到 local 和 global tokens
    2. 在这两种 token 之间进行 point-wise cross-modulation，建模双向交互
    3. 将所有过程简化为简单、简洁、直接的步骤，称为 Fully Adaptive Self-Attention (FASA)
    4. 在 FASA 中，利用细粒度的下采样策略增强 self-attention 机制，提高感知全局特征的精细度
<!-- Building upon FASA, we introduce the Fully Adaptive Transformer (FAT) family. The FATs follow
the hierarchical design [34; 57] and serve as general-purpose backbones for various computer vision
tasks. Through extensive experiments, including image classification, object detection, and semantic
segmentation, we validate the performance superiority of the FAT family. Without extra training data
or supervision, our FAT-B0 achieves a top-1 accuracy of 77.6% on ImageNet-1K with only 4.5M
parameters and 0.7G FLOPs, which is the first model surpasses the most advanced ConvNets with
similar model size and computational cost as far as we know. Additionally, as shown in Fig. 2, our
FAT-B1, B2, and B3 also achieve state-of-the-art results while maintaining similar model sizes and
computational costs. -->
5. 基于 FASA，引入 Fully Adaptive Transformer (FAT) family，FATs 遵循分层设计，适用于各种计算机视觉任务

## 相关工作（略）

## 方法
<!-- The overall architecture of the Fully Adaptive Transformer (FAT) is illustrated in Fig. 3. To process
an input image x ∈R3×H×W , we begin by feeding it into the convolutional stem used in [63]. This
produces tokens of size H
4 ×W
4 . Following the hierarchical designs seen in previous works [48;
50; 46; 49], we divide FAT into four stages to obtain hierarchical representation. Then we perform
average pooling on the feature map containing the richest semantic information. The obtained
one-dimensional vector is subsequently classified using a linear classifier for image classification. -->
整体结构如图：
![](image/Pasted%20image%2020241208113909.png)

给定输入图像 $x \in \mathbb{R}^{3 \times H \times W}$，首先通过 convolutional stem 处理，得到 token 大小为 $H/4 \times W/4$。将 FAT 分为四个阶段来得到分层的表征，然后在包含最丰富语义信息的特征图上进行平均池化。最后对得到的一维向量采用线性分类器进行图像分类。
<!-- A FAT block comprises three key modules: Conditional Positional Encoding (CPE) [6], Fully Adap-
tive Self-Attention (FASA), and Convolutional Feed-Forward Network (ConvFFN). The complete
FAT block is defined by the following equation (Eq. 1): -->
FAT block 包括三个模块：Conditional Positional Encoding (CPE)、Fully Adaptive Self-Attention (FASA) 和 Convolutional Feed-Forward Network (ConvFFN)。FAT block 定义如下：
$$\begin{gathered}
X=\mathrm{CPE}(X_{in})+X_{in}, \\
Y=\mathrm{FASA}(\mathrm{LN}(X))+X, \\
Z=\mathrm{ConvFFN}(\mathrm{LN}(Y))+\mathrm{Short}\mathrm{Cut}(Y).
\end{gathered}$$
<!-- Initially, the input tensor Xin ∈RC×H×W passes through the CPE to introduce positional informa-
tion for each token. The subsequent stage employs FASA to extract local and global representation
adaptively, while a bidirectional adaptation process enables interaction between these two types
of representation. Finally, ConvFFN is applied to enhance local representation further. Two kinds
of ConvFFNs are employed in this process. In the in-stage FAT block, ConvFFN’s convolutional
stride is 1, and ShortCut = Identity. At the intersection of the two stages, the ConvFFN’s convolu-
tional stride is set to 2, and ShortCut = DownSample is achieved via a Depth-Wise Convolution
(DWConv) [23] with the stride of 2 along with a 1 ×1 convolution. The second type of ConvFFN
accomplishes downsampling using only a small number of parameters, avoiding the necessity of
patch merging modules between the two stages, thereby saving the parameters. -->
首先，输入 tensor $X_{in} \in \mathbb{R}^{C \times H \times W}$ 通过 CPE 引入位置信息。然后用 FASA 提取局部和全局表征，通过 bidirectional adaptation 实现表征间的交互。最后用 ConvFFN 增强局部表征。
> 在这个过程中使用两种 ConvFFN。在 in-stage FAT block 中，ConvFFN 的卷积步长为 1，ShortCut = Identity。在两个阶段的交叉点，ConvFFN 的卷积步长设置为 2，通过 Depth-Wise Convolution (DWConv) 实现 ShortCut = DownSample。第二种 ConvFFN 仅使用少量参数实现下采样，避免两个阶段之间的 patch merging 模块，从而节省参数。

### Fully Adaptive Self-Attention
<!-- In Fig. 1, we present the visual system of humans, which not only captures local and global informa-
tion but also models their interaction explicitly. Taking inspiration from this, we aim to develop a 
similar module that can adaptively model local and global information and their interaction. This
leads to the proposal of the Fully Adaptive Self-Attention (FASA) module. Our FASA utilizes
context-aware manners to model all three types of information adaptively. It comprises three modules:
global adaptive aggregation, local adaptive aggregation, and bidirectional adaptive interaction. Given
the input tokens X ∈RC×H×W , each part of the FASA will be elaborated in detail.-->
