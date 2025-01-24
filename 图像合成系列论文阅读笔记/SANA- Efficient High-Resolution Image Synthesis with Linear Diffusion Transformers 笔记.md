> preprint 2024，NVIDIA、MIT、清华

1. 提出 Sana，T2I 模型，可以生成 4096×4096 的图像，模型包括：
    1. 深度压缩的 AE：将图像压缩 32 倍，减少了 latent token 的数量
    2. Linear DiT：用线性 attention 替换 DiT 中的 attention
    3. Decoder-only Text Encoder：用小型 LLM 替换 T5，通过指令增强图像-文本对齐
    4. 高效的训练和采样：提出 Flow-DPM-Solver 减少采样步骤，加速收敛
2. Sana-0.6B 与 Flux-12B 相比，体积小 20 倍，速度快 100 倍，可以在 16GB 笔记本 GPU 上运行，生成 1024×1024 的图像不到 1 秒

## Introduction

1. T2I 的一些关键点：
    1. 用 Transformer 替换 U-Net
    2. 使用 VLM 自动标记图像
    3. 改进 VAE 和 Text Encoder
    4. 超高分辨率图像生成
2. 目标：实现高质量、高分辨率的图像生成
2. 提出 Sana，可以生成 1024×1024 到 4096×4096 的高质量图像，提出：
    1. 深度压缩的 AE：将压缩因子提高到 32，输出的 latent tokens 减少 16 倍
    2. 高效的 Linear DiT：用线性 attention 替换 DiT 中的 attention，将计算复杂度从 $O(N^2)$ 降低到 $O(N)$，同时提出 Mix-FFN，将 3×3 深度卷积引入 MLP，聚合 token 的局部信息
    3. Decoder-only Small LLM 作为 Text Encoder：用 Gemma 作为 text encoder，增强对提示的理解和推理能力，设计复杂的人类指令（CHI）以提高图像-文本对齐
    4. 高效的训练和推理策略：提出一套自动标记和训练策略，提高文本和图像之间的一致性，提出 Flow-DPM-Solver，将推理采样步骤从 28-50 降低到 14-20 
3. Sana-0.6B 生成 4K 图像的速度比 FLUX 快 100 倍，1K 图像的速度比 FLUX 快 40 倍，生成 1024×1024 的图像只需要 0.37s

## 方法

### 深度压缩 AE

最常用的 AE 将图像从像素空间 $\mathbb{R}^{H×W×3}$ 映射到 latent 空间 $\mathbb{R}^{H/8×W/8×C}$，其中 C 是 latent channels 大小，DiT 中，超参 $P$ 为 patch size，latent 特征被分成 $P×P$ 的 patch，生成 $\frac{H}{PF}\times\frac{W}{PF}$ 个 token，一般 $P=2$。

PixArt、SD3 和 Flux 通常使用 AE-F8C4P2 或 AE-F8C16P2，AE 压缩 8 倍，DiT 压缩 2 倍，Sana 将压缩因子提高到 32 倍。

AE-F32C32 的重建能力与 SDXL 的 AE-F8C4 相当，AE 的小差异不会成为 DiT 的瓶颈。AE-F32C32P1 将 token 数量减少 4 倍，提高了训练和推理速度，降低了 GPU 内存需求。

作者实验发现，AE-F32C32P1 的生成结果优于 AE-F8C16P4 和 AE-F16C32P2，这表明让 AE 专注于高比压缩，让 DiT 专注于去噪是最佳选择。

选择 $C=32$ 作为最佳设置，$C=32$ 的重建质量更好，收敛速度与 $C=16$ 相似，$C=64$ 的收敛速度显著慢于 $C=32$。

### 高效的 Linear DiT 设计

DiT 的 self-attention 的计算复杂度为 $O(N^2)$，提出了 Linear DiT，用线性 attention 替换原始 self-attention，同时使用 Mix-FFN 替换原始 MLP-FFN，将 3×3 深度卷积引入以更好地聚合 token 信息，结构如图：
![](image/Pasted%20image%2020250123200836.png)

包含：
+ Linear Attention 模块：用 ReLU Linear attention 替换传统 softmax attention，ReLU Linear attention 主要用于高分辨率密集预测任务，其计算复杂度为 $O(N)$，只需要计算一次共享项，然后每个 query 重复使用：
$$O_i=\sum_{j=1}^N\frac{\mathrm{ReLU}(Q_i)\mathrm{ReLU}(K_j)^TV_j}{\sum_{j=1}^N\mathrm{ReLU}(Q_i)\mathrm{ReLU}(K_j)^T}=\frac{\mathrm{ReLU}(Q_i)\left(\sum_{j=1}^N\mathrm{ReLU}(K_j)^TV_j\right)}{\mathrm{ReLU}(Q_i)\left(\sum_{j=1}^N\mathrm{ReLU}(K_j)^T\right)}$$
+ Mix-FFN 模块：用 Mix-FFN 替换原始 MLP-FFN，Mix-FFN 由反向残差块、3×3 深度卷积和门控线性单元（GLU）组成，深度卷积增强了模型捕获局部信息的能力，补偿了 ReLU Linear attention 捕获局部信息能力较弱的问题
+ 无位置编码的 DiT：省略 DiT 中的位置编码，用 3×3 卷积隐式包含位置信息
+ Triton 加速训练/推理：用 Triton 融合线性 attention 前向和后向传递的内核，加速训练和推理

### Text Encoder

用 Decoder-only LLM 替换 T5 作为 Text Encoder：相比于 T5，decoder-only LLM 推理能力更强，可以使用 CoT 和 ICL 跟随人类指令，这里选择 Gemma-2 作为 text encoder
> Gemma-2-2B 的推理速度比 T5-XXL 快 6 倍，Clip Score 和 FID 与 T5-XXL 相当

用 Gemma-2 decoder 的最后一层特征作为 text embedding，但是发现，如果还是和 T5 一样直接使用 T5 text embedding 作为 key、value 和 image tokens（作为 query）进行交叉注意力训练会导致极端不稳定，训练损失经常变为 NaN

于是在 decoder-only text encoder 后添加了一个归一化层（RMSNorm），将 text embeddings 的方差归一化为 1.0，初始化一个小的可学习缩放因子（例如 0.01）并将其乘以 text embedding 进一步加速模型收敛。

复杂的人类指令可以进一步加强 text embedding，使用 LLM 的 in-context learning 设计复杂的人类指令（CHI），CHI 可以进一步提高图像-文本对齐能力。

当给定一个简短的提示时，CHI 有助于模型生成更稳定的内容，没有 CHI 的模型通常输出与提示无关的内容。

### 高效的训练/推理

一些技巧：
    1. 对于每个图像，使用四个 VLMs 进行标注。
    2. 使用 CLIP-Score 选择对应的 caption。首先计算所有 caption 的 clip score，根据 clip score 采样，引入温度参数 $\tau=\exp(c_i/\tau)/\sum_{j=1}^N\exp(c_j/\tau)$，温度可以调整采样强度，结果表明 caption 的变化对图像质量影响不大，但是可以提高训练过程中的语义对齐。
    3. 级联分辨率训练：跳过 256px 的预训练，直接从 512px 开始预训练，逐步微调模型到 1024px、2K 和 4K 分辨率。
    4. 基于 flow 的训练和推理

## 部署（略）

## 实验（略）
