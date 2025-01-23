> preprint 2024，NVIDIA、MIT、清华
<!-- 翻译 & 理解 -->
<!-- We introduce Sana, a text-to-image framework that can efficiently generate images
up to 4096×4096 resolution. Sana can synthesize high-resolution, high-quality
images with strong text-image alignment at a remarkably fast speed, deployable
on laptop GPU. Core designs include: (1) Deep compression autoencoder: un-
like traditional AEs, which compress images only 8×, we trained an AE that can
compress images 32×, effectively reducing the number of latent tokens. (2) Lin-
ear DiT: we replace all vanilla attention in DiT with linear attention, which is
more efficient at high resolutions without sacrificing quality. (3) Decoder-only
text encoder: we replaced T5 with modern decoder-only small LLM as the text
encoder and designed complex human instruction with in-context learning to en-
hance the image-text alignment. (4) Efficient training and sampling: we propose
Flow-DPM-Solver to reduce sampling steps, with efficient caption labeling and
selection to accelerate convergence. As a result, Sana-0.6B is very competitive
with modern giant diffusion model (e.g. Flux-12B), being 20 times smaller and
100+ times faster in measured throughput. Moreover, Sana-0.6B can be deployed
on a 16GB laptop GPU, taking less than 1 second to generate a 1024×1024 reso-
lution image. Sana enables content creation at low cost. Code and model will be
publicly released. -->
1. 提出 Sana，T2I 模型，可以生成 4096×4096 的图像，模型包括：
    1. 深度压缩的 AE：将图像压缩 32 倍，减少了 latent token 的数量
    2. Linear DiT：用线性 attention 替换 DiT 中的 attention
    3. Decoder-only Text Encoder：用小型 LLM 替换 T5，通过指令增强图像-文本对齐
    4. 高效的训练和采样：提出 Flow-DPM-Solver 减少采样步骤，加速收敛
2. Sana-0.6B 与 Flux-12B 相比，体积小 20 倍，速度快 100 倍，可以在 16GB 笔记本 GPU 上运行，生成 1024×1024 的图像不到 1 秒

## Introduction
<!-- In the past year, latent diffusion models have made significant progress in text-to-image research
and have generated substantial commercial value. On one hand, there is a growing consensus among
researchers regarding several key points: (1) Replace U-Net with Transformer architectures (Chen
et al., 2024b;a; Esser et al., 2024; Labs, 2024), (2) Using Vision Language Models (VLM) for
auto-labelling images (Chen et al., 2024b; OpenAI, 2023; Zhuo et al., 2024; Liu et al., 2024) (3)
Improving Variational Autoencoders (VAEs) and Text encoder (Podell et al., 2023; Esser et al.,
2024; Dai et al., 2023) (4) Achieving ultra High-resolution image generation (Chen et al., 2024a),
etc. On the other hand, industry models are becoming increasingly large, with parameter counts
escalating from PixArt’s 0.6B parameters to SD3 at 8B, LiDiT at 10B, Flux at 12B, and Playground
v3 at 24B. This trend results in extremely high training and inference costs, creating challenges for
most consumers who find these models difficult and expensive to use. Given these challenges, a
pivotal question arises: Can we develop a high-quality and high-resolution image generator that is
computationally efficient and runs very fast on both cloud and edge devices? -->
1. T2I 的一些关键点：
    1. 用 Transformer 替换 U-Net
    2. 使用 VLM 自动标记图像
    3. 改进 VAE 和 Text Encoder
    4. 超高分辨率图像生成
2. 目标：实现高质量、高分辨率的图像生成
<!-- This paper proposes Sana, a pipeline designed to efficiently and cost-effectively train and synthesize
images at resolutions ranging from 1024×1024 to 4096×4096 with high quality. To our knowl-
edge, no published works have directly explored 4K resolution image generation, except for PixArt-
Σ (Chen et al., 2024a). However, PixArt-Σ is limited to generating images close to 4K resolution
(3840×2160) and is relatively slow when producing such high-resolution images. To achieve this
ambitious goal, we propose several core designs: -->
2. 提出 Sana，可以生成 1024×1024 到 4096×4096 的高质量图像，提出：
<!-- Deep Compression Autoencoder: We introduce a new Autoencoder (AE) in Section 2.1 that ag-
gressively increases the scaling factor to 32. In the past, mainstream AEs only compressed the
image’s length and width with a factor of 8 (AE-F8). Compared with AE-F8, our AE-F32 outputs
16 ×fewer latent tokens, which is crucial for efficient training and generating ultra-high-resolution
images, such as 4K resolution. -->
    1. 深度压缩的 AE：将压缩因子提高到 32，输出的 latent tokens 减少 16 倍
<!-- Efficient Linear DiT: We introduce a new linear DiT to replace vanilla quadratic attention modules
(Section 2.2). The computational complexity of the original DiT’s self-attention is O(N2), which
increases quadratically when processing high-resolution images. We replace all vanilla attention
with linear attention, reducing the computational complexity from O(N2) to O(N). At the same
time, we propose Mix-FFN, which integrates 3×3 depth-wise convolution into MLP to aggregate
the local information of tokens. We argue that linear attention can achieve results comparable to
vanilla attention with proper design and is more efficient for high-resolution image generation (e.g.,
accelerating by 1.7×at 4K). Additionally, the indirect benefit of Mix-FFN is that we do not need
position encoding (NoPE). For the first time, we removed the positional embedding in DiT and find
no quality loss. -->
    2. 高效的 Linear DiT：用线性 attention 替换 DiT 中的 attention，将计算复杂度从 $O(N^2)$ 降低到 $O(N)$，同时提出 Mix-FFN，将 3×3 深度卷积引入 MLP，聚合 token 的局部信息
<!-- Decoder-only Small LLM as Text Encoder: In Section 2.3, we utilize the latest Large Language
Model (LLM), Gemma, as our text encoder to enhance the understanding and reasoning capabilities
regarding user prompts. Although text-to-image generation models have advanced significantly over
the years, most existing models still rely on CLIP or T5 for text encoding, which often lack robust
text comprehension and instruction-following abilities. Decoder-only LLMs, such as Gemma, ex-
hibit strong text understanding and reasoning capabilities, demonstrating an ability to follow human
instructions effectively. In this work, we first address the training instability issues that arise from di-
rectly adopting an LLM as a text encoder. Secondly, we design complex human instructions (CHI) to
leverage the LLM’s powerful instruction-following, in-context learning, and reasoning capabilities
to improve image-text alignment. -->
    3. Decoder-only Small LLM 作为 Text Encoder：用 Gemma 作为 text encoder，增强对提示的理解和推理能力，设计复杂的人类指令（CHI）以提高图像-文本对齐
<!-- Efficient Training and Inference Strategy: In Section 3.1, we propose a set of automatic labelling
and training strategies to improve the consistency between text and images. First, for each image, we
utilize multiple VLMs to generate re-captions. Although the capabilities of these VLMs vary, their
complementary strengths improve the diversity of the captions. In addition, we propose a clipscore-
based training strategy (Section 3.2), where we dynamically select captions with high clip scores
for the multiple captions corresponding to an image based on probability. Experiments show that
this approach improve training convergence and text-image alignment. Furthermore, We propose a
Flow-DPM-Solver that reduces the inference sampling steps from 28-50 to 14-20 steps compared to
the widely used Flow-Euler-Solver, while achieving better results. -->
    4. 高效的训练和推理策略：提出一套自动标记和训练策略，提高文本和图像之间的一致性，提出 Flow-DPM-Solver，将推理采样步骤从 28-50 降低到 14-20 步
<!-- In conclusion, our Sana-0.6B achieves a throughput that is over 100×faster than the current state-of-
the-art method (FLUX) for 4K image generation (Figure 2), and 40×faster for 1K resolution (Fig-
ure 4), while delivering competitive results across many benchmarks. In addition, we quantize
Sana-0.6B and deploy it on an edge device, as detailed in Section 4. It takes only 0.37s to generate
a 1024×1024 resolution image on a customer-grade 4090 GPU, providing a powerful foundation
model for real-time image generation. We hope that our model can be efficiently utilized by all
industry professionals and everyday users, offering them significant business value. -->
3. Sana-0.6B 生成 4K 图像的速度比 FLUX 快 100 倍，1K 图像的速度比 FLUX 快 40 倍，生成 1024×1024 的图像只需要 0.37s

## 方法

<!-- DEEP COMPRESSION AUTOENCODER -->
### 深度压缩 AE
<!-- To mitigate the excessive training and inference costs associated with directly running diffusion
models in pixel space, Rombach et al. (2022) proposed latent diffusion models that operate in a
compressed latent space produced by pre-trained autoencoders. The most commonly used autoen-
coders in previous latent diffusion works (Peebles & Xie, 2023; Bao et al., 2022; Cai et al., 2024;
Esser et al., 2024; Dai et al., 2023; Chen et al., 2024b;a) feature a down-sampling factor of F = 8,mapping images from pixel space RH×W×3 to latent space R H
8 ×W
8 ×C, where C represents the
number of latent channels. In DiT-based methods (Peebles & Xie, 2023), the number of tokens pro-
cessed by the diffusion models is also influenced by another hyper-parameter, P, known as patch
size. The latent features are grouped into patches of size P ×P, resulting in H
PF × W
PF tokens. A
typical patch size in previous works is 2.
 -->
最常用的 AE 将图像从像素空间 $\mathbb{R}^{H×W×3}$ 映射到 latent 空间 $\mathbb{R}^{H/8×W/8×C}$，其中 C 是 latent channels 的数量，DiT 方法中，另一个超参数 P 影响了 token 的数量，latent 特征被分成大小为 P×P 的 patch，结果是 $H/PF×W/PF$ 个 token，之前的 patch 大小通常是 2