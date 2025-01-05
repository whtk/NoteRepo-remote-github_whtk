> Adobe、西北大学

1. 提出 Sketch2Sound，可以从时变控制信号（loudness、brightness、pitch 和文本提示）生成高质量音频，可以从 sonic imitations（声音模仿或参考声音形状）合成任意声音
2. Sketch2Sound 可以在任何 T2A latent DiT 中实现，只需 40k 步微调，比 ControlNet 更轻量
3. 为了从 sketchlike sonic imitations 合成音频，提出在训练时对控制信号进行随机中值滤波

## Introduction

1. foley sound 是使用特殊音效设计和表演电影的技术，声音设计师需要修改生成声音的 temporal 特征，与视觉同步
2. 很多方法把音频、平行乐器、旋律、声音事件时间戳和频率或多个结构控制信号作为生成模型的条件
3. 人类的声音是一种 gestural sonic 乐器，可以通过 discussion 和 vocal imitation 近似声音。
4. 提出 Sketch2Sound：可以从 sonic imitation prompt 生成高质量声音，可以提取 loudness、brightness 和 pitch 作为控制信号，通过中值滤波调整控制信号的时间精度
5. 这种方法不仅限于 vocal imitation，任何 sonic imitation 都可以驱动此模型；Sketch2Sound 可以添加到任何现有的 DiT 模型中，只需 40k 步微调，比 ControlNet 更轻量
6. 实验表明 Sketch2Sound 可以生成与 vocal imitation 控制信号相符的声音，同时保持与文本提示和音频质量的高度一致性；中值滤波技术可以提高音频质量和文本一致性


## 方法

提出一种在音频 LDM 模型上使用可解释的时变控制信号的方法，结构如图：
![](image/Pasted%20image%2020250103113457.png)

### 时变控制信号

选择三种控制信号：
+ loudness：从音频信号通过 A-weighted sum 提取每帧的 loudness
+ pitch 和周期性：使用 CREPE pitch 估计模型的原始 pitch 概率，将概率矩阵中小于 0.1 的概率置零
+ spectral centroid：定义为音频帧的频谱质心，将信号从线性频率空间转换为连续的 MIDI-like 表征（其实就是归一化到 0-1）

### 在 LDM 上使用时变控制信号

使用 [Stable Audio- Fast Timing-Conditioned Latent Audio Diffusion 笔记](Stable%20Audio-%20Fast%20Timing-Conditioned%20Latent%20Audio%20Diffusion%20笔记.md) 和 [Stable Audio 2- Long-form music generation with latent diffusion 笔记](Stable%20Audio%202-%20Long-form%20music%20generation%20with%20latent%20diffusion%20笔记.md) 中的 DiT，基于上面的控制信号来生成声音，其包含两部分：
1. VAE：将 48kHz 单声道音频压缩为 64 维连续向量序列，频率 40Hz
2. Transformer 解码器：生成新的 latent 序列，可以通过 VAE decoder 解码为音频

模型在大量专有、许可的音效数据集和公共 CC 许可的通用音频数据集上预训练，然后微调 40k 步 来 adapt 这里的控制信号。

控制信号可以从任何音频信号中提取，所以可以自监督地微调预训练的模型：
+ 给定任何输入音频信号，从中提取 loudness、centroid 和 pitch 作为控制信号，然后使用这些控制信号进行微调。
+ 微调过程与训练过程相同：从带有文本条件的噪声 latent 中学习反向扩散过程（包含控制条件）。

为了与 DiT 的 latent 频率对齐，控制信要与和 VAE latents 相同的帧率，通过引入一个简单的线性投影层将控制信号添加到噪声 latent 中来实现。

具体来说，给定 noise latent $\mathbf{z} \in \mathbb{R}^{D \times N}$（$D$ 维 embedding，$N$ 个序列长度）作为输入，引入控制信号 $\mathbf{c}_{\text{ctrl}} \in \mathbb{R}^{K \times N}$（$K$ 维，$N$ 个序列长度）通过线性投影层 $p_{\theta}(\mathbf{c}_{\text{ctrl}}) \in \mathbb{R}^{D \times N}$，然后将结果直接添加到输入 latent 中：$\mathbf{z}_{\text{ctrl}} = p_{\theta}(\mathbf{c}_{\text{ctrl}}) + \mathbf{z}$。

微调过程中没有对控制信号引入损失，但是也足够模型进行条件生成。同时微调的时候对控制信号进行 dropout（置 0），以确保可以在不需要所有控制信号的情况下生成。

推理时，使用 guidance scales $s_{\text{ctrl}}$ 和 $s_{\text{text}}$，可以调整控制信号和文本条件的强度。
> 发现使用单个 guidance scale 对所有三个控制信号（$s_{\text{ctrl}}$）就足够了。

为了解决 vocal imitation 和 target sound 之间的时间不匹配问题，在将控制信号输入模型之前，对控制信号应用不同窗口大小（1-25 控制帧）的随机中值滤波。
> 中值滤波常用于图像去噪。

## 实验设置

数据集：VimSketch，包含约 12k 个 vocal imitations，每个 imitation 包含文本描述和参考声音。

评估指标：
+ 音频质量：使用 40k 高质量音效数据集计算 FAD，使用 10k 生成的声音作为评估集，计算 VGGish 和 LAION-CLAP embeddings 的 FAD
+ 文本一致性：计算 CLAP embedding 余弦相似度
+ 控制信号一致性：计算输入和生成的控制信号（loudness、centroid、pitch）的 L1 误差

## 实验 & 结果（略）
