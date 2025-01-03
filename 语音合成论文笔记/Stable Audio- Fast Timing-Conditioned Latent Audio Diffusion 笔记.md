> ICML 2024，Stability AI

1. 提出 Stable Audio，专注于生成 44.1kHz 双声道长音频：
    1. 基于 latent diffusion，latent 来自全卷积 VAE
    2. 条件是 text prompts 和 timing embeddings，可以控制生成的内容和长度
2. 在 A100 GPU 上可以在 8 秒内生成 95 秒的 44.1kHz 双声道音频

> 合成时间可控

## Introduction

1. diffusion 模型在训练和推理时计算量大，LDM 效率更高，可以实现更快的推理时间和生成长音频
2. audio diffusion 模型通常生成固定长度的输出，训练时会随机裁剪长音频，导致生成的音频可能在乐句中间开始或结束
3. Stable Audio 基于 LDM，条件是 text prompt 和 timing embeddings，可以控制生成的音频内容和长度
    1. 可以在训练窗口长度内生成指定长度的音频
    2. 在 A100 GPU 上可以在 8 秒内生成 95 秒的 44.1kHz 双声道音频
4. 提出了三种评价指标：
    1. 基于 OpenL3 embeddings 的 Fréchet Distance 评估生成的长音频的可信度
    2. Kullback-Leibler divergence 评估生成音频和参考音频的语义对应性
    3. CLAP score 评估生成音频和 text prompt 的对齐性

## 相关工作（略）

## 架构

Stable Audio 基于 LDM，包含 VAE、conditioning signal 和 diffusion model。结构如图：
![](image/Pasted%20image%2020241230103854.png)

### VAE

VAE 将 44.1kHz 双声道音频压缩为 latent。使用全卷积架构（133M 参数）进行任意长度的音频编解码，用的是 [DAC- High-Fidelity Audio Compression with Improved RVQGAN 笔记](DAC-%20High-Fidelity%20Audio%20Compression%20with%20Improved%20RVQGAN%20笔记.md) 的 encoder 和 decoder。从头开始训练，将输入的双声道音频降采样 1024 倍，得到的 latent 序列的通道维度为 64（将 2×L 的输入映射为 64×L/1024 的 latent），总数据压缩比为 32。

### Conditioning

Text encoder：使用 CLAP embeddings。实验发现，从头训练的 CLAP embeddings 优于开源的 CLAP 和 T5 embeddings。使用 CLAP text encoder 的倒数第二层的特征作为 conditioning signal。

Timing embeddings：计算 audio chunk 的开始时间和总时长，如图：
![](image/Pasted%20image%2020241230104408.png)

然后将这些值转换为离散的 learned embeddings，和文本特征 concate 后传入 U-Net 的 cross-attention layer。
> 对于比训练窗口短的音频，用 silence 填充。
推理时，输入 seconds_start 和 seconds_total 作为 conditioning，从而可以指定输出音频的总长度。

### Diffusion Model

基于 U-Net 架构，包含 4 个对称的下采样 encoder 和上采样 decoder，encoder 和 decoder 之间有 skip connection。4 个 level 的 channel 数分别为 1024、1024、1024 和 1280，downsample 分别为 1、2、2 和 4。最后一个 encoder block 后有 1280-channel bottleneck block。每个 block 由 2 个卷积残差层和一系列 self-attention 和 cross-attention layers 组成。每个 encoder 或 decoder block 有三个 attention layers，第一个 U-Net level 只有一个。通过 FiLM layers 将 diffusion timestep conditioning 传入模型，通过 cross-attention layers 将 prompt 和 timing conditioning 传入模型。

### 推理

推理时使用 DPM-Solver++ 和 classifier-free guidance（scale 为 6），使用 100 个 diffusion steps。Stable Audio 适用于变长、长音频生成。对于短于窗口长度的音频，可以裁剪 silence。

## 训练

数据集：包含 806,284 个音频（19,500 小时），其中音乐（66% 或 94%）、音效（25% 或 5%）和乐器音轨（9% 或 1%），对应的文本来自 AudioSparx。

VAE：训练时，使用多分辨率 sum 和 difference STFT loss，窗口长度为 2048、1024、512、256、128、64 和 32。使用了 adversarial 和 feature matching losses，使用双声道音频的多尺度 STFT 判别器。

CLAP：训练 100 个 epoch，损失为 language-audio contrastive loss。

Diffusion model：训练 640,000 个 steps，使用 exponential moving average 和 automatic mixed precision，音频重采样为 44.1kHz，使用 v-objective 和 cosine noise schedule，应用 dropout（10%）到 conditioning signals。训练时 text encoder 被冻结。

Prompt：训练时，从 metadata 生成 text prompts，对于一半的样本，包含 metadata-type（如 Instruments 或 Moods）并用 | 连接，另一半不包含 metadata-type 并用逗号连接。

## 方法

### 评估指标

+ FDopenl3：使用 Fréchet Distance 评估生成音频和参考音频的相似性
+ KLpasst：使用 PaSST 计算生成音频和参考音频标签的 KL 散度
+ CLAPscore：计算给定 text prompt 的 CLAPLAION text embedding 和生成音频的 CLAPLAION audio embedding 的余弦相似度
+ 人类评分：音频质量、文本对齐、音乐性、立体声正确性、音乐结构

### 评估数据（略）

### Baseline

Baseline：AudioLDM2、MusicGen 和 AudioGen。AudioLDM2 有三个 variant：AudioLDM2-48kHz、AudioLDM2-large 和 AudioLDM2-music。MusicGen 有三个 variant：MusicGen-small、MusicGen-large 和 MusicGen-large-stereo。

## 实验（略）
