> NIPS 2024，Samsung Research

1. 语音增强的难点：实际录音包含各种形式的失真，如背景噪声、混响和麦克风伪影
2. 本文从理论上证明 GAN 对于 speech enhancement 任务至关重要
3. 研究了不同的 perceptual loss 特征提取器，提高了对抗训练的稳定性
4. 将 WavLM-based perceptual loss 集成到 MS-STFT 对抗训练中，从而提高模型训练稳定性

## Introduction

1. 通用的语音增强需要在各类失真中恢复语音
2. diffusion 模型效果很好，但是计算速度慢
3. 语音增强不需要模型学习整个条件分布，只需要获取最可能的干净语音样本
4. 本文展示了 GAN 在 speech enhancement 任务中的优势，贡献如下：
5. 贡献如下：
    1. 理论分析 LS-GAN loss，证明了 generator 会选择密度最大的点
    2. 研究了不同的 perceptual loss 特征提取器，提出了选择标准
    3. 提出了一种新的 universal speech enhancement 模型，集成了 perceptual loss 和 MS-STFT discriminator training，增强了 HiFi++ generator 架构

## Mode Collapse 和语音增强

语音增强的目标是，给定一个损坏的音频信号 $x$ ，恢复原始录音的语音特征，包括声音、语言内容和韵律。数学上看，语音增强模型应该找到给定 $x$ 的最可能的干净语音 $y$，即 $y=\arg\max_yp_\text{clean}(y|x)$ 。

但是上面的公式可能对于很多实际应用来说是多余的，因为生成的不确定性可以通过更直接的方式解决，比如基于语言内容的条件生成。

在实践中，应该将语音增强看作一个回归问题更，即预测条件分布的最高概率点。找到分布的最高点比学习整个分布容易得多。

作者认为 GAN 比 diffusion 模型更适合语音增强。因为 GAN 训练自然地导致生成器的 mode-seeking behaviour，更符合上面的公式。此外，GAN 只需要一次 forward，而不需要迭代。

令 $p_g(y|x)$ 为由 generator $g_\theta(x)$ 生成的波形分布。已有研究表明，使用 LS-GAN 可以最小化 Pearson $\chi^2$ divergence $\chi^2_\text{Pearson}(p_g || \frac{p_\text{clean} + p_g}{2})$。如果 $p_g(y|x)$ 在某种参数化下接近 $\delta(y-g_\theta(x))$，那么最小化这个 divergence 会导致 $g_\theta(x)=\arg\max_y p_\text{clean}(y|x)$。这意味着如果生成器从损坏的信号中确定性地预测干净的波形，LS-GAN loss 会鼓励生成器预测最大 $p_\text{clean}(y|x)$ 密度的点。

证明略。


## 用于语音生成的感知损失

对抗训练不稳定，可能导致 suboptimal 的解、mode collapse 和梯度爆炸。对于语音增强任务，对抗损失通常有额外的回归损失来稳定训练。在 GAN mode-seeking 中，回归损失可以看作是推动生成器走向“正确”（最可能）模式。

之前都是将语音增强问题视为预测任务。给定一个带噪的波形或频谱图，通过在波形和频谱中最小化点间距离或在两个域中最小化来预测干净信号。然而，由于信号的严重退化，对于恢复语音信号存在固有的不确定性，通常导致预测的语音过度平滑。为了减少平滑效应，可以选择一个比波形或频谱空间更少“纠缠”的回归空间。且此空间应该设计为，对于人类无法区分的声音（如相同发音人说的相同音素）的平均表示仍然是这个声音的表示。

提出两个规则来比较不同回归空间的结构：
+ 聚类规则：相同 speech sound 的表征应该形成一个簇，与不同语音形成的簇可分离。
+ SNR 规则：受不同水平的加性噪声污染的 speech sound 的表征应该随着噪声水平的增加单调地远离干净声音的簇。

聚类规则确保在特征空间中最小化样本之间的距离会导致样本对应于相同的声音。SNR 规则确保在特征之间最小化距离不会使信号受到噪声污染，即嘈杂的样本与干净的样本相距较远，如图：
![](image/Pasted%20image%2020241015112519.png)

实现上，通过以下步骤 check 这些条件：

1. 使用多说话人 VITS 采样 speech sounds。定义相同 sound 对应的波形组为：相同说话人生成的相同文本和相同音素持续时间的 0.5 秒的波形段。因此，waveform variation 是由先验分布的 latent 采样得到的。本文定义了 354 组 sound，80 个说话人说 177 个不同短语，每个短语 2 个不同说话人；每组采样 20 个样本，共 7080 个波形片段对应 354 组。
2. 将每个波形段映射到特征空间，然后将得到的 tensor 展平得到每个样本的 vector。使用 K-means 聚类，聚类数为组数（354）。聚类后，计算 K-means 的聚类与初始 sound group 聚类的 Rand index。将结果视为衡量聚类规则的 adherence 的方式。
3. 每个 group 内的波形段随机混入 10-20 dB 的噪声。然后将这些段映射到特征空间，得到每个样本的 vector。对于每个嘈杂的样本，计算其特征与干净特征簇中心的欧氏距离。然后计算 SNR 水平与样本到簇中心距离之间的负 Spearman 秩相关。负相关在所有 sound 组上平均，结果作为 SNR 规则的量化衡量。

使用这些指标，评估了几个语音特征提取器形成的不同特征空间的有效性，以及语音的传统表示。具体来说，使用 Wav2Vec 2.0、WavLM、EnCodec 的 encoder 和 CDPAM 生成特征。传统特征用的是波形和频谱特征。对于 Wav2Vec 2.0 和 WavLM，测试最后一个 transformer 层的输出和卷积编码器的输出 两种情况。还训练了一个 HiFi-GAN generator，每种特征类型用作均方误差损失计算的表征。

## FINALLY

### 架构

模型基于 [HiFi++- a Unified Framework for Bandwidth Extension and Speech Enhancement 笔记](HiFi++-%20a%20Unified%20Framework%20for%20Bandwidth%20Extension%20and%20Speech%20Enhancement%20笔记.md)。其 generator 包含四个部分：
+ SpectralUNet：使用二维卷积在频域进行预处理
+ Upsampler：基于 HiFi-GAN generator，用于将输入 tensor 的长度上采样到波形长度
+ WaveUNet：通过从原始波形中得到的相位信息改进 Upsampler 的输出
+ SpectralMaskNet：频域后处理，去除 WaveUNet 残留的伪影

本文对 HiFi++ generator 进行了两个修改：
1. 将 WavLM-large 模型输出（transformer 的最后一个隐藏状态）作为 Upsampler 的额外输入
2. 在 generator 的末尾引入 Upsample WaveUNet。架构使用 WaveUNet，decoder 中有一个额外的卷积上采样块，将时间分辨率上采样 3 倍。从而模型可以在输入 16 kHz 信号的情况下输出 48 kHz 信号。

### 数据和训练

## 相关工作（略）

## 结果（略）
