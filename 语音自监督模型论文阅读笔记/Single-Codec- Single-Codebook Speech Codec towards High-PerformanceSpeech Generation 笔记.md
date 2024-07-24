> Interspeech 2024，西工大、CUHK、出门问问

1. multi-codebook 的 codec 可以实现将 LLM 用于 TTS，但是影响效率
2. 提出 Single-Codec，单 codebook 的 codec，采用 disentangled VQ-VAE 将语音解耦为 time-invariant embedding 和 phonetically-rich 的离散序列，采用三种方法增强 encoder：
    1. BLSTM 模块用于 contextual modeling
    2. hybrid sampling module 用于减轻上采样和下采样的失真
    3. 重采样模块，使 discrete units 带有更多的语音信息
3. 与 multi-codebook codec（如 EnCodec 和 TiCodec）相比，Single-Codec 在只有 304bps 的带宽下实现更高的重构质量

## Introduction

1. 现有的 codec 需要 LLM 预测多个 discrete sequences，影响效率和稳定性
2. TiCodec 提出了一个全局编码器，将 speech units 中的 time-invariant 信息解耦，减少需要编码的帧级信息
3. 提出 Single-Codec，对 Mel Spectrogram 进行压缩和重构，以实现 speech generation，包含：
    1. global reference encoder，解耦 time-invariant 特征，使用连续的全局表征和更长的参考片段来捕获更多的 acoustic details
    2. BLSTM 模块用于 contextual modeling，发现相邻帧之间的相关性，增强 speech content clustering efficiency
    3. hybrid sampling module，使用卷积和池化实现下采样，使用转置卷积和复制实现上采样，减轻上采样和下采样失真
    4. resampling module，使 encoder 从 acoustic sequence 中提取更多的 phonetics-relevant 信息
4. 第一个专门为 LLM-based speech generation 设计的 single-codebook codec

## 方法

### 架构

架构如图：
![](image/Pasted%20image%2020240723165241.png)

基于 VQVAE，使用 mel 谱作为输入和重构，采用 Conformer-based encoder 编码 mel 谱 segment $seg_2$ 为 latent content representation $c$，然后 VQ 进行量化。卷积 decoder 从量化的 content representation $c$ 重构 mel 谱 $seg_2$。另外，使用 discriminator 提高生成质量。最后，使用 BigVGAN 从 codec 输出 waveform。

为了实现高质量的 single-codebook codec，增加了四个模块：
+ reference encoder，解耦 speech 中的 time-invariant 信息，得到 global representation $g$
+ hybrid sampling module，减轻 sampling loss
+ BLSTM 模块和 resampling module，增强 contextual information 和 phonetics-relevant information

### Reference Encoder

speech 包含多个信息，如 time-variant content、time-invariant timbre 和 acoustic environment。multi-codebook 的 codec 可以编码这些信息，但是对于 single-codebook codec，将所有信息压缩到有限数量的 discrete units 很难，因此解耦 得到 invariable 的 global information（如 timbre 和 acoustic environment），然后将 speech content 离散化。

引入 reference encoder，得到 global representation $g$，主要与 timbre 和 acoustic environment 相关。reference encoder 的输入是从输入 utterance 随机选择的 segment $seg_1$。设置 segment $seg_1$ 的长度为 600 帧，codec encoder 的输入 segment $seg_2$ 为 200 帧，短的 segment $seg_2$ 可以减少计算量和内存开销，而更长的 reference segment $seg_1$ 可以帮助获得更强大的 global features。reference encoder 的输出 $g$ 经过不同的线性层后，传递给 codec encoder 和 decoder，减去 encoder blocks 的输出并加到 decoder blocks 的输入。

### BLSTM 模块

speech content 的多样性导致很难实现 single-codebook codec。与 EnCodec 引入 LSTM 进行序列建模来提高 SI-SNR 不同，这里在 quantizer 前后添加 BLSTM 模块以增强 contextual information。发现可以提高 speech content 建模效率，更容易形成稳定的 clustering centers。
### Hybrid Sampling Module

neural codecs 通常使用采样模块来减少序列长度。之前的 codec 中的上采样和下采样操作通常通过卷积、转置卷积或池化和重复来实现。从而不可避免地产生 sampling loss，导致编解码能力降低。受 MR-HuBERT 启发，引入了改进的 hybrid sampling module，使用卷积和池化实现下采样，转置卷积和复制实现上采样。不同 sampling 方法的组合可以减轻 sampling 失真。
### Resampling Module

single-codebook speech codec 的主要目标是从 acoustic representations 中提取 short-term invariant speech units。acoustic representations 的多样性给 codebook vectors 的学习带来挑战。为了解决这个问题，提出一种新的 resampling module，首先对输入特征进行下采样用于 local modeling，然后上采样后进行 residual connect。这种沿时间轴的 bottlenecking 操作使得 encoder 可以从 acoustic sequence 中提取更多 phonetics-relevant 信息。

## 实验

使用五个开源数据集训练 speech codecs 和 VALL-E，包括 LibriTTS、Hi-Fi TTS、VCTK、AISHELL-1 和 AISHELL-3，共 1165.3 小时的英文和中文语音。

采用 EnCodec 和 TiCodec 作为 baseline。对于 VALL-E，使用 EnCodec 和 TiCodec 作为 baseline 评估 speech synthesis 的性能。

为了验证 Single-Codec 中设计的模块的有效性，进行了消融实验。

音频采样率为 24khz，Mel Spectrogram 的 hop length 和 window length 分别为 256 和 1024。下采样率为 4，总共下采样 1024 次（大约每秒 23 个 discrete tokens）。codebook 大小为 8192。codec 的模型大小为 256。卷积块中的中间隐藏状态大小为 256、512 和 1024，Conformer 块的隐藏大小为 1024。reference encoder 包含 6 层 2D 卷积和一个 GRU 层。residual block 包含两个 residual units，每个 residual unit 包含两个 kernel size 分别为 3 和 1 的一维卷积。discriminator 包含 4 层 kernel size 为 5 的 2D 卷积和 2 层 kernel size 为 3 的 2D 卷积。BLSTM 模块包含两个 LSTM 层，隐藏大小为 128。

训练时，Single-Codec 进行 300k 次迭代，batch size 为 1024。baseline model Encodec 使用 HifiCodec 复现的版本，训练 25 个 epochs。TiCodec 在两个 V100 GPU 上训练 300k 步，batch size 为 40。对于 VALL-E，使用 Amphion 复现的版本，动态 batch sizing，每个 batch 最大 token 限制为 4000。single-codebook codec 只使用 AR 阶段，而 multi-codebook codec 同时训练 AR 和 NAR 阶段。使用 8 个 A800 GPU 和 70 个 epochs 训练 VALL-E。

计算 STOI、PESQ、MCD、UTMOS 和 speaker cosine similarity（SPK）来客观评估 speech reconstruction 的质量。测试集由 100 个未见过的说话者随机选择的句子组成。结果如下：
![](image/Pasted%20image%2020240723174224.png)

### 消融结果

与 VQVAE 相比，Ref-short 和 Ref-long 在所有指标上表现更好。这表明解耦 speech 中的 global information 对于 single-codebook codec 是有效的。此外，Ref-long 在重构和 speaker similarity 上优于 Ref-short，说明更长的 reference segments 有助于捕获更准确的 time-invariant 信息和增强 content modeling。Ref-BLSTM、Ref-HybSam 和 Ref-BLSTM-HybSam 获得更高的重构质量，显示了 BLSTM 和 hybrid sampling modules 的有效性。此外，Ref-BLSTM-HybSam-Con 与 Ref-BLSTM-HybSam 相当，添加 resampling module 后进一步提高，即 Single-Code 效果最好。

### Commitment Loss 分析

进一步分析训练中的 commitment loss，探讨不同设计模块对 single-codebook codec 的影响。commitment loss 是量化前后的表征之间的差异。commitment loss 的收敛程度可以反映 encoder 输出和 codebook 中 cluster center 之间的关系。如图：
![](image/Pasted%20image%2020240723174955.png)

VQ-VAE 的 commitment loss 在模型训练后趋于发散，表明 time-invariant global information 和 time-variant content information 的纠缠阻碍了形成有限但多样的 content-related speech units。考虑 time-invariant decoupled modeling 后，Ref-short 的 loss 缓慢增加，表明 global information disentanglement 对 speech unit 学习的有效性。Ref-long 进一步验证了这一结果，说明更长的 reference segment 的有效性。Ref-HybSam 的 loss 曲线平坦，表明 hybrid sampling module 有效地提高了 codec 性能。此外，通过 BLSTM 模块进行 context modeling 的模型的 loss 都收敛了。这表明模型在量化之前已经学会了稳定的 phonetic units，说明 codec 中 context modeling 的有效性。

此外，考虑上表中的结果，观察到 commitment loss 与重构质量并不严格成反比。然而，commitment loss 的收敛状态（发散、平坦、收敛）确实与重构质量相关。具体来说，收敛的 codec 超过未收敛的 codec。这一结果进一步强调了在 single codebook codec 中实现稳定的 clustering center 的重要性，这直接影响整体的重构质量。

### 语音重构分析

比较 Single-Codec 和其他 codec 的 speech reconstruction 性能。结果表明，尽管带宽较低，Single-Codec 在重构质量和 speaker similarity 上超过其他 1 个 codebook 的 codec，并与 2 个 codebook 的 TiCodec 相当。VQVAE 优于 EnCodec，表明在 Mel Spectrogram 上的 codec 具有高量化效率。与 TiCodec 相比，Single-Codec 实现了更高的 speaker similarity 和重构质量，表明连续的 time-invariant representations 和更长的 reference length 的有效性。

### Zero-shot TTS 效果

为了评估应用于 speech synthesis 任务的 codec 的性能，使用从 EnCodec（1、4、8 codebooks）、TiCodec（1 codebook）和 Single-Codec 提取的 discrete tokens 训练 VALL-E。进行自然度 Mean Opinion Score（N-MOS）和 speaker similarity MOS（S-MOS）进行主观评估。测试集包括 30 个句子，包括中文和英文语音。邀请 20 位中文普通话母语者和熟悉英语的听众参加每个 MOS 测试。同时，使用 ASR 模型计算 word error rate（WER）来衡量语音可懂性。使用 WeSpeaker 提取说话者嵌入以计算说话者嵌入 cosine similarity。

结果如下：
![](image/Pasted%20image%2020240723175537.png)

Single-Codec 在自然度和 speaker similarity 方面优于其他模型。在 single-codebook 场景中，TiCodec-1VQ 和 Single-Codec 在 speaker similarity、自然度和稳定性方面明显优于其他 codec 模型。这是因为解耦 global information 使 frame-level codebook 更多关注 content modeling，并实现更多的 global information 传输。同时，Single-Codec 优于 Ticodec，表明连续的 global representation 和额外的 content modeling 的有效性。此外，Single-Codec 在 speaker similarity 和自然度方面优于多 codebook codec，WER 稍高于 Encodec-8VQ。这主要是因为更高的带宽带来更高分辨率的 speech unit 感知。