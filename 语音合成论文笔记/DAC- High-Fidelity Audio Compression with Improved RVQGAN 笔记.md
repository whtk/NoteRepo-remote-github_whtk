> NIPS 2023，Descript, Inc

1. 提出一种高保真、通用的神经音频压缩算法，将 44.1 KHz 音频压缩到 8kbps 带宽的离散 token 中，实现 90 倍压缩：
    1. 用的是图像里面的量化技术
    2. 还有对抗和重构损失
2. 用一个通用模型压缩所有音频（语音、环境、音乐等）

## Introduction

1. 音频生成通常分为两个阶段：
    1. 基于中间表征（如 mel 谱）预测音频
    2. 基于条件信息（如文本）预测中间表征
    3. 然后可以用分层生成模型来解释这个过程
2. 用 VQ-VAE 训练离散潜变量的 VAE
    1. 离散潜变量更好，因为可以用强大的自回归模型训练先验
    2. 虽然建模先验很简单，用量化自编码器建模 discrete latent code 还是很难
3. 学习离散码可以看作是一个有损压缩任务，通过将音频信号压缩到一个离散潜空间，使用固定长度码书的自编码器的表示进行矢量量化
4. 音频压缩模型需要满足以下特性：
    1. 高保真、无伪影的重构音频
    2. 高压缩率，同时进行降采样，学习紧凑表征，保留高级结构
    3. 可以用通用模型处理所有类型的音频，如语音、音乐、环境声音、不同的音频编码（如 mp3）以及不同的采样率
5. 最近的神经音频压缩算法（如 SoundStream 和 EnCodec）虽然部分满足这些特性，但通常会出现 GAN 生成模型的问题
    1. 如音频伪影、音高和周期性伪影、高频不完美建模等
    2. 模型通常针对特定类型的音频信号，如语音或音乐，难以建模通用声音
6. 本文贡献：
    1. 提出了 Improved RVQGAN，可以将 44.1 KHz 音频压缩到 8 kbps 的离散 code 中，质量损失小，伪影少
    2. 通过改进 codebook，解决了现有模型中存在的不充分利用带宽的问题
    3. 认为 quantizer dropout 虽然使得单个模型支持可变比特率，但是实际上损害了全频带音频质量
    4. 添加 periodic inductive biase、多尺度 STFT 判别器、多尺度 mel 损失 
    5. 是一个通用音频压缩模型，能够处理语音、音乐、环境声音、不同的采样率和音频编码格式

## 相关工作（略）

## Improved RVQGAN 模型

基于 VQ-GANs 框架，用的是 [SoundStream- An End-to-End Neural Audio Codec 笔记](../语音领域其他论文笔记/SoundStream-%20An%20End-to-End%20Neural%20Audio%20Codec%20笔记.md) 和 [EnCodec- High Fidelity Neural Audio Compression 笔记](../语音领域其他论文笔记/EnCodec-%20High%20Fidelity%20Neural%20Audio%20Compression%20笔记.md) 的模式，使用了 全卷积的 encoder-decoder 网络，包含时间降采样。使用 RVQ 对编码进行量化，。训练时使用 quantizer dropout，使得单个模型可以在多个目标比特率下运行。训练时使用频域重构损失、对抗损失和感知损失。

音频信号的采样率为 $f_s$ (Hz)，编码器的 striding factor 为 $M$，RVQ 的层数为 $N_q$，则离散的 code matrix 为 $S \times N_q$，其中 $S$ 是帧率，定义为 $f_s / M$。下表为不同模型对比：
![](image/Pasted%20image%2020240716121009.png)

### Periodic activation function

音频波形有周期性，而当前的非自回归模型会出现音高和周期性伪影。激活函数（如 Leaky ReLU）在周期信号时表现不佳，对于音频合成的 out-of-distribution 泛化能力也不好。

采用 Snake 激活函数。Snake 函数定义为 $snake(x) = x + \frac{1}{\alpha} \sin^2(\alpha x)$，其中 $\alpha$ 控制信号周期性的频率。实验中发现，用 Snake 函数替换 Leaky ReLU 激活函数可以显著提高音频保真度。

### Improved residual vector quantization

VQ-VAE 训练时存在很多问题，如 codebook 初始化不好，导致很多 code 没有使用，有效 codebook 大小减小，目标比特率降低，重构质量变差。

最近的方法使用 k-means 初始化 codebook ，并在某些 codebook 未使用时手动 restart。但是发现 EnCodec 模型仍然存在 codebook 未充分利用的问题。

本文使用 Improved VQGAN 的方法：
+ factorized codes：解耦 code lookup 和 code embedding，在低维空间进行 code lookup，高维空间进行 code embedding
+ L2-normalized codes：将编码后的和 codebook 中的向量进行 L2 归一化，将欧氏距离转换为余弦相似度，有助于提高稳定性和质量

这两个技巧显著提高了 codebook 的使用率和重构质量，同时更容易实现。模型可以使用原始 VQ-VAE codebook 和 commitment losses 进行训练，无需 k-means 初始化。

### Quantizer dropout rate

SoundStream 引入了 quantizer dropout，用于训练单个可变比特率的压缩模型。每个输入样本随机采样 $n \sim \{1,2,...,N_q\}$，只使用前 $n_q$ 个量化器进行训练。但是发现 quantizer dropout 会降低全频带音频的重构质量，如图：
![](image/Pasted%20image%2020240716154635.png)


这里则将 quantizer dropout 用到每个输入样本，概率为 $p$。发现 $p=0.5$ 时，在低比特率下，重构质量接近 baseline，同时缩小了与不使用 quantizer dropout （$p=0.0$） 训练的全频带质量的差距。

> 为什么 quantizer dropout 有效：
> 1. 可以使得 quantized codes 学习从最重要到最不重要的信息
> 2. 当用 1 到 $N_q$ 个 codebook 重构 codes 时，每个 codebook 都会增加更多的细节
> 3. 从而可以训练基于这些 codes 的分层生成模型，例如将 codes 划分为“coarse” tokens 和 “fine” tokens

### Discriminator design

使用多尺度（MSD）和多周期波形判别器（MPD）提高音频保真度。但是生成音频的频谱图仍然可能出现模糊，高频部分过度平滑。UnivNet 提出了多分辨率频谱图判别器（MRSD），BigVGAN 发现这有助于减少音高和周期性伪影。但是使用幅度谱图会丢失相位信息，而相位信息可以用来惩罚相位建模错误。此外，高采样率下的高频建模还是很难。

于是使用 complex STFT 判别器，在多个 time-scales 下提高相位建模能力。还发现，将 STFT 分成子带可以稍微提升对高频的预测，减轻混叠伪影，因为判别器可以学习特定子带的区分特征，并为生成器提供更强的梯度信号。

### 损失函数

+ 频域重构损失：mel 重构损失提高稳定性、保真度和收敛速度，多尺度频谱损失则提高不同时间尺度的频率建模。本文结合两种方法，使用 mel-spectrograms 的 L1 损失，窗口长度为 [32, 64, 128, 256, 512, 1024, 2048]，hop length 为 window_length / 4

> EnCodec 使用类似的损失函数，但是有 L1 和 L2 损失项，mel bin size 为 64。发现固定 mel bin size 会导致频谱图中出现空洞，因此使用 mel bin size 为 [5, 10, 20, 40, 80, 160, 320]。

+ 对抗损失：使用 multi-period discriminator 进行波形判别，使用 multi-band multi-scale STFT 判别器进行频域判别。使用 HingeGAN 对抗损失、L1 的 feature matching 损失

+ Codebook 学习：使用原始 VQ-VAE 的 codebook 和 commitment losses，使用 straight-through estimator 反向传播梯度

+ 权重：mel 损失 15.0，feature matching 损失 2.0，对抗损失 1.0，codebook 和 commitment 损失分别为 1.0 和 0.25。不使用 EnCodec 中提出的 loss balancer。

## 实验

数据集包含语音、音乐和环境声音：
+ 语音：DAPS 数据集、DNS Challenge 4 的干净语音片段、Common Voice 数据集、VCTK 数据集
+ 音乐：MUSDB 数据集、Jamendo 数据集
+ 环境声音：AudioSet 的平衡和不平衡训练片段

所有音频重采样到 44kHz。

训练时，从每个音频文件中提取短片段，归一化到 -24 dB LUFS。随机 shift 相位来进行数据增强。使用 AudioSet 的评估片段、DAPS 中的两个说话者（F10、M10）和 MUSDB 的测试集。提取 3000 10 秒片段作为测试集。

引入了平衡的数据采样技术来解决数据集中的采样率不均匀问题。

将数据集分为全频带和非全频带两部分，采样时确保采样到全频带的数据。
确保每个 batch 中有相等数量的语音、音乐和环境声音数据。

模型包含 encoder、RVQ 和 decoder。encoder 有 4 层，每层下采样率为 [2, 4, 8, 8]。decoder 有 4 层，上采样率为 [8, 8, 4, 2]。decoder 维度设置为 1536。总共有 76M 参数，其中 encoder 22M，decoder 54M。还尝试了 decoder 维度为 512（31M 参数）和 1024（49M 参数）。

使用 multi-period discriminator 和复杂的多尺度 STFT 判别器。对于 multi-period discriminator，使用 periods 为 [2, 3, 5, 7, 11]，对于 STFT 判别器，使用 window lengths 为 [2048, 1024, 512]，hop-length 为 window length 的 1/4。对于 STFT 的 band-splitting，使用 band-limits 为 [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]。对于重构损失，使用 log-mel spectrograms 的距离，window lengths 为 [32, 64, 128, 256, 512, 1024, 2048]，对应的 mels 为 [5, 10, 20, 40, 80, 160, 320]。hop length 为 window length 的 1/4。使用 feature matching 和 codebook losses。

在 ablation study 中，batch size 为 12，迭代 250k 次。在最终模型中，batch size 为 72，迭代 400k 次。使用 AdamW 优化器，学习率为 $1e-4$，$\beta_1=0.8$，$\beta_2=0.9$。每步衰减学习率，$\gamma=0.999996$。

采用下面的评价指标：
+ ViSQOL：基于频谱相似性估计平均意见分数
+ Mel distance：重构和真实波形的 log mel spectrograms 之间的距离
+ STFT distance：重构和真实波形的 log magnitude spectrograms 之间的距离
+ SI-SDR：类似于信噪比的波形距离，但是不受幅度差异影响
+ Bitrate efficiency：对大型测试集应用每个 codebook 的熵之和除以所有 codebook 的比特数

进行了 MUSHRA 测试，听众对 12 个随机选取的 10 秒样本进行评分，每个域选取 4 个样本。比较了本文提出的系统在 2.67kbps、5.33kbps 和 8kbps 下的表现和 EnCodec 在 3kbps、6kbps 和 12kbps 下的表现。

### 消融实验

### 和其他方法的比较

和 EnCodec、Lyra、Opus 进行比较，使用公开的开源实现。在不同比特率下进行客观和主观评估。结果如下表：
![](image/Pasted%20image%2020240716163405.png)

结论：提出的 codec 在所有比特率下都优于竞争对手，同时建模了更宽的 22kHz 带宽。

MUSHRA 测试结果如下图：
![](image/Pasted%20image%2020240716163529.png)

codec 在所有比特率下都比 EnCodec 得分高。即使在最高比特率下，仍然低于参考 MUSHRA 分数，说明还有改进的空间。
> 最终模型的指标仍然低于消融实验中训练的 24kbps 模型，表明可能通过增加最大比特率来缩小差距。


在相同的配置下（24 kbps）与 EnCodec 进行比较，如下：
![](image/Pasted%20image%2020240716163754.png)

不同音频类型的 MUSHRA 结果如下：
![](image/Pasted%20image%2020240716163958.png)


## 补充
关于 FACTORIZED CODEBOOK 的补充，来自论文 Vector-quantized Image Modeling with Improved VQGAN 的附录：
![](image/Pasted%20image%2020240716171006.png)