> ICASSP 2022，西工大、ASLP

1. 提出 VISinger，端到端高质量歌声合成，可以直接从歌词和乐谱中合成歌声
2. 基于 VITS，但对 prior encoder 进行了改进
    1. 引入 length regulator 和 frame prior network，得到 frame-level 声学特征的均值和方差
    2. 引入 F0 predictor，引导 frame prior network，提高稳定性
    3. 修改 duration predictor，预测 phoneme 到 note（音符） 的 duration ratio，辅助歌声 note normalization
3. 在中文数据集上实验，效果优于 FastSpeech + Neural Vocoder 和 oracle VITS

## Introduction

1. 相比于 TTS，SVS 需要更多的声学特征，如 F0，更多的发音技巧，如颤音，更难建模
2. 本文基于 VITS，提出 VISinger，第一个解决两阶段 mismatch 问题的端到端歌声合成系统，相比于语音，歌声合成又一些难点：
    1. VITS 中用的是 phoneme-level 的均值和方差，这里引入 length regulator 和 frame prior network，得到 frame-level 均值和方差
    2. 引入 F0 predictor，引导 frame prior network，提高稳定性
    3. 修改 duration predictor，预测 phoneme to note 的 duration ratio，辅助歌声 note normalization

## 方法

如图：
![](image/Pasted%20image%2020240205171550.png)
包含三个部分：
+ posterior encoder 从波形 $y$ 中提取表征 $z$
+ prior encoder
+ decoder 从 $z$ 重构波形 $\hat{y}$

公式表述为：
$$\begin{aligned}z&=Enc(y)\sim q(z|y)\\\hat{y}&=Dec(z)\sim p(y|z)\end{aligned}$$

然后用 prior encoder 得到给定音乐乐谱条件下的潜变量 $z$ 的先验分布 $p(z|c)$。CVAE 采用重构目标 $L_{\text{recon}}$ 和先验正则项：
$$L_{cvae}=L_{recon}+D_{KL}(q(z|y)||p(z|c))+L_{ctc},$$
其中 $D_{KL}$ 是 KL 散度，$L_{ctc}$ 是 CTC 损失。重构损失用 mel-spectrum 的 L1 距离。

### Posterior Encoder

posterior encoder 将波形 $y$ 编码为潜变量 $z$。首先得到线性谱，然后用几个 WaveNet 残差块提取 hidden vector，通过线性投影得到后验分布 $p(z|y)$ 的均值和方差。最后用重参数化技巧从 $p(z|y)$ 中采样得到潜变量 $z$。

### Decoder

decoder 根据中间表征 $z$ 生成音频波形。用 GAN 训练提高重构语音的质量。判别器 $D$ 采用 HiFiGAN 的 Multi-Period Discriminator (MPD) 和 Multi-Scale Discriminator (MSD)。生成器 $G$ 和判别器 $D$ 的 GAN 损失定义为：
$$\begin{gathered}L_{adv}(G)=\mathbb{E}_{(z)}\left[(D(G(z))-1)^2\right]\\L_{adv}(D)=\mathbb{E}_{(y,z)}\left[\left(D(y)-1\right)^2+\left(D(G(z))\right)^2\right]\end{gathered}$$
此外，用特征匹配损失 $L_{fm}$ 作为额外损失，减小每个判别器中间层的特征图的 L1 距离。

### Prior Encoder

给定乐谱 $c$，prior encoder 提供建模先验分布 $p(z|c)$。

text encoder 输入为乐谱，产生 phoneme-level 表征。然后用 [FastSpeech 2- Fast and High-Quality End-to-End Text to Speech 笔记](../FastSpeech%202-%20Fast%20and%20High-Quality%20End-to-End%20Text%20to%20Speech%20笔记.md) 中的 Length Regulator 将 phoneme-level 表征扩展到 frame-level 表征 $h_{\text{text}}$。由于歌声中声学变化更明显，不同帧可能服从不同分布，加入 frame prior network 生成细粒度的先验正态分布，均值为 $\mu_{\theta}$，方差为 $\sigma_{\theta}$。为了提高先验分布的表现力，引入 normalizing flow $f_{\theta}$ 和 phoneme predictor。phoneme predictor 由两层 FFT 组成，其输出计算 CTC 损失：
$$p(z|c)=N(f_\theta(z);\mu_\theta(c),\sigma_\theta(c)))|det\frac{\partial f_\theta(z)}{\partial_z}|$$

歌曲的乐谱主要包括歌词、音符（note）持续时间和音符音高。首先将歌词转换为 phoneme 序列。音符持续时间是每个音符对应的帧数，音符音高按照 MIDI 标准转换为 Pitch ID。音符持续时间序列和音符音高序列扩展到 phoneme 序列的长度。text encoder 由多个 FFT 块组成，输入为上述三个序列，输出乐谱的 phoneme-level 表征。

由于歌声中每个 phoneme 的发音比较复杂，加入 Length Regulator (LR) 模块将 phoneme-level 表征扩展到 frame-level 表征 $h_{\text{text}}$。训练时，用每个 phoneme 对应的真实持续时间 $d$ 进行扩展，预测持续时间 $\hat{d}$ 用于合成。
> 注意：歌声合成中的 note duration 通常是已知的标签（但不知道 phoneme duration），而语音合成中通常没有 duration 标签。

duration predictor 由多个一维卷积层组成。因为乐谱中的持续时间定义了歌声的整体节奏和速度，所以不使用随机持续时间预测器。

音符持续时间（注意不是 phoneme duration）包含了 duration prediction 的先验，所以基于音符持续时间进行 duration prediction。phoneme duration 与对应的  note duration 的比值定义为 $r$。此时 duration loss 为：
$$\begin{aligned}L_{dur}&=\left\|r\times d_N-d\right\|_2\\\hat{d}&=r\times d_N\end{aligned}$$
$r$ 与 $d_N$ 的乘积是预测的帧数 $d$。

在 VITS 模型的训练过程中，text encoder 提取 phoneme-level 文本信息，作为潜变量 z 的先验知识。但是在歌声合成任务中，每个 phoneme 的声学序列变化非常丰富，所以仅用 phoneme 的均值和方差来表示对应的帧序列是不够的。于是向模型中添加 frame prior network，包含多个一维卷积层。
> 而且发现简单地增加 flow model 的层数不能实现和添加 frame prior network 模块相同的效果。

frame prior network 对 frame-level 序列进行后处理，得到 frame-level 均值 $\mu_{\theta}$ 和方差 $\sigma_{\theta}$。

还引入 F0 信息来进一步引导 frame prior network。F0 由 F0 predictor 得到，包含多个 FFT 块。LF0 损失为：
$$\begin{aligned}L_{LF0}=\left\|L\hat{F}0-LF0\right\|_2\end{aligned}$$

flow decoder 用的就是 VITS 中的。

最后总损失为：
$$\begin{aligned}L=&L_{adv}(G)+L_{fm}(G)+L_{cvae}+\lambda L_{dur}+\beta L_{LF0}\\&L(D)=L_{adv}(D)\end{aligned}$$

其中 $L_{adv}(G)$ 和 $L_{adv}(D)$ 分别是 G 和 D 的 GAN 损失，feature matching loss $L_{fm}$ 用于提高训练的稳定性。CVAE 损失 $L_{cvae}$ 包含重构损失、KL 损失和 CTC 损失。

## 实验（略）
