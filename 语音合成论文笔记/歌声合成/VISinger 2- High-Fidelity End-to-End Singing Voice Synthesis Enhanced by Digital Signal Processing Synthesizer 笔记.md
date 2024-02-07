> Interspeech 2023，西工大、ASLP

1. VISinger 可以在更少的参数下实现很好的性能，但是有一些问题：
    1. text-to-phase：端模型学习了无意义的文本到相位的映射
    2. glitches：对应于周期信号的谐波分量突然变化，产生伪影
    3. 低采样率：24KHz 的采样率不能满足高保真生成的应用需求
2. 提出 VISinger 2：
    1. 将 DSP synthesizer 和 decoder 结合
    2. DSP synthesizer 由 harmonic synthesizer 和 noise synthesizer 组成，从潜变量 $z$ 中生成周期和非周期信号，监督 posterior encoder 提取不带相位信息的潜变量，避免 prior encoder 建模文本到相位的映射
    3. 修改 HiFiGAN，接受 DSP synthesizer 生成的波形作为条件生成歌声

## Introduction

1. VISinger 有一些缺点：
    1. 仍然存在伪影，如频谱不连续和偶尔的发音错误，会降低歌声的自然度
    2. 生成的歌声采样率为 24KHz，不能满足高保真（HiFi）应用的需求
1. 重新分析 VISinger：
    1. posterior encoder 提取的潜变量 $z$ 可能包含相位信息，因为 decoder 传回的梯度可能会导致这种情况，这可能导致发音错误
    2. HiFiGAN 不适合 SVS 任务，不能很好地建模歌声中的变化，可能导致伪影
    3. 更高的采样率需要更好的 decoder
2. 提出 VISinger 2：
    1. 将 DSP synthesizer 加入 VISinger
    2. DSP synthesizer 由 harmonic synthesizer 和 noise synthesizer 组成，分别从潜变量 $z$ 生成周期和非周期信号
    3. 两种信号拼接起来作为 HiFiGAN 的条件，两者的和作为波形计算损失函数
    4. 优势：
        1. 两种 synthesizer 只需要振幅信息作为输入，完全压缩了 $z$ 中的相位信息，避免了文本到相位的挑战
        2. 周期和非周期信号可以增强 HiFi-GAN 建模能力，实现更高的采样率
        3. 参数减少了 30%

## 方法

结构如图：
![](image/Pasted%20image%2020240207164125.png)
包含 posterior encoder、prior encoder 和 decoder，和 [VISinger- Variational Inference with Adversarial Learning for End-to-End Singing Voice Synthesis](VISinger-%20Variational%20Inference%20with%20Adversarial%20Learning%20for%20End-to-End%20Singing%20Voice%20Synthesis.md) 一样。

### Posterior Encoder

posterior encoder 由多层 1D 卷积层组成，从 mel-spectrum 中提取 $z$。最后一层产生后验分布的均值和方差，用重参数化方法得到后验 $z$。

### Decoder

decoder 从 $z$ 生成波形。将 DSP synthesizer 加入 decoder。使用 harmonic synthesizer 和 noise synthesizer 从 posterior $z$ 生成周期和非周期部分的波形。生成的波形作为 HiFi-GAN 的辅助条件输入。

两个 synthesizer 的输入只包含振幅信息，从而 posterior $z$ 不包含相位信息。

#### Harmonic Synthesizer

使用 harmonic synthesizer 生成音频的谐波分量，使用 sin 信号模拟单声源音频的每个共振峰的波形。第 $k$ 个正弦分量信号 $y_k$ 可以表示为：
$$\begin{aligned}y_k(n)=H_k(n)sin(\phi_k(n))\end{aligned}$$
其中 $n$ 表示 time step，$H_k$ 是第 $k$ 个正弦分量的时变振幅。相位 $\phi_k(n)$ 通过对样本序列进行积分得到：
$$\begin{aligned}\phi_k(n)=2\pi\sum_{m=0}^n\frac{f_k(m)}{Sr}+\phi_{0,k}\end{aligned}$$
其中 $f_k$ 表示第 $k$ 个正弦分量的频率，$Sr$ 表示采样率，$\phi_{0,k}$ 表示初始相位。通过累积操作得到 sin 信号 $y_k$ 的相位。频率 $f_k$ 可以通过 $f_k(n)=kf_0(n)$ 计算，其中 $f_0$ 是基频。时变的 $f_k$ 和 $H_k$ 从 frame-level 特征插值得到。使用 Harvest 算法提取基频。

#### Noise Synthesizer

noise synthesizer 中，使用傅里叶逆变换（iSTFT）生成音频的随机分量。非周期分量更接近噪声，但是能量分布在不同频段不均匀。随机分量信号 $y_{\text{noise}}$ 可以表示为：
$$\begin{aligned}y_{noise}=iSTFT(N,P)\end{aligned}$$
iSTFT 的相位谱 $P$ 是在区间 $[-\pi,\pi]$ 的均匀噪声，幅度谱 $N$ 由网络预测。

#### Decoder 损失函数

DSP synthesizer 生成的波形包含谐波和随机分量。DSP 波形 $y_{\text{DSP}}$ 和 DSP synthesizer 的损失 $L_{{DSP}}$ 定义为：
$$\begin{aligned}y_{DSP}&=\sum_{k=0}^Ky_k+y_{noise}\\L_{DSP}&=\lambda_{DSP}|\text{Mel}(y_{DSP})-\text{Mel}(y)|_1\end{aligned}$$

其中 $K$ 表示正弦分量的数量，Mel 表示从波形中提取 mel-spectrum 的过程。

使用下采样网络逐渐将 DSP 波形下采样到 frame-level 特征。HiFi-GAN 输入为 posterior $z$ 和下采样网络生成的中间特征，生成最终波形 $\hat{y}$。HiFi-GAN 的生成器 $G$ 的 GAN 损失定义为：
$$\begin{aligned}L_G=L_{adv}(G)+\lambda_{fm}L_{fm}+\lambda_{Mel}L_{Mel}\end{aligned}$$

其中 $L_{adv}$ 是对抗损失，$L_{fm}$ 是特征匹配损失，$L_{Mel}$ 是 Mel-Spectrogram 损失。

#### Discriminator

用了两组判别器：
+ UnviNet 中的多分辨率频谱判别器（MRSD）
+ HiFi-GAN 中的 Multi-Period Discriminator (MPD) 和 Multi-Scale Discriminator (MSD)

### Prior Encoder

prior encoder 输入为乐谱。前面 posterior $z$ 用于预测 decoder 中的 $H$ 和 $N$，其中 $H$ 表示正弦共振的振幅，$N$ 表示非周期分量的振幅。$H$ 和 $N$ 只包含振幅信息，不包含相位信息，所以 posterior $z$ 也不包含相位信息。因此，prior encoder 不会在基于乐谱预测 posterior $z$ 时建模文本到相位的映射。

prior encoder 采用和 Fastspeech 相同的结构。VITS 中的 flow 模块占用了大量的模型参数，所以直接计算先验 $z$ 和后验 $z$ 之间的 KL 散度 $L_{kl}$。

使用单独的 FastSpeech 模型预测基频和 mel-spectrum，引导 frame-level 先验网络。辅助特征的损失定义为：
$$\begin{aligned}L_{af}=|LF0-\widehat{LF0}|_2+|Mel-\widehat{Mel}|_1\end{aligned}$$

其中 $\widehat{LF0}$ 是预测的 log-F0，$\widehat{Mel}$ 是预测的 mel-spectrogram。

在训练和推理过程中，将预测的 mel-spectrum 作为 frame-level prior network 的辅助特征。frame-level prior network 在辅助 mel-spectrum 的引导下预测先验 $z$，进一步减轻文本到相位的问题。

在推理过程中，harmonic synthesizer 输入为预测的基频，在训练过程中采用真实的基频。

duration predictor 输入为乐谱，同时预测 phoneme duration 和 note duration。duration loss 表示为：
$$L_{dur}=|d_{phone}-\widehat{d_{phone}}|_2+|d_{note}-\widehat{d_{note}}|_2$$

其中 $d_{phone}$ 是真实的 phoneme duration，$\widehat{d_{phone}}$ 是预测的 phoneme duration，$d_{note}$ 是真实的 note duration，$\widehat{d_{note}}$ 是预测的 note duration。

### 总损失

最终的目标可以表示为：
$$\begin{gathered}L(G)=L_G+L_{kl}+L_{DSP}+L_{dur}+L_{af}\\L(D)=L_{adv}(D)\end{gathered}$$

其中 $L_G$ 是生成器 G 的 GAN 损失，$L_{kl}$ 是先验 $z$ 和后验 $z$ 之间的 KL 散度，$L_{af}$ 是辅助特征的损失，$L_{adv}(D)$ 是判别器 D 的 GAN 损失。

## 实验（略）
