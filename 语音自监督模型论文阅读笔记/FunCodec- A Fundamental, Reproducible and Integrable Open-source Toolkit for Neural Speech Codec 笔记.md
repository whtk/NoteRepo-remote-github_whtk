> ICASSP 2024，阿里云

1. 提出 FunCodec，一个 fundamental 的 codec 工具包，为 FunASR 的拓展：
    1. 提供以后 codec 模型的训练和推理脚本，如 SoundStream 和 Encodec
    2. 可以集成到下游任务中
    3. 提供预训练模型
2. 基于 toolkit，提出了 frequency-domain codec 模型 FreqCodec，可以在更低的计算复杂度和参数复杂度下实现相当的合成质量

## Introduction

1. 传统 speech codecs 依赖于专家知识，利用心理声学和语音合成来实现高效编码
2. 基于深度学习技术，neural speech codecs 被引入，表现优于传统 speech codecs
    1. raw waveforms 通过深度编码器提取表征
    2. 使用 RVQ 获取并行 token 流
    3. 训练解码器重构信号
    4. 使用对抗训练提高重构质量
    5. 两种流行的 neural codec 模型：SoundStream 和 Encodec
        1. SoundStream 使用流式 SEANet 作为 encoder 和 decoder
        2. Encodec 包含额外的 LSTM 层和基于 Transformer 的模型来提高序列建模能力
3. neural speech codecs 也可以用于生成模型中提取离散表征
    1. 在 VALL-E 中，text 和 speech tokens 连成序列，训练语言模型估计概率
    2. neural speech codecs 促进 speech 和 text 的建模，使模型能够听和说
4. 提出 FunCodec，一个 fundamental、reproducible 和 integrable 的开源工具包
    1. 可以构建、训练和评估各种 neural speech codecs
    2. 提供了 finetune 预训练模型或从头训练模型的 recipe
    3. 提出了 frequency-domain codec 模型 FreqCodec，可以在更少的参数和更低的计算复杂度下实现相当的性能
    4. 评估了语义信息对 speech codec 的影响，可以在低比特率下提高 speech 质量
    5. 通过 Huggingface 和 ModelScope 发布了预训练的学术和通用模型
    6. 提供了推理和评估脚本，支持批处理模式以充分利用 GPU 的并行能力

## 相关工作

和以下几个开源的库比：
![](image/Pasted%20image%2020240528223717.png)

+ Encodec
+ EncTrainer
+ Dac
+ AudioDec

FunCodec 和其他工具包的区别：
+ FunCodec 有七个模型，数量更多
+ FunCodec 的 recipe 只要一个训练 stage
+ FunCodec 包含多个 discriminator 来提高语音质量
+ 支持多 GPU 分布式训练
+ 可以流式地同时产生多个 token
+ 可以允许 codebook 的 k-means 初始化

## FunCodec

FunCodec 包含：
+ 神经网络库：用 pytorch 写的
+ recipes：Kaldi-style

### 模型架构

架构如图：
![](image/Pasted%20image%2020240528224221.png)

给定语音信号 $x$，首先通过 domain transformation 模块。对于时域模型，此模块为恒等映射；而对于频域模型，会得到两个表征：
$$\begin{aligned}
\mathbf{X}& =\mathrm{STFT}(x)  \\
X_{\mathrm{mag,ang}}& =\log\left(|X|\right),\mathrm{angle}(X_i,X_r)  \\
X_{\mathrm{mag,pha}}& =\log\left(\left|X\right|\right),\frac{X_r}{\left|X\right|},\frac{X_i}{\left|X\right|} 
\end{aligned}$$
其中 $X_r$ 和 $X_i$ 分别表示复数谱的实部和虚部，$| \cdot |$ 表示复数膜长。经过 domain transformation 模块后，语音输入到 encoder 提取 acoustic representations：$V_a = \mathrm{Encoder}(X)$。对于时域模型，用的是和 Encodec 和 SoundStream 相同的 SEANet 架构。对于频域模型（FreqCodec），encoder 结构如下表：
![](image/Pasted%20image%2020240528225525.png)
decoder 和 encoder 镜像。

最后，domain inversion 模块用于从 decoder 输出重构原始波形。

### 语义增强的 RVQ

为了得到离散 speech tokens，使用 RVQ 模块，其包含多个量化器：
$$Q_n=\text{VQ}\left(Q_0-\sum_{i=1}^{n-1}Q_i\right)$$
其中 $Q_n$ 表示第 $n$ 个 VQ 的输出，$Q_0$ 表示 RVQ 的输入。为了提高对 code 的利用，使用 k-means 初始化 VQ codebook。然后使用滑动平均更新 code，如果一个 code 在 mini-batch 中激活次数少于两次，会被重新分配。

除了 encoder 输出，还有三种方法将语义信息融入 codec 模型：
$$\begin{aligned}
&f_{\mathrm{cat}}(V_a,V_s) =\mathrm{Concat}(\mathrm{RVQ}(V_a),V_s)  \\
&f_{\mathrm{add}}(V_a,V_s) =\mathrm{RVQ}(V_a)+V_s  \\
&f_{\mathrm{res}}(V_a,V_s) =\mathrm{RVQ}(V_a-V_s)+V_s 
\end{aligned}$$
其中 $V_s$ 表示 semantic token，包括 frame-aligned phoneme labels 和 Hubert embeddings。

### 基于多个 discriminator 的对抗训练

训练目标包括三个部分：重构损失、对抗损失和 RVQ commit 损失。时域上的 L1 距离和频域上的 L1 和 L2 距离：
$$\begin{aligned}\mathcal{L}_f(x,\hat{x})&=\frac1{|\alpha|}\sum_{i\in\alpha}(||\mathcal{S}_i(x)-\mathcal{S}_i(\hat{x})||_1+||\mathcal{S}_i(x)-\mathcal{S}_i(\hat{x})||_2\\&+||\mathcal{M}_i(x)-\mathcal{M}_i(\hat{x})||_1+||\mathcal{M}_i(x)-\mathcal{M}_i(\hat{x})||_2)\end{aligned}$$
其中 $\mathcal{S}_i$ 和 $\mathcal{M}_i$ 分别表示 log-compressed power 和 Mel 谱，窗口大小为 $2^i$，移动长度为 $2^i/4$。$\alpha$ 取 $[5,6,\ldots,11]$。

对于对抗损失，FunCodec 包含多个 discriminator，包括 multi-scale discriminator (MSD)、multi-period discriminator (MPD) 和 multi-scale STFT-based (MSTFTD) discriminator。通过提供统一接口，FunCodec 允许这些 discriminator 的各种组合，提高了判别能力。此外，还有一个“feature” matching loss：
$$\mathcal{L}_{\mathrm{adv}}(\hat{x})=\mathbb{E}_{\hat{x}}\left[\frac1K\sum_{k,t}\frac1{T_k}\mathrm{max}(0,1-\mathcal{D}_{k,t}(\hat{x}))\right]\\\mathcal{L}_{\mathrm{feat}}(x,\hat{x})=\mathbb{E}_{x,\hat{x}}\left[\frac1{KL}\sum_{k,t,l}\frac1{T_k}\left|\left|\mathcal{D}_{k,t}^{(l)}(x)-\mathcal{D}_{k,t}^{(l)}(\hat{x})\right|\right|_1\right]$$
其中 $\mathcal{D}_{k,t}$ 表示时间步 $t$ 的 discriminator $k$ 的输出，$\mathcal{D}^{(l)}$ 表示第 $l$ 层的输出。

commit loss 包括整个 RVQ 模块和子量化器的量化误差：
$$\mathcal{L}_{\mathrm{cm}}=\left|\left|V-\mathrm{RVQ}(V)\right|\right|_2+\frac1N\sum_{i=1}^N\left|\left|Q_{i-1}-\mathrm{VQ}_i(Q_{i-1})\right|\right|_2$$

其中 $V$ 表示 RVQ 的输入。总训练目标为：
$$\mathcal{L}=\lambda_t\mathcal{L}_t+\lambda_f\mathcal{L}_f+\lambda_\mathrm{adv}\mathcal{L}_\mathrm{adv}+\lambda_\mathrm{feat}\mathcal{L}_\mathrm{feat}+\lambda_\mathrm{cm}\mathcal{L}_\mathrm{cm}$$

## 实验设置

在两种情况下进行实验：
+ academic：使用 LibriTTS 训练模型，在多个开源数据集的测试集上评估，包括 Librispeech、aishell-1、aishell-2、Wenet 和 Gigaspeech。将所有 utterances 重采样到 16k Hz
+ generalized：使用一个大规模的内部数据集，包含 27.68M 双语（英语和普通话）utterances，总时长约 25,000 小时

使用 ViSQOL 作为主要评估指标，范围为 1 到 5，分数越高表示质量越好。引入 token ratio (TKR)，表示模型需要多少 token 来表示一秒的 speech。在相同的 TKR 下，ViSQOL 分数更高的模型更好。

训练时，随机裁剪一个连续的 3.2 秒片段作为训练样本。输入 encoder 之前进行 RMS 标准化。
+ 对于 LibriTTS，在两台 Tesla-V100 GPU 上训练，batch size 为 32
+ generalized 模型在四台 Tesla-A100 GPU 上训练，batch size 为 128

在对抗训练下，更新 codec 模型 300,000 次。为了防止 discriminator 太强，只有当其损失超过 codec 模型时才更新。

超参数 $\lambda_t , \lambda_f , \lambda_{\mathrm{adv}} , \lambda_{\mathrm{feat}}$ 和 $\lambda_{\mathrm{cm}}$ 分别设置为 1.0, 1.0, 1/9, 100/9 和 1.0。对于 FreqCodec 模型，speech segment 首先使用窗口大小为 512 和移动大小为 160 的 STFT 转换为 spectrogram。

## 结果

academic 模型结果如下表：
![](image/Pasted%20image%2020240530221556.png)
在 LibriTTS 上，本文复现的 SoundStream 和 Encodec 模型在不同 token rates 下都达到了 SOTA 的性能。log-compressed power spectrum loss 的加入可以提高合成质量。

基于 FunCodec，提出了 stride 为两倍和四倍的 low-frame-rate 模型，FunCodec-2x 和 FunCodec-4x，发现：
+ FunCodec-2x 在 ViSQOL 分数上更高
+ FunCodec-4x 在较低的 token rates 下表现较差
说明：将 frame rate 减半，可以在时间和量化分辨率之间实现平衡

下图为不同 token rates 下的不同开源模型结果：
![](image/Pasted%20image%2020240530221804.png)
发现：
+ token rates 越高，压缩质量越好
+ 与其他模型相比，本文模型在相同 token rates 下，英语和普通话的质量都更好
+ 所有模型在 Wenet 测试集上表现不好，可能是因为 Wenet 语料库在复杂的声学环境中录制，包含更多的非语音噪声

下表为频域和时域模型的比较：
![](image/Pasted%20image%2020240530224459.png)
结论：
+ FreqCodec 在较高的 token rates 下表现更好
+ 使用 depthwise convolutions 可以减少参数数量和计算复杂度
+ M4-M7 的结果表明，通过适当地分组 encoder 和 decoder，可以显著减少模型和计算复杂度，而不会影响质量


semantic 增强的效果：
![](image/Pasted%20image%2020240530224622.png)
结果表明，语义 token 的加入可以提高 speech 质量。"residual" 组合方法对 speech codec 更适用。


除了语音质量，还在下游任务中评估 FunCodec。ASR 任务结果如下：
![](image/Pasted%20image%2020240530224704.png)
由于 quantized tokens 保留了大量的 content，从而识别错误率低。codebook embedding 对 ASR 也很重要。通过比较 clean 和其他测试的结果，发现 codec-based 的离散输入相比于 fbank 特征对声学环境更敏感。