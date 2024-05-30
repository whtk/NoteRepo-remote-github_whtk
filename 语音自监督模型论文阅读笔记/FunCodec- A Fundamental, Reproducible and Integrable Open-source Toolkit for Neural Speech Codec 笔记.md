> ICASSP 2024，阿里云
<!-- 翻译 & 理解 -->
<!-- This paper presents FunCodec, a fundamental neural speech codec toolkit, which is an extension of the open-source speech processing toolkit FunASR. FunCodec provides reproducible training recipes and inference scripts for the latest neural speech codec models, such as SoundStream and Encodec. Thanks to the unified design with FunASR, FunCodec can be easily integrated into downstream tasks, such as speech recognition. Along with FunCodec, pre- trained models are also provided, which can be used for academic or generalized purposes. Based on the toolkit, we further pro- pose the frequency-domain codec models, FreqCodec, which can achieve comparable speech quality with much lower computation and parameter complexity. Experimental results show that, under the same compression ratio, FunCodec can achieve better recon- struction quality compared with other toolkits and released models. We also demonstrate that the pre-trained models are suitable for downstream tasks, including automatic speech recognition and personalized text-to-speech synthesis. This toolkit is publicly avail- able at https://github.com/alibaba-damo-academy/ FunCodec. -->
1. 提出 FunCodec，一个 fundamental 的 codec 工具包，为 FunASR 的拓展：
    1. 提供以后 codec 模型的训练和推理脚本，如 SoundStream 和 Encodec
    2. 可以集成到下游任务中
    3. 提供预训练模型
2. 基于 toolkit，提出了 frequency-domain codec 模型 FreqCodec，可以在更低的计算复杂度和参数复杂度下实现相当的合成质量

## Introduction
<!-- Speech codecs are designed to compress and decompress speech sig- nals for efficient transmission and storage. They consist of an en- coder, which encodes speech into a compact representation, and a decoder to reconstruct the signal. Traditional speech codecs rely on a carefully designed pipeline that incorporates expert knowledge of psycho-acoustics and speech synthesis to achieve efficient cod- ing [1, 2]. -->
<!-- Thanks to advancements in deep learning techniques, neural speech codecs have been introduced, demonstrating superior per- formance compared to traditional speech codecs. In neural codec models, raw waveforms are fed into deep neural network-based encoders to extract compact representations. This is followed by a residual vector quantizer (RVQ) [3, 4] to obtain parallel token streams. Meanwhile, a neural network-based decoder is also trained alongside the encoder and RVQ to reconstruct the signal. Building upon the progress in text-to-speech synthesis [5], adversarial train- ing losses are employed to enhance reconstruction quality. There are two popular neural codec models, SounStream [4] and Encodec [6]. While SoundStream utilizes streaming SEANet [7, 8] as its encoder and decoder, Encodec incorporates extra LSTM [9] layers and a Transformer-based language model [10] to improve the sequence modeling ability. Following this line of work, extensive efforts have been made on reducing the bit rate [11–13]. Additionally, the modified discrete cosine transform (MDCT) domain has also been explored [14, 15]. -->
1. 传统 speech codecs 依赖于专家知识，利用心理声学和语音合成来实现高效编码
2. 基于深度学习技术，neural speech codecs 被引入，表现优于传统 speech codecs
    1. raw waveforms 通过深度编码器提取表征
    2. 使用 RVQ 获取并行 token 流
    3. 训练解码器重构信号
    4. 使用对抗训练提高重构质量
    5. 两种流行的 neural codec 模型：SoundStream 和 Encodec
        1. SoundStream 使用流式 SEANet 作为 encoder 和 decoder
        2. Encodec 包含额外的 LSTM 层和基于 Transformer 的模型来提高序列建模能力
<!-- Although neural speech codecs were originally proposed to compress signals for telecommunication, they can also be used to extract discrete speech representations in generative models. In VALL-E [16], text and speech tokens are joined into a sequence, and a language model is trained to estimate their probabilities. This formulation demonstrates impressive zero-shot synthesis capability. Moreover, neural speech codecs facilitate the modeling of speech and text within a single framework, enabling the model to both listen and speak [17,18]. Recently, several speech codec toolkits have been released for telecommunications purposes [19, 20]. However, there is still a lack of open-source toolkits that provide a reproducible and integrable framework for developing and evaluating neural speech codecs in the context of speech-text modeling. -->
3. neural speech codecs 也可以用于生成模型中提取离散表征
    1. 在 VALL-E 中，text 和 speech tokens 连成序列，训练语言模型估计概率
    2. neural speech codecs 促进 speech 和 text 的建模，使模型能够听和说
<!-- To address this gap, we present FunCodec, a fundamental, reproducible, and integrable open-source toolkit for neural speech codecs. FunCodec provides a versatile platform enabling researchers to build, train, and evaluate various neural speech codecs. Fig.1 shows an overview of the FunCodec design. The contributions of FunCodec are as follows: (1) The open-source codebase provides recipes to finetune pre-trained models or train a model from scratch. (2) Frequency-domain codec (FreqCodec) models are proposed, which can achieve comparable performance with less parameters and lower computation complexity. (3) The impact of semantic in- formation is evaluated for speech codec, which improves the speech quality under low bit rate. (4) Pre-trained academic and generalized models are released through Huggingface and ModelScope 1. (5) Inference and evaluation scripts are also provided, which support batch mode to fully utilize the parallelism capability of GPUs. -->
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

<!-- Table 1 summaries the differences between FunCodec and these toolkits. While the other toolkits offer a limited number of pre- trained models, FunCodec provides seven models for both academic and generalized purposes. This allows researchers to use them as a baseline system and also enables general users to directly apply them to downstream tasks. Additionally, FunCodec provides comprehen- sive and efficient recipes that require only a single training stage. To enhance the speech quality, FunCodec incorporates various discrimi- nators, including multiple scale discriminator (MSD) [3, 7], multiple period discriminator (MPD) [5], multiple short-time Fourier trans- formation discriminator (MSTFTD) [4], and their combinations. For training efficiency, FunCodec supports distributed training across multiple GPUs. Moreover, FunCodec ensures high inference effi- ciency by simultaneously producing token streams for all samples in a mini-batch. Furthermore, FunCodec enables k-means initialization for quantization codebooks, improving the code utilization [22, 23]. Based on these features, FunCodec introduces low-frame-rate mod- els. The frequency-domain transformation and semantic augmenta- tion are also explored in FunCodec. -->
FunCodec 和其他工具包的区别：
+ FunCodec 有七个模型，数量更多
+ FunCodec 的 recipe 只要一个训练 stage
+ FunCodec 包含多个 discriminator 来提高语音质量
+ 支持多 GPU 分布式训练
+ 可以流式地同时产生多个 token
+ 可以允许 codebook 的 k-means 初始化

## FunCodec
<!-- FunCodec codebase consists of two main components: a library of neural network models and recipes for replicating the experiments. The library part is written in python with PyTorch [24]. The recipes are all-in-one Bash scripts written in the Kaldi-style [25]. -->
FunCodec 包含：
+ 神经网络库：用 pytorch 写的
+ recipes：Kaldi-style

### 模型架构

架构如图：
![](image/Pasted%20image%2020240528224221.png)
<!-- The architecture of FunCodec models is depicted in Fig. 2. Given a speech signal x, it is passed through the domain transformation mod- ule. For time-domain models like SoundStream and Encodec, the module functions as an identity mapping. However, for frequency- domain models, two representations Xmag,ang and Xmag,pha are ex- plored: -->
给定语音信号 $x$，首先通过 domain transformation 模块。对于时域模型，此模块为恒等映射；而对于频域模型，会得到两个表征：
$$\begin{aligned}
\mathbf{X}& =\mathrm{STFT}(x)  \\
X_{\mathrm{mag,ang}}& =\log\left(|X|\right),\mathrm{angle}(X_i,X_r)  \\
X_{\mathrm{mag,pha}}& =\log\left(\left|X\right|\right),\frac{X_r}{\left|X\right|},\frac{X_i}{\left|X\right|} 
\end{aligned}$$
<!-- where Xr , Xi denote the real and imaginary parts of complex spec-trum, respectively. | · | represents the norm of a complex value. After the domain transformation module, the speech is inputted into an encoder to extract acoustic representations: Va = Encoder(X). For time-domain models, we adopt the same SEANet architectures as Encodec and SoundStream. In the case of frequency-domain models (FreqCodec), the encoder details are given in Table 2. The decoder has a mirror architecture of the encoder. More details can be found in our released codebase. Finally, a domain inversion module is uti- lized to reconstruct the raw waveforms from decoder outputs. -->
其中 $X_r$ 和 $X_i$ 分别表示复数谱的实部和虚部，$| \cdot |$ 表示复数膜长。经过 domain transformation 模块后，语音输入到 encoder 提取 acoustic representations：$V_a = \mathrm{Encoder}(X)$。对于时域模型，用的是和 Encodec 和 SoundStream 相同的 SEANet 架构。对于频域模型（FreqCodec），encoder 结构如下表：
![](image/Pasted%20image%2020240528225525.png)

decoder 和 encoder 镜像。

最后，domain inversion 模块用于从 decoder 输出重构原始波形。

<!-- Semantic-augmentedresidualvectorquantization -->
### 语义增强的 RVQ
<!-- To obtain discrete speech tokens, we employ a residual vector quan- tization (RVQ) module consisting of several quantizers: -->
为了得到离散 speech tokens，使用了一个 RVQ 模块，包含多个量化器：
$$Q_n=\text{VQ}\left(Q_0-\sum_{i=1}^{n-1}Q_i\right)$$
<!-- where Qn represents the outputs of n-th vector quantizer (VQ) and Q0 represents the input of RVQ. To improve code utilization, we ini- tialize the VQ codebook by clustering the samples in the first mini- batch with k-means. The codes are then updated using a moving average with a decay rate of 0.99. Moreover, if a code is activated fewer than two times in a mini-batch, it will be reassigned. -->
其中 $Q_n$ 表示第 $n$ 个 VQ 的输出，$Q_0$ 表示 RVQ 的输入。为了提高对 code 的利用，使用 k-means 初始化 VQ codebook。然后使用滑动平均更新 code，如果一个 code 在 mini-batch 中激活次数少于两次，会被重新分配。

<!-- In addition to the encoder outputs, we explore three methods to incorporate semantic information into the codec models: -->
除了 encoder 输出，还有三种方法将语义信息融入 codec 模型：
$$\begin{aligned}
&f_{\mathrm{cat}}(V_a,V_s) =\mathrm{Concat}(\mathrm{RVQ}(V_a),V_s)  \\
&f_{\mathrm{add}}(V_a,V_s) =\mathrm{RVQ}(V_a)+V_s  \\
&f_{\mathrm{res}}(V_a,V_s) =\mathrm{RVQ}(V_a-V_s)+V_s 
\end{aligned}$$
<!-- where Vs denotes semantic tokens, such as frame-aligned phoneme labels and Hubert embeddings [26]. To make a single model oper- ate across variable bitrates, structured quantization dropout is also implemented in FunCodec. -->
其中 $V_s$ 表示 semantic token，包括 frame-aligned phoneme labels 和 Hubert embeddings。

<!-- Adversarialtrainingobjectivewithmultiplediscriminators -->
### 基于多个 discriminator 的对抗训练
<!-- The training objective consists of three components: reconstruction loss terms, adversarial loss terms, and the RVQ commit losses. The L1 distance between original x and reconstructed speech xˆ is mini- mized over time domain: Lt (x, xˆ) = ||x − xˆ||1 . For the frequency domain, both L1 and L2 distances are minimized on multiple Mel and magnitude spectra: -->
训练目标包括三个部分：重构损失、对抗损失和 RVQ commit 损失。时域上的 L1 距离和频域上的 L1 和 L2 距离：
$$\begin{aligned}\mathcal{L}_f(x,\hat{x})&=\frac1{|\alpha|}\sum_{i\in\alpha}(||\mathcal{S}_i(x)-\mathcal{S}_i(\hat{x})||_1+||\mathcal{S}_i(x)-\mathcal{S}_i(\hat{x})||_2\\&+||\mathcal{M}_i(x)-\mathcal{M}_i(\hat{x})||_1+||\mathcal{M}_i(x)-\mathcal{M}_i(\hat{x})||_2)\end{aligned}$$
<!-- where, Si and Mi represent the log-compressed power and Mel spectra with a window size of 2i and a shift length of 2i/4. α is set to [5, 6, . . . , 11]. It worth noting that the log-compressed power spectrum loss improves speech quality in the middle and high fre- quencies, which is missed in other toolkits and models. -->
其中 $\mathcal{S}_i$ 和 $\mathcal{M}_i$ 分别表示 log-compressed power 和 Mel 谱，窗口大小为 $2^i$，移动长度为 $2^i/4$。$\alpha$ 取 $[5,6,\ldots,11]$。