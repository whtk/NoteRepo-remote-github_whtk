> NIPS 2023，Descript, Inc
<!-- 翻译 & 理解 -->
<!-- Language models have been successfully used to model natural signals, such as images, speech, and music. A key component of these models is a high quality neural compression model that can compress high-dimensional natural signals into lower dimensional discrete tokens. To that end, we introduce a high-fidelity universal neural audio compression algorithm that achieves 90x compression of 44.1 KHz audio into tokens at just 8kbps bandwidth. We achieve this by combining advances in high-fidelity audio generation with better vector quantization tech- niques from the image domain, along with improved adversarial and reconstruction losses. We compress all domains (speech, environment, music, etc.) with a single universal model, making it widely applicable to generative modeling of all audio. We compare with competing audio compression algorithms, and find our method outperforms them significantly. We provide thorough ablations for every design choice, as well as open-source code and trained model weights. We hope our work can lay the foundation for the next generation of high-fidelity audio modeling. -->
1. 提出一种高保真、通用的神经音频压缩算法，将 44.1 KHz 音频压缩到 8kbps 带宽的离散 token 中，实现 90 倍压缩：
    1. 用的是图像里面的量化技术
    2. 还有对抗和重构损失
2. 用一个通用模型压缩所有音频（语音、环境、音乐等）

## Introduction
<!-- Generative modeling of high-resolution audio is difficult due to high dimensionality (~44,100 samples per second of audio) [24, 19], and presence of structure at different time-scales with both short and long-term dependencies. To mitigate this problem, audio generation is typically divided into two stages: 1) predicting audio conditioned on some intermediate representation such as mel-spectrograms [24, 28, 19, 30] and 2) predicting the intermediate representation given some conditioning information, such as text [35, 34]. This can be interpreted as a hierarchical generative model, with observed intermediate variables. Naturally, an alternate formulation is to learn the intermediate variables using the variational auto-encoder (VAE) framework, with a learned conditional prior to predict the latent variables given some conditioning. This formulation, with continuous latent variables and training an expressive prior using normalizing flows has been quite successful for speech synthesis [17, 36]. -->
1. 音频生成通常分为两个阶段：
    1. 基于中间表征（如 mel 谱）预测音频
    2. 基于条件信息（如文本）预测中间表征
    3. 然后可以用分层生成模型来解释这个过程
<!-- A closely related idea is to train the same varitional-autoencoder with discrete latent variables using VQ-VAE [38]. Arguably, discrete latent variables are a better choice since expressive priors can be trained using powerful autoregressive models that have been developed for modeling distributions over discrete variables [27]. Specifically, transformer language models [39] have already exhibited the capacity to scale with data and model capacity to learn arbitrarily complex distributions such as text[6], images[12, 44], audio [5, 41], music [1], etc. While modeling the prior is straightforward, modeling the discrete latent codes using a quantized auto-encoder remains a challenge -->
2. 用 VQ-VAE 训练离散潜变量的 VAE
    1. 离散潜变量更好，因为可以用强大的自回归模型训练先验
    2. 虽然建模先验很简单，用量化自编码器建模 discrete latent code 还是很难
<!-- Learning these discrete codes can be interpreted as a lossy compression task, where the audio signal is compressed into a discrete latent space by vector-quantizing the representations of an autoencoder using a fixed length codebook. This audio compression model needs to satisfy the following properties: 1) Reconstruct audio with high fidelity and free of artifacts 2) Achieve high level of compression along with temporal downscaling to learn a compact representation that discards low-level imperceptible details while preserving high-level structure [38, 33] 3) Handle all types of audio such as speech, music, environmental sounds, different audio encodings (such as mp3) as well as different sampling rates using a single universal model. -->
3. 学习离散码可以看作是一个有损压缩任务，通过将音频信号压缩到一个离散潜空间，使用固定长度码书的自编码器的表示进行矢量量化
4. 音频压缩模型需要满足以下特性：
    1. 高保真、无伪影的重构音频
    2. 高压缩率，同时进行降采样，学习紧凑表征，保留高级结构
    3. 可以用通用模型处理所有类型的音频，如语音、音乐、环境声音、不同的音频编码（如 mp3）以及不同的采样率
<!-- While the recent neural audio compression algorithms such as SoundStream [46] and EnCodec [8] partially satisfy these properties, they often suffer from the same issues that plague GAN-based generation models. Specifically, such models exhibit audio artifacts such as tonal artifacts [29], pitch and periodicity artifacts [25] and imperfectly model high-frequencies leading to audio that are clearly distinguishable from originals. These models are often tailored to a specific type of audio signal such as speech or music and struggle to model generic sounds. We make the following contributions: -->
5. 最近的神经音频压缩算法（如 SoundStream 和 EnCodec）虽然部分满足这些特性，但通常会出现 GAN 生成模型的问题
    1. 如音频伪影、音高和周期性伪影、高频不完美建模等
    2. 模型通常针对特定类型的音频信号，如语音或音乐，难以建模通用声音
<!-- • We introduce Improved RVQGAN a high fidelity universal audio compression model, that can compress 44.1 KHz audio into discrete codes at 8 kbps bitrate (~90x compression) with minimal loss in quality and fewer artifacts. Our model outperforms state-of-the-art methods by a large margin even at lower bitrates (higher compression) , when evaluated with both quantitative metrics and qualitative listening tests.
• We identify a critical issue in existing models which don’t utilize the full bandwidth due to codebook collapse (where a fraction of the codes are unused) and fix it using improved codebook learning techniques.
• We identify a side-effect of quantizer dropout - a technique designed to allow a single model to support variable bitrates, actually hurts the full-bandwidth audio quality and propose a solution to mitigate it.
• We make impactful design changes to existing neural audio codecs by adding periodic inductive biases, multi-scale STFT discriminator, multi-scale mel loss and provide thorough ablations and intuitions to motivate them.
• Our proposed method is a universal audio compression model, capable of handling speech, music, environmental sounds, different sampling rates and audio encoding formats. -->
6. 本文贡献：
    1. 提出了 Improved RVQGAN，可以将 44.1 KHz 音频压缩到 8 kbps 的离散 code 中，质量损失小，伪影少
    2. 通过改进 codebook，解决了现有模型中存在的不充分利用带宽的问题
    3. 认为 quantizer dropout 虽然使得单个模型支持可变比特率，但是实际上损害了全频带音频质量
    4. 添加 periodic inductive biase、多尺度 STFT 判别器、多尺度 mel 损失 
    5. 是一个通用音频压缩模型，能够处理语音、音乐、环境声音、不同的采样率和音频编码格式

## 相关工作（略）

## Improved RVQGAN 模型
