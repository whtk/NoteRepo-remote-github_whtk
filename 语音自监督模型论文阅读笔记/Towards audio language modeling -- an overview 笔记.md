> preprint，2024，李宏毅

1. neural audio codecs 最开始用于压缩音频，但是也可以用于 LM 中的 tokenization，将连续音频转换为离散的 code
2. 本文是对 neural audio codec 和基于它的 LM 的概述

## Introduction

1. 音频不仅包含文本内容，还包含说话人的音色、情感等信息
2. 现有的 codec-based LM 和 codec 模型如图：![](image/Pasted%20image%2020240524112551.png)

3. 理想的 codec 应该保留内容的同时保留说话人相关信息；理想的 LM 应该能够泛化到各种音频类型，如语音、音乐等
4. 本文对过去三年中的 codec 和 LM 进行综述，包括了六个开源 neural codec 模型和十一个 codec-based LM

## Neural Audio Codec 模型

传统的 codec 是基于 psycho-acoustics 和语音合成的，而 neural codec 在压缩和信号重建上表现更好。六个 neural codec 中，每个模型都有不同的训练细节，总共有 15 个不同的模型：
![](image/Pasted%20image%2020240524113117.png)

### 不同 codec 概述

+ SoundStream 包括 encoder、quantizer 和 decoder，使用 SEANets 作为 encoder 和 decoder，quantizer 使用 RVQ，训练时使用 reconstruction 和 adversarial loss
    + SoundStorm 是 SoundStream 的改进版，使用了专门针对音频 token 层次结构的架构，用的是并行的非自回归解码
+ Encodec 结构类似于 SoundStream，但是用了 LSTM 层来提高建模能力，采用 基于 transformer 的 LM 来建模 RVQ 的编码，可以提高序列建模能力
    + AudioDec 为 Encodec 的改进，采用分组卷积来实现实时的、流式的网络，可以生成 48khz 的音频样本
+ AcademiCodec 采用 group-residual vector quantization（GRVQ），采用多个并行的 RVQ 组，可以在受限数量的 codebook 下提高重构性能，从而可实现非常低的 BPS
+ SpeechTokenizer 采用 RVQ + encoder-decoder 架构，同时引入 semantic 和 acoustic tokens，将语音信息的不同方面分到不同的 RVQ 层中，第一层 RVQ 用于学习 Hubert tokens，这种方法可以增强不同 RVQ 层之间的信息解耦
+ DAC 是一个通用的 neural codec，可以在音频、音乐和语音 等数据上都实现高质量的合成。采用了 periodic activation functions、增强的 RVQ、随机 quantizer dropout 等训练技巧（periodic activation functions 很重要）
+ Fun-Codec 是一个 frequency-domain codec，可以在更少的参数和更低的计算复杂度下实现相当的性能，同时在 codec tokens 中加入语义信息可以提高 speech quality

### 方法对比

这些 codec 采用的技巧对比如下：
![](image/Pasted%20image%2020240524115232.png)

对于 discriminator，Encodec 使用 MS-STFTD，AudioDec 使用 HiFi-GAN-based MPD，AcademiCodec 使用 Encodec 的 MS-STFTD 和 HiFi-GAN-based MPD 和 MSD，SpeechTokenizer 和 FunCodec 使用和 AcademiCodec 相同的 discriminator，DAC 使用 MS-MB-STFTD 来提高 phase modeling 和减少 aliasing artifacts

此外，SpeechTokenizer 使用 Hubert 第九层 的 semantic tokens 作为 RVQ 的 teacher，FunCodec 将 semantic tokens 和 codec 结合来提高音频质量。SpeechTokenizer 和 FunCodec 使用 K-means 来初始化 VQ codebook，DAC 使用 snake activation 来控制周期信号的频率，AcademiCodec 使用多个 RVQ codebooks 来表示中间特征，Encodec 使用一个小 transformer 模型来对 quantized units 进行 entropy coding，减少带宽并加速编解码。


## 现有的 codec-based 语音 LM

下图是 codec-based 语音 LM 的流程：
![](image/Pasted%20image%2020240524130507.png)

+ 首先将 context 信息（如文本、MIDI）转为 context codes，同时将音频编码为 codec codes
+ 在 LM 阶段，用 context 和 codec codes 来生成目标 codec code 序列
+ 得到的 codec codes 传给 codec decoder 生成音频

### codec-based LM 概述

+ AudioLM 为第一个采用 codec 进行 LM 的模型，其分为两个阶段：
    + 阶段一：采用自监督的 w2v-BERT 模型生成 semantic token
    + 阶段二：用阶段一生成的 token 作为条件，采用 SoundStream 来生成 acoustic token
+ VALL-E、VALL-E X 和 SpeechX 都是都是基于 EnCodec 训练的 LM，可以生成高质量的个性化语音
    + VALL-E 可以用 3 秒的录音生成高质量的个性化语音
    + VALL-E X 可以用目标语言的一句话作为 prompt 来生成高质量的语音
    + SpeechX 提供了一个统一的框架，不仅可以做 zero-shot TTS，还可以做 speech enhancement 和 speech editing
+ VioLA、AudioPaLM 和 LauraGPT 都可以生成文本和音频：
    + VioLA 通过整合 text tokens 和 audio tokens 来进行 LM，同时使用 task IDs 和 language IDs
    + AudioPaLM 使用统一的词表，可以生成文本和语音，从 PaLM-2 进行初始化，其音频 tokenization 类似于 AudioLM，使用了 SoundStorm
    + LauraGPT 采用 Conformer encoder 编码音频为连续表征，使用 FunCodec 生成离散 code，其基于 Qwen-2B，可以处理文本和音频输入，生成文本和音频
+ UniAudio 可以生成 speech、sounds、music 和 singing，输入为文本或 acoustic tokens，采用 multi-scale Transformer 模型来提高自回归预测速度，其 codec 来自 EnCodec
+ AudioGen 训练了一个 SoundStream 模型来得到 audio tokens，然后训练了一个 LM 来利用文本特征生成 audio tokens
+ MusicLM 和 MusicGen 也是类似的模型，MusicLM 使用 Mulan 生成 semantic tokens，然后用 Soundstream 生成 acoustic features，MusicGen 输入为文本描述或旋律特征，生成 tokens，可以合成高保真的音乐

+ 另一种方法是用量化 self-supervised speech representations 得到的离散单元，其包含丰富的 acoustic 和 linguistic 信息，但缺少说话人和 paralinguistic 信息。这个方向主要关注 speech 的语义，可以使用 encoder 学习说话人特征和 prosody。speech-resynthesis 使用这些离散单元和 prosody 和 speaker encoders 来编码 speech 为低比特率的 code，然后用 decoder 重构 speech 信号，实现低比特率传输。这些离散单元可以看作是“pseudo-text”，可以用来训练 textless speech LM，如 GSLM、pGSLM、dGSLM 和 TWIST。这些 speech LM 可以执行 spoken language modeling 和 speech continuation。在 speech translation 领域，Unit mBART 和 wav2vec 2.0 encoder 可以直接预测翻译的离散单元，UnitY 进一步整合 text modality 来增强 speech translation。Seamless models 结合 UnitY 框架来执行 speech-to-text 和 speech-to-speech translation。这些强大的 speech LM 的发展，研究人员开始探索在 speech LM 上使用 prompting 来进行各种 speech 处理任务，如 prompt tuning、in-context learning 和 instruction tuning 等


### codec-based LM 对比

下表对比了不同 codec-based LM 的输入、输出和下游任务：
![](image/Pasted%20image%2020240524133503.png)

