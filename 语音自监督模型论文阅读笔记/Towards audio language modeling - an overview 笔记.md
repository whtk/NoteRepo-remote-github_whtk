> MIT、台大，preprint 2024

1. audio codec 最开始用于压缩音频，最近则用来作为 tokenizer 将连续的音频转为离散的 code
2. 本文概述了 neural audio codec 模型和基于 codec 的 LM

## Introduction

1. 现有的 codec-based LM 和 codec models 如图：
![](image/Pasted%20image%2020240719173815.png)
2. 理想的 codec 应该保留内容的同时保留语音和说话人相关的信息
3. 本文分析 6 个开源 neural codec models 和 11 个 codec-based LM

## Neural Audio Codecs 比较

传统的 codec 基于 psycho-acoustics 和 speech synthesis，而 neural codec models 在压缩和信号重建方面表现更好。本文总结了 6 个 codec models，共 15 个模型：
![](image/Pasted%20image%2020240719174222.png)

### 概述

1. SoundStream 是一个经典的 neural codec 模型，包含 encoder、quantizer 和 decoder 模块，使用 SEANets 作为 encoder 和 decoder，quantizer 使用 RVQ；训练时使用 reconstruction 和 adversarial loss；SoundStorm 是 SoundStream 的改进版，实现了高效和高质量的音频生成，使用了专门针对音频 token 的 hierarchical 架构，使用了并行的非自回归解码
2. Encodec 在 SoundStream 的基础上增加了 LSTM 和 Transformer-based LM，AudioDec 是 Encodec 的增强，使用了 group convolution 和 HiFi-GAN 生成 48 kHz 的高保真音频
3. AcademiCodec 使用 group-residual vector quantization，使用多个并行 RVQ groups 提高重建性能，同时使用有限的 codebooks，达到了低 BPS，解决了 speech language modeling 中长语音 token 的问题
4. SpeechTokenizer 是一个 speech LM，使用了 Encoder-Decoder 架构和 RVQ，通过集成 semantic 和 acoustic tokens，SpeechTokenizer 分层地将 speech 信息分开，通过学习 Hubert tokens 来增强第一层 RVQ 的 semantic 信息
5. DAC 是一个 universal neural codec model，通过使用周期激活函数、增强的 RVQ、random quantizer dropout、调整 adversarial 和 reconstruction loss 来保持高保真音频质量；作者强调了周期激活函数的重要性
6. Fun-Codec 是一个 frequency-domain codec，可以在更少的参数和更低的计算复杂度下实现可比的性能，同时发现在 codec tokens 中加入 semantic 信息可以提高 speech 质量

### 从方法的角度比较

比较了几种 codec 的技术，如下表：
![](image/Pasted%20image%2020240719175256.png)

A-F 代表不同的 codec models。

discriminator 是 codec models 中的关键：
+ Encodec 使用 Multi-scale-STFT Discriminator
+ AudioDec 使用 HiFi-GAN-based MPD
+ AcademiCodec 使用 Encodec 的 MS-STFTD 和 HiFi-GAN-based MPD 和 MSD
+ SpeechTokenizer 和 Funcodec 使用和 AcademiCodec 相同的 discriminator
+ DAC 使用 MS-MB-STFTD 来提高 phase modeling 和减少 aliasing artifacts

其他方面：
+ SpeechTokenizer 使用 Hubert L9 的 semantic tokens 作为 RVQ 的 teacher
+ FunCodec 通过将 semantic tokens 和 audio codec 结合来提高音频质量
+ SpeechTokenizer 和 FunCodec 使用 K-means 来初始化 VQ codebook，提高 code 的利用率
+ DAC 使用 snake activation 来控制周期信号的频率
+ AcademiCodec 使用多个 RVQ codebooks 来表示中间特征
+ Encodec 使用一个小 transformer model 来对 quantized units 进行 entropy coding，减少带宽并加速编码和解码

### 实现细节

上表也列出了 codebook number、training data、sampling rate 和 BPS。从 training data 来看：
+ SpeechTokenizer、AudioDec 和 FunCodec 使用英文 speech dataset
+ AcademiCodec 使用双语 speech dataset，包括 AISHELL 和 LibriTTS 和 VCTK
+ DAC 和 Encodec 使用 speech、music 和 audio 数据

## 现有的 Codec-based Speech Language Models

如图：
![](image/Pasted%20image%2020240719175757.png)

codec-based audio language modeling 先将 context 信息转为 context codes，同时将 audio 编码为 codec codes，然后在 LM 阶段使用这些 codes 生成目标 codec code sequence，最后将目标 codec code sequence 传给 codec decoder 得到 auduio。

### 概述

1. AudioLM 是第一个使用 codec codes 进行 LM 的模型，使用了 hierarchical approach，包含两个阶段，第一个阶段使用 self-supervised w2v-BERT model 生成 semantic tokens，第二个阶段使用 SoundStream neural codec 生成 acoustic tokens
> 是不是自监督模型生成的 token 都叫 semantic tokens，codec 生成的 token 都叫 acoustic tokens？
2. VALL-E、VALL-E X 和 SpeechX 都来自 Microsoft，是 neural codec LM，可以生成从 EnCodec 得到的 discrete codes，基于 textual 或 acoustic inputs。VALL-E 可以通过 3 秒的 enroll-ment recording 生成高质量的 personalized speech，VALL-E X 可以通过单个 speech utterance 生成目标语言的 speech，SpeechX 提供了一个统一框架，不仅可以做 zero-shot TTS，还可以做 speech enhancement 和 speech editing
3. ViaLA、AudioPaLM 和 LauraGPT 可以生成 text 和 audio。ViaLA 使用了 text tokens 和 audio tokens，同时使用了 task IDs 和 language IDs。AudioPaLM 使用了 unified vocabulary，是一个 decoder-only 的自回归模型，可以处理和生成 text 和 speech。LauraGPT 也可以处理 audio 和 text，使用了 Conformer encoder 和 FunCodec decoder
4. UniAudio 可以生成 speech、sounds、music 和 singing，使用 textual 或 acoustic tokens 作为输入，使用 multi-scale Transformer model 来提高自回归预测速度，使用了大的 global transformer 来预测第一层 codec codes，小的 local transformer 来预测后续的 codec codes；codec 模型来自 EnCodec
5. AudioGen 训练了 SoundStream 模型得到 audio tokens，然后训练 LM 以使用 textual features 生成 audio tokens；MusicLM 和 MusicGen 类似，MusicLM 使用了 Mulan 的 music tokens 生成 semantic tokens，然后使用 music tokens 和 semantic tokens 生成 acoustic features；MusicGen 输入 textual descriptions 或 melodic features 生成 tokens
6. 另一种 speech LM 利用 quantizing self-supervised speech representations 得到的 discrete units，这些 units 包含丰富的 acoustic 和 linguistic 信息，但缺乏 speaker 和 paralinguistic 信息。这个方向关注 speech 的 semantics，可以使用 encoders 学习 speaker 特征和 prosody。speech-resynthesis 时，用这些 discrete units 和 prosody 和 speaker encoders 将 speech 编码为 low-bitrate codes，然后 decoder 将这些 codes 重建为 speech signal。这些 discrete units 可以被看作是 pseudo-text，用于训练 textless speech LM。例如：GSLM、pGSLM、dGSLM 和 TWIST，这些 speech LM 可以实现 spoken language modeling 和 speech continuation。对于 speech translation，Unit mBART 和 wav2vec 2.0 encoder 可以直接预测翻译的 discrete units，UnitY 进一步结合 text modality 来增强 speech translation。Seamless models 结合 UnitY 框架进行 speech-to-text 和 speech-to-speech translation。

### 比较

下表比较了不同 codec-based LM 的输入输出和下游任务：
![](image/Pasted%20image%2020240719194106.png)

