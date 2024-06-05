> Google，ICLR 2024 reject。。。
<!-- 翻译 & 理解 -->
<!-- We present SoundStorm, a model for efficient, non-autoregressive audio generation. Sound- Storm receives as input the semantic tokens of AudioLM, and relies on bidirectional attention and confidence-based parallel decoding to gen- erate the tokens of a neural audio codec. Com- pared to the autoregressive generation approach of AudioLM, our model produces audio of the same quality and with higher consistency in voice and acoustic conditions, while being two orders of magnitude faster. SoundStorm gen- erates 30 seconds of audio in 0.5 seconds on a TPU-v4. We demonstrate the ability of our model to scale audio generation to longer se- quences by synthesizing high-quality, natural di- alogue segments, given a transcript annotated with speaker turns and a short prompt with the speakers’ voices. -->
1. 提出 SoundStorm，非自回归音频生成模型：
    1. 输入是 AudioLM 的 semantic token
    2. 依赖 bidirectional attention 和 confidence-based parallel decoding 生成对应于 codec 的 token
2. 相比于 AudioLM，在相同质量和更高一致性下 更快（在 TPU-v4 上 0.5 秒内生成 30 秒音频）
3. 可以生成长序列

## Introduction
<!-- The problem of generating long audio token sequences can be addressed by at least three orthogonal approaches, or a combination thereof: i) efficient attention mechanisms (Kitaev et al., 2020; Choromanski et al., 2021; Xiong et al., 2021; Hawthorne et al., 2022), ii) non-autoregressive, parallel decoding schemes (Gu et al., 2017; Ghazvininejad et al., 2019; Chang et al., 2022), iii) custom architectures adapted to the special structure of the tokens produced by neural audio codecs (Kreuk et al., 2022; Wang et al., 2023; Lee et al., 2022). However, in the context of modeling the token sequence of neural audio codecs, either unconditionally or based on weak conditioning such as text, the efficient generation of long, high-quality audio segments remains an open problem. -->
1. 解决长音频 token 序列生成问题有三种方法：
    + 高效的 attention 机制
    + 非自回归、并行解码
    + 可以适应 codec 产生的 token 的自定义架
<!-- We believe that it is the special structure of the audio token sequence that holds the most promise for future advances in long-sequence audio modeling. Concretely, both Sound- Stream (Zeghidour et al., 2022) and EnCodec (De ́fossez et al., 2022) rely on Residual Vector Quantization (RVQ), where each compressed audio frame is quantized by a series of quantizers, with each quantizer operating on the residual of the previous one, and the number of quantizers control- ling the overall bitrate. This induces a hierarchical token structure, where tokens from finer RVQ levels contribute less to the perceptual quality, allowing for efficient factor- izations and approximations of the joint distribution of the token sequence. Hence, the models and decoding schemes should take this special structure of the input into account for efficient training and inference. -->
2. 作者认为，音频 token 序列的特殊结构有助于长序列音频建模
<!-- In this work, we present SoundStorm, a method for ef- ficient and high-quality audio generation. SoundStorm addresses the problem of generating long audio token se- quences by relying on: i) an architecture adapted to the hierarchical structure of the audio tokens, ii) a parallel, non-autoregressive, confidence-based decoding scheme in- spired by MaskGIT (Chang et al., 2022) for residual vector- quantized token sequences. -->
3. 提出 SoundStorm，解决长音频 token 生成问题：
    + 提出适应音频 token 的层次结构的架构
    + 是并行、非自回归、基于 confidence 的解码方法
<!-- SoundStorm relies on a bidirectional attention-based Con- former (Gulati et al., 2020) that is trained to predict masked audio tokens produced by SoundStream given a condition- ing signal such as the semantic tokens of AudioLM (Borsos et al., 2022). On the input side, it sums up the embeddings of the tokens corresponding to the same SoundStream frame, such that the internal sequence length for the self-attention is identical to the number of SoundStream frames, and in- dependent of the number of quantizers in the RVQ. The output embeddings are then processed by separate heads per RVQ level to predict the masked target tokens. At inference time, given the conditioning signal, SoundStorm starts with all audio tokens masked out, and fills in the masked tokens RVQ level-by-level over several iterations, predicting multi- ple tokens in parallel during a single iteration within a level. To support this inference scheme, we propose a masking scheme for training that mimics the inference procedure. -->
4. SoundStorm 依赖于双向 attention-based Conformer，训练预测 SoundStream 生成的 masked 音频 token：
    + 输入端，将相同 SoundStream frame 的 token embeddings 相加，使得 self-attention 的内部序列长度与 SoundStream frame 数相同，与 RVQ 中的 quantizers 数无关
    + 输出 embeddings 由每个 RVQ level 的独立 heads 处理，预测 masked 目标 tokens
    + 推理时，给定条件，SoundStorm 从所有音频 token 开始，逐 level 填充 masked tokens，并行预测多个 token
<!-- We demonstrate that SoundStorm can serve as AudioLM’s acoustic generator, replacing both AudioLM’s stage two (coarse acoustic model) and stage three (fine acoustic model). SoundStorm produces audio two orders of mag- nitude faster than AudioLM’s hierarchical autoregressive acoustic generator with matching quality and improved con- sistency in terms of speaker identity and acoustic condi- tions. Furthermore, we show that SoundStorm, coupled with the text-to-semantic modeling stage of SPEAR-TTS (Kharitonov et al., 2023), can synthesize high-quality, nat- ural dialogues, allowing one to control the spoken content (via transcripts), speaker voices (via short voice prompts) and speaker turns (via transcript annotations). When synthe- sizing dialogues of 30 seconds, we measure a runtime of 2 seconds on a single TPU-v4 (Jouppi et al., 2023). -->
5. SoundStorm 可以作为 AudioLM 的 acoustic generator，替代 stage two 和 stage three，生成音频比 hierarchical autoregressive acoustic generator 快，效果更好

## 相关工作（略）

## 