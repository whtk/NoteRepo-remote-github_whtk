> arxiv 2023，Google
<!-- We introduce AudioPaLM, a large language model for speech understanding and
generation. AudioPaLM fuses text-based and speech-based language models,
PaLM-2 [Anil et al., 2023] and AudioLM [Borsos et al., 2022], into a unified
multimodal architecture that can process and generate text and speech with applica-
tions including speech recognition and speech-to-speech translation. AudioPaLM
inherits the capability to preserve paralinguistic information such as speaker iden-
tity and intonation from AudioLM and the linguistic knowledge present only in
text large language models such as PaLM-2. We demonstrate that initializing
AudioPaLM with the weights of a text-only large language model improves speech
processing, successfully leveraging the larger quantity of text training data used
in pretraining to assist with the speech tasks. The resulting model significantly
outperforms existing systems for speech translation tasks and has the ability to
perform zero-shot speech-to-text translation for many languages for which in-
put/target language combinations were not seen in training. AudioPaLM also
demonstrates features of audio language models, such as transferring a voice across
languages based on a short spoken prompt. We release examples of our method at:
https://google-research.github.io/seanet/audiopalm/examples -->
1. 提出 AudioPaLM，实现语音理解和生成的 LLM：
+ 将文本 LLM PaLM-2 和语音 LLM AudioLM 融合为统一的多模态架构，实现文本和语音生成
+ AudioPaLM 可以保留音色信息，如说话者身份和语调，以及 PaLM-2 中的 linguistic knowledge
2. 表明使用 text-only LLM 的权重初始化 AudioPaLM，可以提高 speech processing 性能

## Introduction
<!-- Large language models (LLMs) [Brown et al., 2020, Rae et al., 2021, Chowdhery et al., 2022] excel
at generating text for tasks that require the modeling of complex interactions as well as knowledge
retrieval, such as open-domain question answering or few-shot machine translation [Anil et al., 2023].
The remarkable generative abilities of the underlying system — a Transformer [Vaswani et al., 2017]
trained to predict sequences of discrete tokens — have been subsequently extended to continuous,
natural signals with images [Yu et al., 2022b] or audio waveforms [Lakhotia et al., 2021, Kreuk et al.,
2022, Wang et al., 2023] being converted into a stream of discrete units through a lossy compression
algorithm and then modeled in a sequential fashion as would be text. -->
<!-- In the context of audio generation, the AudioLM framework [Borsos et al., 2022] has introduced
a hierarchical approach which combines two types of audio tokens, with high-level coarse tokens
extracted from self-supervised embeddings [Chung et al., 2021] being used to condition the generation
of lower-level codes of a neural codec [Zeghidour et al., 2021]. This general framework, which makes 
little assumptions about the nature of the modeled audio signals, has been used to generate speech and
music [Kharitonov et al., 2023, Agostinelli et al., 2023, Donahue et al., 2023]. In the particular case
of text-to-music [Agostinelli et al., 2023] or text-to-speech [Kharitonov et al., 2023], a Transformer
model takes text tokens as inputs and generates audio tokens, such that text and audio vocabularies
do not interact with each other. Such models could naturally be converted into, respectively, music
captioning and speech recognition systems by swapping their inputs and outputs. Following this
observation, combining text and audio vocabularies into a multimodal, single vocabulary would allow
for training a single model in both directions.-->
1. 