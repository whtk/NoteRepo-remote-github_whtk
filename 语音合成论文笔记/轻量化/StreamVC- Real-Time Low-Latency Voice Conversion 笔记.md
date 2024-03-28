> ICASSP 2024，Google
<!-- 翻译&理解 -->
<!-- We present StreamVC, a streaming voice conversion solution that preserves the content and prosody of any source speech while matching the voice timbre from any target speech. Un- like previous approaches, StreamVC produces the resulting waveform at low latency from the input signal even on a mo- bile platform, making it applicable to real-time communica- tion scenarios like calls and video conferencing, and address- ing use cases such as voice anonymization in these scenar- ios. Our design leverages the architecture and training strat- egy of the SoundStream neural audio codec for lightweight high-quality speech synthesis. We demonstrate the feasibility of learning soft speech units causally, as well as the effec- tiveness of supplying whitened fundamental frequency infor- mation to improve pitch stability without leaking the source timbre information. -->
1. 提出 StreamVC，流式语音转换，可以 source speech 的韵律和内容，同时匹配 target speech 的音色
2. 可以在实时生成波形，适用于实时通信场景
3. 模型基于 SoundStream，实现轻量化高质量语音合成
4. 展示了学习 soft speech units 的可行性，以及可以通过提供白化的基频（whitened fundamental frequency）来提高音高稳定性而不泄漏源音色

## Introduction
<!-- Voice conversion refers to altering the style of a speech signal while preserving its linguistic content. While style encom- passes many aspects of speech, such as emotion, prosody, ac- cent, and whispering, in this work we focus on the conversion of speaker timbre only while keeping the linguistic and para- linguistic information unchanged. -->
1. VC 目标是，改变语音的风格，同时保留其内容
2. 风格包括很多方面，如情感、韵律、口音、耳语，本文只关注音色的转换，保持语言和语音信息不变
