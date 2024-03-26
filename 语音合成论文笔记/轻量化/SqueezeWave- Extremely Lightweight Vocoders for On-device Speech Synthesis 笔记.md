> computer science 2020，UC Berkeley
<!-- 翻译 & 理解 -->
<!-- Automatic speech synthesis is a challenging task that is becoming increasingly important as edge devices begin to interact with users through speech. Typical text-to-speech pipelines include a vocoder, which translates intermediate audio representations into an audio waveform. Most existing vocoders are difficult to parallelize since each generated sample is conditioned on previous samples. WaveGlow is a flow-based feed-forward alternative to these auto-regressive models (Prenger et al., 2019). However, while WaveGlow can be easily parallelized, the model is too expensive for real-time speech synthesis on the edge. This paper presents SqueezeWave, a family of lightweight vocoders based on WaveGlow that can generate audio of similar quality to WaveGlow with 61x - 214x fewer MACs -->
1. 大多现有的 vocoder 很难并行化，因为每个生成的样本都是基于之前的样本
2. 提出 SqueezeWave，基于 WaveGlow 的一系列轻量级 vocoder，可以生成质量相似的音频，但是 MACs 减少了 61-214 倍

## Introduction
