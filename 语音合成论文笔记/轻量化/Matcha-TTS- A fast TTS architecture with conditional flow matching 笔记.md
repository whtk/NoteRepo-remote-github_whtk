> ICASSP 2024，KTH Royal Institute of Technology
<!-- 翻译 & 理解 -->
<!-- We introduce Matcha-TTS, a new encoder-decoder architecture for speedy TTS acoustic modelling, trained using optimal-transport conditional flow matching (OT-CFM). This yields an ODE-based decoder capable of high output quality in fewer synthesis steps than models trained using score matching. Careful design choices ad- ditionally ensure each synthesis step is fast to run. The method is probabilistic, non-autoregressive, and learns to speak from scratch without external alignments. Compared to strong pre-trained baseline models, the Matcha-TTS system has the smallest memory footprint, rivals the speed of the fastest model on long utterances, and attains the highest mean opinion score in a listening test. -->
1. 提出 Matcha-TTS，快速的 TTS 声学模型，采用 optimal-transport conditional flow matching（OT-CFM）训练
2. Matcha-TTS 采用 ODE-based 解码器，能够在更少的合成步骤中产生高质量输出
3. Matcha-TTS 是 probabilistic 且非自回归的，可以在没有外部 alignments 的情况下从头学
4. Matcha-TTS 占用内存小，MOS 效果好

## Introduction

1. DPM 的采样速度慢
2. 提出 Matcha-TTS，基于 CNFs 的 TTS 声学模型，两个创新点：
    <!-- Tobeginwith,weproposeanimprovedencoder-decoderTTS architecture that uses a combination of 1D CNNs and Trans- formers in the decoder. This reduces memory consumption and is fast to evaluate, improving synthesis speed. -->
	1. 在 decoder 中采用 1D CNNs 和 Transformer 来减少内存消耗，提高合成速度
    <!-- Second,wetrainthesemodelsusingoptimal-transportcondi- tional flow matching (OT-CFM) [14], which is a new method to learn ODEs that sample from a data distribution. Com- pared to conventional CNFs and score-matching probability flow ODEs, OT-CFM defines simpler paths from source to target, enabling accurate synthesis in fewer steps than DPMs. -->
    2. 采用 OT-CFM 训练模型，比传统 CNFs 和 score-matching probability flow ODEs 更简单，能够在更少的 step 中实现准确合成