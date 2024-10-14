> NIPS 2024，Samsung Research
<!-- 翻译&理解 -->
<!-- In this paper, we address the challenge of speech enhancement in real-world
recordings, which often contain various forms of distortion, such as background
noise, reverberation, and microphone artefacts. We revisit the use of Generative
Adversarial Networks (GANs) for speech enhancement and theoretically show
that GANs are naturally inclined to seek the point of maximum density within
the conditional clean speech distribution, which, as we argue, is essential for the
speech enhancement task. We study various feature extractors for perceptual loss
to facilitate the stability of adversarial training, developing a methodology for
probing the structure of the feature space. This leads us to integrate WavLM-based
perceptual loss into MS-STFT adversarial training pipeline, creating an effective
and stable training procedure for the speech enhancement model. The resulting
speech enhancement model, which we refer to as FINALLY, builds upon the HiFi++
architecture, augmented with a WavLM encoder and a novel training pipeline.
Empirical results on various datasets confirm our model’s ability to produce clear,
high-quality speech at 48 kHz, achieving state-of-the-art performance in the field
of speech enhancement. -->
1. 语音增强的难点：实际录音包含各种形式的失真，如背景噪声、混响和麦克风伪影
2. 本文从理论上证明 GAN 对于 speech enhancement 任务至关重要
3. 研究了不同的 perceptual loss 特征提取器，提高了对抗训练的稳定性
4. 将 WavLM-based perceptual loss 集成到 MS-STFT 对抗训练中，从而提高模型训练稳定性

## Introduction
<!-- Speech recordings are often contaminated with background noise, reverberation, reduced frequency
bandwidth, and other distortions. Unlike classical speech enhancement (Ephraim & Malah, 1984;
Pascual et al., 2017), which considers each task separately, universal speech enhancement (Serrà
et al., 2022; Su et al., 2021; Liu et al., 2022) aims to restore speech from all types of distortions
simultaneously. Thus, universal speech enhancement seeks to generalize across a wide range of
distortions, making it more suitable for real-world applications where multiple distortions may
coexist. -->
1. 通用的语音增强需要在各类失真中恢复语音
<!-- Recent studies have categorized the problem of speech enhancement as a task of learning the clean
speech distribution conditioned on degraded signals (Lemercier et al., 2023; Serrà et al., 2022; Richter
et al., 2023). This problem is often addressed using diffusion models (Ho et al., 2020; Song et al.,
2020), which are renowned for their exceptional ability to learn distributions. Diffusion models
have recently achieved state-of-the-art results in universal speech enhancement (Serrà et al., 2022). 
However, the impressive performance of diffusion models comes with the high computational cost of
their iterative inference process.-->
2. diffusion 模型效果很好，但是计算速度慢
<!-- It is important to note that the speech enhancement problem does not require the model to learn the
entire conditional distribution. In practice, when presented with a noisy speech sample, the goal is
often to obtain the most probable clean speech sample that retains the lexical content and voice of the
original. This contrasts with applications such as text-to-image synthesis (Ramesh et al., 2022; Lee
et al., 2024; Rombach et al., 2022), where the objective is to generate a variety of images for each text
prompt due to the higher level of uncertainty and the need for diverse options to select the best image.
For most speech enhancement applications, such as voice calls and compensation for poor recording
conditions, capturing the entire conditional distribution is not necessary. Instead, it is more important
to retrieve the most likely sample of this distribution (the main mode), which might be a simpler task. -->
3. 语音增强不需要模型学习整个条件分布，只需要获取最可能的干净语音样本