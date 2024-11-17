> NIPS 2023，MetaAI
<!-- 翻译&理解 -->
<!-- We tackle the task of conditional music generation. We introduce MUSICGEN, a sin-
gle Language Model (LM) that operates over several streams of compressed discrete
music representation, i.e., tokens. Unlike prior work, MUSICGEN is comprised of
a single-stage transformer LM together with efficient token interleaving patterns,
which eliminates the need for cascading several models, e.g., hierarchically or up-
sampling. Following this approach, we demonstrate how MUSICGEN can generate
high-quality samples, both mono and stereo, while being conditioned on textual
description or melodic features, allowing better controls over the generated output.
We conduct extensive empirical evaluation, considering both automatic and human
studies, showing the proposed approach is superior to the evaluated baselines on a
standard text-to-music benchmark. Through ablation studies, we shed light over
the importance of each of the components comprising MUSICGEN. Music samples,
code, and models are available at github.com/facebookresearch/audiocraft. -->
1. 提出 MUSICGEN，单个 LM 实现 conditional 音乐生成
2. 模型由单个 transformer LM 和 token 组成，可以合成高质量的 mono 和 stereo 音乐
3. 代码模型开源：github.com/facebookresearch/audiocraft

## Introduction
<!-- Text-to-music is the task of generating musical pieces given text descriptions, e.g., “90s rock song with
a guitar riff”. Generating music is a challenging task as it requires modeling long range sequences.
Unlike speech, music requires the use of the full frequency spectrum [Müller, 2015]. That means
sampling the signal at a higher rate, i.e., the standard sampling rates of music recordings are 44.1
kHz or 48 kHz vs. 16 kHz for speech. Moreover, music contains harmonies and melodies from
different instruments, which create complex structures. Human listeners are highly sensitive to
disharmony [Fedorenko et al., 2012, Norman-Haignere et al., 2019], hence generating music does not
leave a lot of room for making melodic errors. Lastly, the ability to control the generation process in
a diverse set of methods, e.g., key, instruments, melody, genre, etc. is essential for music creators. -->
1. Text-to-music 是根据文本描述生成音乐，相比于语音，音乐需要更高的采样率，包含多种乐器的和声和旋律
<!-- Recent advances in self-supervised audio representation learning [Balestriero et al., 2023], sequential
modeling [Touvron et al., 2023], and audio synthesis [Tan et al., 2021] provide the conditions to
develop such models. To make audio modeling more tractable, recent studies proposed representing
audio signals as multiple streams of discrete tokens representing the same signal [Défossez et al.,
2022]. This allows both high-quality audio generation and effective audio modeling. However, this
comes at the cost of jointly modeling several parallel dependent streams. -->
2. 