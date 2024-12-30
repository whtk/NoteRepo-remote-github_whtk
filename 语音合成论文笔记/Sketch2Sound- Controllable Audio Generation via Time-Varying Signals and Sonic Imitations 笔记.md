> Adobe、西北大学
<!-- 翻译 & 理解 -->
<!-- Abstract—We present Sketch2Sound, a generative audio model capable
of creating high-quality sounds from a set of interpretable time-varying
control signals: loudness, brightness, and pitch, as well as text prompts.
Sketch2Sound can synthesize arbitrary sounds from sonic imitations
(i.e., a vocal imitation or a reference sound-shape). Sketch2Sound can
be implemented on top of any text-to-audio latent diffusion transformer
(DiT), and requires only 40k steps of fine-tuning and a single linear
layer per control, making it more lightweight than existing methods like
ControlNet. To synthesize from sketchlike sonic imitations, we propose
applying random median filters to the control signals during training,
allowing Sketch2Sound to be prompted using controls with flexible levels
of temporal specificity. We show that Sketch2Sound can synthesize sounds
that follow the gist of input controls from a vocal imitation while retaining
the adherence to an input text prompt and audio quality compared to
a text-only baseline. Sketch2Sound allows sound artists to create sounds
with the semantic flexibility of text prompts and the expressivity and
precision of a sonic gesture or vocal imitation. Sound examples are
available at https://hugofloresgarcia.art/sketch2sound/. -->
1. 提出 Sketch2Sound，可以从时变控制信号（loudness、brightness、pitch 和文本提示）生成高质量音频，可以从 sonic imitations（声音模仿或参考声音形状）合成任意声音
2. Sketch2Sound 可以在任何 T2A latent DiT 中实现，只需 40k 步微调，比 ControlNet 更轻量
3. 为了从 sketchlike sonic imitations 合成音频，提出在训练时对控制信号进行随机中值滤波

## Introduction
<!-- Sound design is the craft of storytelling through sonic composition.
Within sound design, Foley sound is a technique where special sound
effects are designed and performed in sync to a film during post-
production [1]. These sound scenes are typically performed by a
Foley artist on a stage equipped with abundant sound instruments
and other soundmaking materials1. Foley sound is a skilled and
gestural performance art: performing a sound scene with sound-
making objects and instruments (instead of arranging pre-recorded
samples post hoc) allows sound artists to create fluent and temporally
aligned sounds with a “human” (i.e., gestural) touch. Adding this
gestural touch to the resulting sound composition often results in a
sonic product of great aesthetic and production value. -->
1. foley sound 是使用特殊音效设计和表演电影的技术，声音设计师需要修改生成声音的 temporal 特征，与视觉同步
<!-- Recent research in generative modeling for sound has paved the
way for text-to-sound systems [2]–[4], where a user can create sound
samples from text descriptions of a sound (e.g., “explosion”). While
the text-to-sound paradigm can help a sound designer find sounds
more quickly (and, perhaps in the future, with a higher degree of
specificity), a sound designer still has to painstakingly modify the
temporal characteristics of the generated sound so that they can be
in sync with the visuals in the editing timeline. This is in opposition
to the natural way that Foley artists gesturally create sound effects
by physically performing with physical soundmaking objects. -->
<!-- To overcome the drawbacks of a purely text-to-audio interaction,
several works in the music domain sought to condition generative
models on audio [5], parallel instrument stems [6], melody [7], sound
event timestamps and frequency [8], or multiple structural control
signals like song structure and dynamics [9]. Notably, [10] condition
an audio VAE on control signals such as brightness and loudness,
though their experiments are limited to models trained on narrow
sound distributions (e.g., violin, darbouka, speech) and not a multi-
distribution text-to-audio model. For speech, [11] proposes a fully
interpretable and disentangled representation for speech generation
and editing, which allows for fine-grained control over the pitch,
loudness, and phonetic pronunciation of speech. -->
2. 很多方法把音频、平行乐器、旋律、声音事件时间戳和频率或多个结构控制信号作为生成模型的条件
<!-- The human voice is a gestural sonic instrument [12]: it allows us to
realize sounds without having to perform any symbolic abstraction
(i.e., putting a sound into words) beforehand. When humans com-
municate audio concepts to other people (rather than software), they
typically combine descriptive language and vocal imitation [13]–[15].
In doing so, one approximates the audio by mapping the pitch, timbre,
and temporal properties of the sound to those of the voice. This is a
more natural method than describing the evolution of pitch, timing,
and timbre via pure text descriptions [13]. -->
3. 人类的声音是一种 gestural sonic 乐器，可以通过 discussion 和 vocal imitation 近似声音。
<!-- We propose Sketch2Sound: a text-to-audio model that can
create high-quality sounds from sonic imitation prompts by
following interpretable, fine-grained time-varying control signals
that can be easily extracted from any audio signal at different
levels of temporal detail: loudness, brightness (spectral centroid)
and pitch. We expand upon previous work [16] by developing a
method capable of following the loudness, brightness and pitch of
a vocal imitation, with the option to drop any of the three controls.
Additionally, we propose a technique that varies the temporal detail
of the control signals used during training by applying median filters
of different window sizes to the control signals before using them
as input. This allows sound artists to specify the degree of temporal
precision to which a generative model should follow the specified
control signals, which improves sound quality in sounds that may be
too hard to perfectly imitate with one’s voice. -->
4. 提出 Sketch2Sound：可以从 sonic imitation prompt 生成高质量声音，可以提取 loudness、brightness 和 pitch 作为控制信号，通过中值滤波调整控制信号的时间精度
<!-- This method is not limited to just vocal imitation: any kind
of sonic imitation can be used to drive our proposed generative
model – we place the focus on vocal imitation due to people’s innate
ability to imitate sounds with our voices. Vocal imitations can always
be augmented through other sonic gestures like clapping, tapping,
playing instruments, etc. Sketch2Sound can be added to any existing
latent diffusion transformer (DiT) sound generation model with as
little as 40k fine-tuning steps. Unlike ControlNet methods [17], [18]
that require an extra trainable copy of the entire neural network
encoder, Sketch2Sound requires only a single linear layer per control. -->
5. 这种方法不仅限于 vocal imitation，任何 sonic imitation 都可以驱动此模型；Sketch2Sound 可以添加到任何现有的 DiT 模型中，只需 40k 步微调，比 ControlNet 更轻量
<!-- Our experiments show that Sketch2Sound can generate sounds that
closely follow the input control signals (loudness, spectral centroid,
and pitch/periodicity) from a vocal imitation while still achieving
a high degree of adherence to a text prompt and an audio quality
comparable to the text-only pre-trained model. We show that our
median filtering technique leads to improved audio quality and text
adherence when generating sounds from vocal imitations. We also
show that, during inference, a user can arbitrarily specify a degree
of temporal detail by choosing a median filter size, allowing them to
navigate the trade-off between strict adherence to the vocal imitations
and audio quality and text adherence. -->
6. 实验表明 Sketch2Sound 可以生成与 vocal imitation 控制信号相符的声音，同时保持与文本提示和音频质量的高度一致性；中值滤波技术可以提高音频质量和文本一致性
<!-- To the best of our knowledge, this is the first sound generation
model capable of following vocal imitations and text prompts by
conditioning on a set of holistic control signals suitable for generating
sound objects with fine-grained, gestural control of pitch, loudness,
and brightness. We believe Sketch2Sound will give sound artists a
more expressive, controllable, and gestural interaction for generating
sound-objects than existing text-to-audio and other conditional sound
generation systems. We highly encourage the reader to listen to our
expansive set of audio examples demonstrating Sketch2Sound. -->

## 方法
<!-- We propose a method for conditioning an audio latent diffusion
model on a set of interpretable, time-varying control signals that
are suitable for multiple tasks. These include: generating variations
of sounds, modifying existing sounds, and generating new sounds
expressively via (optionally text-prompted) sonic imitations. -->
本文提出一种在音频 LDM 模型上使用可解释的时变控制信号的方法。

<!-- Time-varying control signals for sound objects -->
### 时变控制信号
<!-- We choose the following three control signals to be used as
conditioning for Sketch2Sound: -->
选择三种控制信号：
<!-- Loudness: We extract the per-frame loudness of an audio signal
by performing an A-weighted sum across the frequency bins in
a magnitude spectrogram [11] and taking the RMS of the result. -->
+ loudness：从音频信号通过 A-weighted sum 提取每帧的 loudness
<!-- Pitch and Periodicity: We use the raw pitch probabilities of
the CREPE [19], [20] (“tiny” variant) pitch estimation model.
To avoid leaking timbral information in this signal, we zero out
all probabilities below 0.1 in the pitch probability matrix. -->
+ pitch 和周期性：使用 CREPE pitch 估计模型的原始 pitch 概率，将概率矩阵中小于 0.1 的概率置零
<!-- Spectral Centroid is defined as the center of mass of the
frequency spectrum for a given audio frame. Frames with a
higher spectral centroid will be perceived as having a brighter
timbre. To preprocess the centroid, we convert the signal from
linear frequency space (i.e., Hz) to a continuous MIDI-like
representation, scaled to roughly a (0,1) range by dividing the
input signal by 127 (note G9, roughly 12.5kHz), which we found
to stabilize the first steps of training. -->
+ spectral centroid：定义为音频帧的频谱质心，将信号从线性频率空间转换为连续的 MIDI 表示
<!-- Conditioning a latent audio DiT on time-varying control signals -->
### 在 LDM 上使用时变控制信号
<!-- We use a large pre-trained text-to-sound latent diffusion trans-
former (DiT), similar to the one described in [2], [21] (text-
conditioned only, no timing conditioning) and adapt it to generate
sounds conditioned on the time-varying control signals mentioned
above. The latent diffusion model for text-to-sound generation has
two parts: first, a variational autoencoder (VAE) compresses 48kHz
mono audio to a sequence of continuous vectors of dim 64 at a rate of
40Hz. Then, a transformer decoder-only model is trained to generate
new sequences of latents, which can be decoded into audio using the
VAE decoder. This text-to-audio DiT was pre-trained on a large mix
of proprietary, licensed sound effect datasets and publicly available
CC-licensed general audio datasets. Once the model is pre-trained,
we fine-tune it for 40k steps and adapt it to handle our time-varying
control signals as conditioning -->
使用 [Stable Audio- Fast Timing-Conditioned Latent Audio Diffusion 笔记](Stable%20Audio-%20Fast%20Timing-Conditioned%20Latent%20Audio%20Diffusion%20笔记.md) 中的 DiT，