> ICLR 2023，Meta AI
<!-- 翻译&理解 -->
<!-- We tackle the problem of generating audio samples conditioned on descriptive
text captions. In this work, we propose AUDIOGEN, an auto-regressive generative
model that generates audio samples conditioned on text inputs. AUDIOGEN op-
erates on a learnt discrete audio representation. The task of text-to-audio genera-
tion poses multiple challenges. Due to the way audio travels through a medium,
differentiating “objects” can be a difficult task (e.g., separating multiple people
simultaneously speaking). This is further complicated by real-world recording
conditions (e.g., background noise, reverberation, etc.). Scarce text annotations
impose another constraint, limiting the ability to scale models. Finally, model-
ing high-fidelity audio requires encoding audio at high sampling rate, leading to
extremely long sequences. To alleviate the aforementioned challenges we pro-
pose an augmentation technique that mixes different audio samples, driving the
model to internally learn to separate multiple sources. We curated 10 datasets
containing different types of audio and text annotations to handle the scarcity of
text-audio data points. For faster inference, we explore the use of multi-stream
modeling, allowing the use of shorter sequences while maintaining a similar bi-
trate and perceptual quality. We apply classifier-free guidance to improve adher-
ence to text. Comparing to the evaluated baselines, AUDIOGEN outperforms over
both objective and subjective metrics. Finally, we explore the ability of the pro-
posed method to generate audio continuation conditionally and unconditionally.
Samples: https://felixkreuk.github.io/audiogen. -->

1. 提出 AudioGen，根据输入文本生成语音的自回归生成模型，基于离散的语音标准
2. 提出增强方法，混合不同的音频样本，使模型学习多个 source
3. 使用 10 个数据集来缓解额文本-音频数据稀缺问题
4. 使用 multi-stream modeling 来加速推理，实现更短的序列同时保持相似的比特率和感知质量

## Introduction
<!-- Neural generative models have challenged the way we create digital content. From generating high-
quality images (Karras et al., 2019; Park et al., 2019) and speech (Ren et al., 2021; Oord et al.,
2016), through generating long textual spans (Brown et al., 2020; Zhang et al., 2022), to the recently
proposed text prompted image generation (Ramesh et al., 2022; Rombach et al., 2022), these mod-
els have shown impressive results. This begs the question what would be the audio equivalent to
textually guided generative models? From generating soundscapes to music or speech, a solution to
this problem that is high fidelity, controllable, and diverse in its outputs, would be a useful addition
to the modern toolbox of creators of movies, video games, and any virtual environments. -->
<!-- While image generation and audio generation have a lot in common, there are a few key differ-
ences. Audio is intrinsically a one dimensional signal and thus has less degrees of freedom to
differentiate overlapping “objects” (Capon, 1969; Frost, 1972). Real-world audio inherently has re-
verberations, which makes the task of differentiating objects from the surrounding environment even
harder. Moreover, psychoacoustic and psychovisual properties differ, for instance hearing “resolu-
tion” (equal-loudness) is U-shaped in frequencies with a dip at 4kHz and bump at 8kHz (Suzuki
et al., 2003). Last but not least, the availability of audio data with textual descriptions is orders of
magnitude below that of text-image paired data. This makes generating unseen audio compositions a
hard task (e.g. generating an audio equivalent of an image of “an astronaut riding a horse in space”). -->
1. 图像生成和音频生成的一些区别：
    + 音频是一维信号，对于 overlap 的区分更困难
    + 现实世界的音频具有混响，区分起来更加困难
    + 心理声学和心理视觉特性不同，听觉“分辨率”（等响度）在频率上呈 U 形，4kHz 有一个低谷，8kHz 有一个高峰
    + 文本描述的音频数据远远少于文本-图像配对数据
<!-- In this work, we tackle the problem of generating audio samples conditioned on descriptive text
captions. We additionally extend the proposed method to conditional and unconditional audio con-
tinuation. Here, we generate “a dog barks while somebody plays the trumpet in a busy street”. In the
above prompt, the model must generate three categories of acoustic content, with varying degrees
of background/foreground, durations, and relative position in the temporal axis, the composition of
which is highly unlikely to be present in the training set. Generating such audio is thus a challenge
in generalization, acoustic fidelity, production and mastering. -->
<!-- We propose AUDIOGEN, an autoregressive textually guided audio generation model. AUDIO-
GEN consists of two main stages. The first encodes raw audio to a discrete sequence of tokens
using a neural audio compression model (e.g. Zeghidour et al. (2021)). This model is trained in
an end-to-end fashion to reconstruct the input audio from the compressed representation, with an
addition of a perceptual loss in the form of a set of discriminators. Such an audio representation
is designed to generate high-fidelity audio samples while still being compact. The second stage,
leverages an autoregressive Transformer-decoder language-model that operates on the discrete au-
dio tokens obtained from the first stage while also being conditioned on textual inputs. We represent
text using a separate text encoder model pre-trained on a large corpus of text, namely T5 (Raffel
et al., 2020). The pre-trained text encoder enables the generalization to text concepts that are absent
from current text-audio datasets. This is especially important when working with text annotations
limited in terms of diversity and descriptiveness. -->
2. 本文提出 AudioGen，包含两个阶段：
    + 一：使用 codec 模型将音频编码为离散 token 序列
    + 二：使用 Transformer-decoder LM 模型基于离散音频 token 和文本来生成音频
<!-- Compared to the existing text-to-audio work (Yang et al., 2022), AUDIOGEN generates samples
that obtain better objective and subjective metrics. In particular, AUDIOGEN creates more natural
sounding unseen audio compositions. Lastly, we empirically show how the proposed approach can
be extended to audio continuation considering both conditional and unconditional generation. -->
<!-- Our contributions: (i) We propose a state-of-the-art auto-regressive audio generation model condi-
tioned on textual descriptions or audio prompts, as evaluated with objective and subjective (human
listeners) scores. Specifically we propose two model variations, one with 285M parameters and
another one with 1B parameters; (ii) We improve text-to-audio generation in two axes. We im-
prove text adherence by applying classifier free guidance on top of the audio language model. We
improve compositionality by performing on the fly text and audio mixing; (iii) We show that the
proposed approach can be extended to audio continuation conditioned on text and unconditionally;
(iv) We explore the trade-off between audio-fidelity and sampling time by utilizing residual vector
quantization (for acoustic units) and multi-stream transformers. -->

## 相关工作（略）

## 方法
