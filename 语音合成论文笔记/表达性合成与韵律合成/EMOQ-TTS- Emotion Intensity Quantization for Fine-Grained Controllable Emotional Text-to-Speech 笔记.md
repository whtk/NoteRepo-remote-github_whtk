> ICASSP 2022，Korea University
<!-- 翻译 & 理解 -->
<!-- Although recent advances in text-to-speech (TTS) have shown significant improvement, it is still limited to emotional speech synthesis. To produce emotional speech, most works utilize emo- tion information extracted from emotion labels or reference audio. However, they result in monotonous emotional expression due to the utterance-level emotion conditions. In this paper, we propose EmoQ-TTS, which synthesizes expressive emotional speech by conditioning phoneme-wise emotion information with fine-grained emotion intensity. Here, the intensity of emotion information is rendered by distance-based intensity quantization without human la- beling. We can also control the emotional expression of synthesized speech by conditioning intensity labels manually. The experimen- tal results demonstrate the superiority of EmoQ-TTS in emotional expressiveness and controllability. -->
1. 现有的 TTS 在情感语音合成上的性能受限
2. 大多数方法使用情感标签或参考音频提取情感信息，但由于基于整个句子的情感条件，导致情感表达单调
3. 提出 EmoQ-TTS，把 phoneme-wise emotion 信息进行 fine-grained 的强度控制，来合成情感语音
    1. emotion 信息的强度通过 distance-based intensity quantization 实现，无需人工标注
    2. 也可以通过手动设置 intensity labels 来控制合成语音的情感

## Introduction
<!-- Recently, there has been significant improvement in end-to-end text- to-speech (TTS) systems [1, 2, 3, 4] due to the advancement of deep learning [5, 6]. Although synthesized speech from the current TTS model has already achieved outstanding performance, there still re- mains a limitation in synthesizing expressive speech with paralin- guistic features such as pitch, tone, and tempo. In particular, emo- tional speech synthesis is a challenging task since emotion informa- tion is affected by various paralinguistic characteristics of speech. -->
1. emotion 信息受到 pitch、tone 和 tempo 等 paralinguistic 特征的影响，情感语音合成很有挑战性
<!-- For emotional speech synthesis, the common approach is to con- dition global emotion information extracted from reference audio [7, 8] or emotion labels [9, 10]. However, these methods have a disadvantage where the synthesized speech has monotonous expres- sion since the whole sentence is regulated by only one global infor- mation. To generate expressive emotional speech similar to spon- taneous human speech, fine-grained emotional expressions accord- ing to emotion intensity should be considered at the phoneme-level. Several studies have attempted to reflect fine-grained emotional ex- pression by scaling [11, 12] or interpolating [13, 14] the represen- tative emotion embedding. Nevertheless, they have a problem with unstable audio quality, and it is also difficult to find proper parame- ters for scaling or interpolation. In the case of [15], the model pre- dicts phoneme-wise intensity scalar extracted from a learned ranking function [16]. However, this method tends to depend heavily on the global label, thus it is unstable to control the emotional expression based on intensity scalar. -->
2. 传统方法使用全局的情感信息，导致合成语音表现单调，fine-grained 的情感表达应该在 phoneme-level 上：
    1. 一些研究尝试通过缩放或插值情感 embedding 来反映 fine-grained 的情感，但音质不稳定，参数难以确定
    2. 也有模型预测从学习的 ranking function 中提取的 phoneme-wise intensity scalar，但是很难稳定控制情感表达
<!-- To address the above problems, this paper proposes EmoQ-TTS, which synthesizes expressive emotional speech by conditioning phoneme-wise emotion information based on fine-grained emotion intensity. To reflect appropriate emotional expression, we utilize intensity pseudo-labels and via distance-based intensity quantiza- tion without human labeling. EmoQ-TTS synthesizes speech more expressively by predicting appropriate emotion intensity from the text only. Furthermore, we can control emotion expression easily by conditioning intensity labels manually. The experimental re- sults show that our system successfully achieves better emotional expressiveness and controllability than conventional methods. The synthesized audio samples are available at https://prml-lab - speech- team.github.io/demo/EmoQ- TTS/ -->
3. 提出 EmoQ-TTS，通过 phoneme-wise emotion 信息进行 fine-grained 的情感强度控制，无需人工标注：
    1. 使用 intensity pseudo-labels 和 distance-based intensity quantization 来反映适当的情感
    2. 从文本预测情感强度
    3. 可以手动设置 intensity labels 来控制情感表达

<!-- EMOQ-TTS -->
## EMOQ-TTS

### Model Architecture
<!-- The entire architecture of EmoQ-TTS is depicted in Fig.1a. EmoQ- TTS is based on FastSpeech2 [17] which consists of an encoder, a decoder, and a variance adaptor. To synthesize fine-grained emo- tional speech, we modify FastSpeech2 architecture as follows: First, we introduce an emotion renderer to provide phoneme-level emo- tion information according to fine-grained emotion intensity. This enables all variance information, including the pitch, energy, and duration to be affected by the fine-grained emotion intensity. Sec- ond, the duration predictor is moved to the end of the variance adap- tor. This leads to all variance information to be processed at the phoneme-level, which has been proved better performance than the frame-level method in speech quality [18]. -->
EmoQ-TTS 如图：
![](image/Pasted%20image%2020240409105927.png)

基于 FastSpeech2，包括 encoder、decoder 和 variance adaptor，但是做了两点修改：
+ 引入 emotion renderer，根据 fine-grained 的情感强度提供 phoneme-level 的 emotion 信息，影响所有 variance 信息
+ 将 duration predictor 移动到 variance adaptor 的末尾，使所有 variance 信息在 phoneme-level 处理，比 frame-level 方法有更好的性能

