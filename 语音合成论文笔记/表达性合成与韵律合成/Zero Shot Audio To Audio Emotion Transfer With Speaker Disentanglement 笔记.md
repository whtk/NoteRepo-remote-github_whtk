> ICASSP 2024，Indian Institute of Science
<!-- 翻译 & 理解 -->
<!-- The problem of audio-to-audio (A2A) style transfer involves replac- ing the style features of the source audio with those from the target audio while preserving the content related attributes of the source audio. In this paper, we propose an efficient approach, termed as Zero-shot Emotion Style Transfer (ZEST), that allows the trans- fer of emotional content present in the given source audio with the one embedded in the target audio while retaining the speaker and speech content from the source. The proposed system builds upon decomposing speech into semantic tokens, speaker representations and emotion embeddings. Using these factors, we propose a frame- work to reconstruct the pitch contour of the given speech signal and train a decoder that reconstructs the speech signal. The model is trained using a self-supervision based reconstruction loss. During conversion, the emotion embedding is alone derived from the target audio, while rest of the factors are derived from the source audio. In our experiments, we show that, even without using parallel train- ing data or labels from the source or target audio, we illustrate zero shot emotion transfer capabilities of the proposed ZEST model using objective and subjective quality evaluations. -->
1. audio-to-audio (A2A) 风格迁移用于将 source audio 的风格特征替换为 target audio 的风格特征，同时保留 source audio 的内容
2. 提出  Zero-shot 情感风格迁移 (ZEST)，将语音解耦为 semantic token、speaker representation 和 emotion embedding，训练一个框架，其给定语音信号来重构 pitch contour，然后训练一个 decoder 重构语音信号：
    1. 采用基于自监督的 reconstruction loss 训练模型
    2. 在转换过程中，emotion embedding 从 target audio 中获取，其他特征从 source audio 中获取
3. 实验表明，在没有平行训练数据下，模型也可以实现 zero shot 情感迁移

## Introduction
<!-- Artificial emotional intelligence [1] encompasses methods that en- able machines to understand and interact with human expressions of emotions. The style transfer approach to manipulating emotion, given a source and target data sample, is the task of converting emo- tion in the source sample to match the emotional style of the target sample while retaining rest of the attributes of the source. While the task has shown promising results in image domain [2], the applica- tions in audio domain is more challenging [3, 4]. In this paper, we explore the task of emotion style transfer in speech data. -->
1. 风格迁移是将 source sample 的情感转换为 target sample 的情感风格，同时保留 source 的其他属性
<!-- Voice conversion of speech primarily explored converting the speaker identity of a voice [5]. However, speech also contains infor- mation about the underlying emotional trait of the speaker in vary- ing levels [6]. The initial frameworks using Gaussian mixture model (GMM) [7], hidden Markov model [8] and deep learning [9] based conversion approaches have recently been advanced with generative adversarial networks (GAN) [10] and sequence-to-sequence auto en- coding models [11]. -->
2. 语音转换主要是转换说话者的身份，但语音也包含说话者的情感特征
<!-- In many of the prior emotion conversion approaches, the emo- tion targets are treated as discrete labels. However, emotion is a fine-grained attribute which has varying levels of granularities [6]. Forcing the emotion attribute to a small number of discrete target la- bels may not allow the models to capture the wide range of diverse and heterogeneous sentiments elicited in human speech. Hence, we argue that the most natural form of emotion conversion is to trans- fer the emotion expressed in a target audio to the source audio, a.k.a A2A emotion style transfer. This motivation is also echoed in a re- cent work on A2A style transfer [12]. In spite of these efforts, audio- to-audio (A2A) style transfer in zero shot setting (unseen speakers and emotion classes) is challenging. -->
3. 之前的情感转换方法中，emotion targt 被视为离散的 label，但是其实 emotion 是一个细粒度的属性，强制将 emotion 属性转换为离散 label 可能无法捕捉到人类语音的多样化情感，作者认为自然的情感转换是将 target audio 中的情感转移到 source audio 中，即 A2A 情感风格迁移
<!-- On a separate front, representation learning of speech has shown remarkable progress in the recent years. The wav2vec [13] models have been improved with masked language modeling (MLM) objec- tives (for example, HuBERT [14]) in self-supervised learning (SSL) settings. The derivation of speaker representations have mostly been pursued with a supervised model [15]. Further, reconstructing speech from factored representations of speaker, content and pitch contours [16] has shown that models like Tacotron [17], AudioLM [18] and HiFi-GAN [19] allow good quality speech generation. -->
4. speech 的 representation learning 取得了显著进展，wav2vec 模型通过 masked language modeling (MLM) 目标在自监督学习 (SSL) 设置中得到改进，speaker representation 主要通过监督模型进行，从 speaker、content 和 pitch contours 的因子表示中重构 speech 的模型表明 Tacotron、AudioLM 和 HiFi-GAN 可以生成高质量的 speech
<!-- In this paper, we propose a framework called, zero shot emotion transfer - ZEST, which leverages the advances made in representa- tion learning and speech reconstruction. The proposed framework decomposes the given audio into semantic tokens (using HuBERT model [14]), speaker representations (x-vectors [15]), and emotion embeddings (derived from a pre-trained emotion classifier). Inspired by speaker disentangling proposed for speech synthesis [20], we also perform a single step of speaker and emotion disentanglement in the embeddings. Since pitch (F0) is also a component that embeds content, speaker and emotion, we investigate a cross-attention based model for predicting the F0 contour of a given utterance. Using the three representations (speech, speaker and emotion) along with the predicted F0 contours, the proposed ZEST framework utilizes the HiFi-GAN [19] decoder model for reconstructing the speech. Dur- ing emotion conversion, the proposed ZEST approach does not use text or parallel training and simply imports the emotion embedding from the target audio for style transfer. -->
5. 提出 ZEST 框架，将给定的 audio 解耦为：
    + semantic tokens（使用 HuBERT model）
    + speaker representations（x-vectors）
    + emotion embeddings（从预训练的 emotion classifier 中获取），在 embeddings 中进行一步将 speaker 和 emotion 的解耦
6. 由于 F0 也包含 content、speaker 和 emotion 信息，使用 cross-attention 预测给定 utterance 的 F0 contour
7. 有了 speech、speaker 和 emotion 的三个表征和 F0 contour 后，用 HiFi-GAN decoder 重构语音，训练过程中，不使用文本或平行训练，只从 target audio 中引入 emotion embedding 进行风格转移
<!-- The experiments are performed on emotion speech dataset (ESD) [21]. We also explore a zero shot setting, where an unseen emotion from a different dataset is used as the reference audio. Fur- ther, a setting where the source speech is derived from an unseen speaker is also investigated. We perform several objective and sub- jective quality evaluations and compare with benchmark methods to highlight the style transfer capability of the proposed framework. The key contributions from this work can be summarized as follows, -->
8. 在 emotion speech dataset (ESD) 上进行实验，贡献如下：
<!-- Proposinganovelframeworkforpredictingthepitchcontourofa given audio file using the semantic tokens from HuBERT, speaker embeddings and emotion embeddings.
• Enabling speaker-emotion disentanglement using adversarial training.
• Illustrating zero shot emotion transfer capabilities from unseen emotion categories, novel speakers and content. -->
    + 提出使用 HuBERT 的 semantic tokens、speaker embeddings 和 emotion embeddings 预测给定音频的 pitch contour
    + 使用对抗训练实现 speaker-emotion 解耦
    + 实现 zero shot 情感迁移，包括未知的情感类别、新的说话人和新的文本情况

## 相关工作
<!-- A2A EST Using World Vocoder: One of the earliest attempts for EST involved using the world vocoder, as proposed by Gao et al. [22]. This work used the statistics of F0 and spectral components from the target speaker before reconstruction using the decoder. Our work uses recent advances in speaker, emotion and content embed- dings for emotion style transfer. Further, we aim to transfer the emotion style from the reference speech to the source speech sig- nal rather than just modifying the emotion category. -->
采用 World vocoder 的 A2A EST：早期使用 target speaker 的 F0 和 spectral components 的统计信息，这里则用了 speaker、emotion 和 content embeddings 进行情感风格迁移，目标是将 reference speech 的情感风格转移到 source speech 信号，而不仅仅是修改情感类别
<!-- Expressive text-to-speech synthesis: The work by Li et al. [20] explored using speaker disentanglement for generating emotional speech from text. Similarly, Emovox proposed by Zhou et al. [6] used phonetic transcription for emotional voice conversion. How- ever, our work explores EST without using any linguistic or phonetic transcriptions of the source or target speech. -->
表达性 TTS：有些工作用 speaker disentanglement 从文本生成情感语音，Emovox 使用 phonetic transcription 进行情感语音转换，这里则不使用 source 或 target speech 的任何语言或 phonetic transcription
<!-- Non-parallel and unseen emotion conversion: Recent work by Chen et al. [12] explored using attention models for performing EST. However, this work forced the source and target speech to be from the same speaker, limiting the utility of the EST applications. -->
非平行和未知情感转换：有使用 attention 模型进行 EST，但是要求 source 和 target speech 来自同一个说话者，限制了 EST 的应用

## 方法

### Content encoder
<!-- The content encoder used in the proposed framework is the HuBERT SSL model [14]. The HuBERT model gives continuous valued vec- tor representations for each speech segment, which is subsequently converted into discrete tokens with a k-means clustering. -->
使用 [HuBERT- Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units 笔记](../../语音自监督模型论文阅读笔记/HuBERT-%20Self-Supervised%20Speech%20Representation%20Learning%20by%20Masked%20Prediction%20of%20Hidden%20Units%20笔记.md) 模型作为 content encoder，对每个 speech segment 生成 vector representations，然后用 k-means 聚类转为离散 tokens。

### 情感无关的 speaker encoder
<!-- The speaker embeddings for each audio file are extracted using a enhanced channel attention-time delay neural network (ECAPA- TDNN) [15] model. This model is pre-trained on 2794 hours and 7363 speakers from the VoxCeleb dataset [23], for the task of speaker classification. The model involves an utterance level pooling of the frame-level embeddings, called x-vectors. The x-vectors have been shown to encode emotion information [24, 25]. In order to suppress the emotion information, inspired by the disentanglement approach proposed in Li et al. [20], we add two fully connected layers to the x-vector model and further train the model with an emotion adversarial loss [26]. We refer to these vectors as the Emo- tion Agnostic Speaker Encoder (EASE) vectors. The loss function is given by -->
使用 [ECAPA-TDNN- Emphasized channel attention, propagation and aggregation in tdnn based speaker verification 笔记](../../声纹识别论文阅读笔记/ECAPA-TDNN-%20Emphasized%20channel%20attention,%20propagation%20and%20aggregation%20in%20tdnn%20based%20speaker%20verification%20笔记.md) 模型提取音频的 speaker embeddings（模型在 VoxCeleb 数据集上进行了预训练，会对 frame-level embeddings 进行 utterance level pooling，得到 x-vectors），为了 suppress emotion 信息，添加两个全连接层到 x-vector ，并使用 emotion 对抗损失进行训练，得到所谓的 Emotion Agnostic Speaker Encoder (EASE) vectors。损失函数为：
$$\large\mathcal{L}_{tot-spkr}=\mathcal{L}_{ce}^{spkr}-\lambda_{adv}^{emo}\mathcal{L}_{ce}^{emo}$$

<!-- SpeakerAdversarialClassifierofEmotions(SACE) -->
### Speaker Adversarial Classifier of Emotions(SACE)
<!-- A speaker adversarial classifier of emotions (SACE) classifier is de- signed based on the wav2vec2.0 representations [13], similar to the one proposed by Pepino et al. [27]. The wav2vec model, pre-trained on 300 hours (543 speakers) of switchboard corpus [28], is used for extracting features from the raw speech signal [29]. The con- volutional feature extractors are kept frozen while the transformer layers along with two position wise feed forward layers are trained for the task of emotion recognition on the Emotional Speech Dataset (ESD) [21]. The model is trained with speaker adversarial loss (the emotion classifier equivalent of Eq. 1). The representations averaged over the entire utterance are used as the emotion embedding. -->
基于 wav2vec2 表征设计 speaker adversarial classifier of emotions (SACE) 分类器，使用预训练的 wav2vec 模型提取原始语音波形的特征，其中的 CNN 特征提取器保持冻结，transformer layers 和两个 position wise FF 层在 Emotional Speech Dataset (ESD) 进行情感识别训练，采用 speaker adversarial loss 进行训练，整个 utterance 的平均表征作为 emotion embedding。

### Pitch Contour Predictor
<!-- The framework for the pitch (F0) predictor is shown in Figure 1. The HuBERT tokens for the speech signal are converted to an se- quence of vectors by means of an embedding layer. This sequence, denoted by Q = {q1 , .., qT }Tt=1 , is used as the query sequence for cross-attention. The frame-level SACE embeddings are added with speaker embedding (EASE) to form the key-value pair for the cross attention module [30]. This is followed by a 1D-CNN network to predict the F0 contour. The target pitch contour is the one derived using the YAAPT algorithm [31]. We use the L1 loss between the predicted and target F0 contour. -->
pitch (F0) 预测器框架如下：
![](image/Pasted%20image%2020240410163946.png)

语音信号的 HuBERT token 通过 embedding 层转为向量序列，记为 $Q=\{q_1,...,q_T\}_{t=1}^T$，作为 cross-attention 的 query，frame-level SACE embeddings 和 speaker embedding (EASE) 作为 cross attention 的 key-value 对，然后通过 1D-CNN 网络预测 F0 contour，目标 F0 contour 使用 YAAPT 算法得到，使用预测和目标 F0 contour 之间的 L1 loss。

### 语音重构
<!-- The speech reconstruction framework is shown in Fig. 2(a). For re- constructing the speech signal, the HuBERT tokens, SACE embed- ding, EASE vector and the predicted F0 contour are used. The Hu- BERT tokens are converted to a sequence of real-valued vectors with a learnable embedding layer. In order to add contextual information during the speech reconstruction phase, the tokens and F0pred are passed through two separate networks consisting of CNN layers and bidirectional long-short term memory (BLSTM) layers. Finally, all the components are passed through a HiFi-GAN vocoder [19] to re- construct the speech signal. More details of the HiFi GAN model are provided in Polyak et al. [16]. -->
语音重构框架如下图 a：
![](image/Pasted%20image%2020240410164230.png)

用 HuBERT tokens、SACE embedding、EASE vector 和预测的 F0 contour 来进行重构。HuBERT tokens 通过可学习的 embedding 层转为向量序列。为了在重构阶段添加上下文信息，tokens 和预测的 F0 分别通过两个网络（CNN 和 BLSTM），最后通过 HiFi-GAN vocoder 重构。

### 情感转换
<!-- The ZEST framework for emotion conversion is shown in Fig- ure 2(b). The HuBERT tokens and the speaker vector are extracted from the source speech while the emotion embeddings are derived from the target speech. The emotion embedding sequence, being extracted from the reference speech, may differ in length from the source speech. However, as the query sequence in the cross-attention (Figure 1) is driven by the HuBERT tokens of the source signal, the F0 contour generated during conversion will match the length of the source signal. The HuBERT tokens, speaker vector, predicted F0 contour and the emotion embedding are then used to generate the converted speech through the pre-trained HiFi-GAN model. The conversion phase does not involve any model training steps. -->
情感转换框架如上图 b。流程如下：
+ 从 source speech 中提取 HuBERT tokens 和 speaker vector ，从 target speech 中提取 emotion embedding
> 由于 emotion embedding sequence 从 reference speech 中提取，可能与 source speech 的长度不同，但是 cross-attention 的 query sequence 由 source signal 的 HuBERT tokens 驱动，所以转换过程中生成的 F0 contour 与 source signal 的长度匹配。
+ HuBERT tokens、speaker vector、predicted F0 contour 和 emotion embedding 输入到预训练的 HiFi-GAN 来生成转换后的语音

## 实验和结果（略）
