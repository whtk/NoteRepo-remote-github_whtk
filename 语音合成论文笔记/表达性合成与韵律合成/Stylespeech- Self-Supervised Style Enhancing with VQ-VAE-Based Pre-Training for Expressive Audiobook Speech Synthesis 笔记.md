> ICASSP 2024，CUHK、清华、微软
<!-- 翻译 & 理解 -->
<!-- The expressive quality of synthesized speech for audiobooks is lim- ited by generalized model architecture and unbalanced style dis- tribution in the training data. To address these issues, in this pa- per, we propose a self-supervised style enhancing method with VQ- VAE-based pre-training for expressive audiobook speech synthesis. Firstly, a text style encoder is pre-trained with a large amount of un- labeled text-only data. Secondly, a spectrogram style extractor based on VQ-VAE is pre-trained in a self-supervised manner, with plenty of audio data that covers complex style variations. Then a novel ar- chitecture with two encoder-decoder paths is specially designed to model the pronunciation and high-level style expressiveness respec- tively, with the guidance of the style extractor. Both objective and subjective evaluations demonstrate that our proposed method can ef- fectively improve the naturalness and expressiveness of the synthe- sized speech in audiobook synthesis especially for the role and out- of-domain scenarios.1 -->
1. 用于 audiobooks 的语音合成的表达质量受限于 模型架构 和训练数据中不平衡的风格分布
2. 提出基于 VQ-VAE 预训练的自监督风格增强方法，实现 expressive audiobook 语音合成：
    1. 先用大规模的无标签的文本数据预训练 style encoder
    2. 以自监督的方式，预训练基于 VQ-VAE 的 spectrogram style extractor，训练数据是包含各种 style 的
    3. 设计含有两个 encoder-deocder 路径的架构，分别用于建模发音和 high-level style expressiveness
3. 方法可以提高 audiobook 合成中的自然度和表达性，对 out-of-domain 场景尤其有效

## Introduction
<!-- Recent text-to-speech (TTS) models, e.g., Tacotron 2 [1], Trans- formerTTS [2], FastSpeech 2 [3], have been developed with the capability to generate high-quality speech with a neutral speaking style. However, limited expressiveness persists as one of the ma- jor gaps between synthesized speech and real human speech, which draws growing attention to expressive speech synthesis studies [4, 5, 6, 7]. Synthesizing long-form expressive datasets, e.g., audiobooks, is still a challenging task, since wide-ranging voice characteristics tend to collapse into an averaged prosodic style. -->
1. 合成 long-form expressive 语音，如 audio books，仍然是一个挑战，因为各种声音特征会坍缩成平均的 prosodic 风格
<!-- There are a lot of works focusing on audiobook speech synthe- sis [8, 9, 10]. Recently, [11] proposes to use the neighbor sentences to improve the prosody generation. To make better use of contex- tual information, a hierarchical context encoder that considers ad- jacent sentences with a fixed-size sliding window is used to predict a global style representation directly from text [12]. Besides, [13] tries to consider as much information as possible (e.g., BERT em- beddings, text embeddings and sentence ID) to improve style pre- diction. On top of these, a multi-scale hierarchical context encoder is proposed to predict both global-scale and local-scale style embed- dings from context in a hierarchical structure [14]. All these existing works mainly focus on how to use the semantic information of con- textual text to predict the expressiveness through an additional style encoder module. Too much information (phoneme, timbre, style, etc.) is simply mixed in the encoder part, leading to challenges for mel-spectrogram decoder. In addition, another serious problem for audiobook synthesis is the unbalanced style distribution in audio- book dataset. Most sentences are relatively plain narration voices, and only a small part is role voices with rich style variations, which brings a great challenge to modeling of style and expressiveness rep- resentation with limited audiobook training data, especially for role and out-of-domain scenarios. -->
2. 现有的方法主要集中在，通过额外的 style encoder，使用上下文中的 semantic 信息来预测 expressiveness
3. 问题：
    1. encoder 部分混合了太多信息，导致 mel-spectrogram decoder 很难做
    2. audiobook 中风格分布不平衡，大部分是平淡的叙述声音，只有少部分是有风格变化的声音
<!-- To solve the above-mentioned poor expressiveness problem in audiobook speech synthesis caused by generalized model architec- ture and unbalanced style distribution in the training data, this paper proposes a self-supervised style enhancing method with VQ-VAE- based pre-training for expressive audiobook synthesis. Firstly, a text style encoder is pre-trained with the help of a large amount of easily obtained unlabeled text-only data. Secondly, a spectrogram style ex- tractor based on VQ-VAE is pre-trained using plenty of audio data that covers multiple expressive scenarios in other domains. On top of these, a special model architecture is designed with two encoder- decoder paths with the guidance of style extractor. To summarize, the main contributions of this paper are: -->
4. 提出基于 VQ-VAE 预训练的自监督风格增强方法，贡献如下：
<!-- WeproposeaVQ-VAE-basedstyleextractortomodelabetter style representation latent space and relieve the unbalanced style distribution issues, which is pre-trained by plenty of eas- ily obtained audio data that can cover complex style varia- tions in a self-supervised manner.
• We design a novel TTS architecture with two encoder- decoder paths to model the pronunciation and high-level style expressiveness respectively, so as to enrich the expres- sive variation of synthesized speech in complex scenarios by strengthening both the encoder and decoder of TTS model.
• Both objective and subjective experimental results show that our proposed style enhancing approach achieves an effective improvement in terms of speech naturalness and expressive- ness especially for the role and out-of-domain scenarios. -->
    1. 提出基于 VQ-VAE 的 style extractor，来建模 style representation latent space
    2. 设计了一个新的 TTS 架构，包含两个 encoder-decoder 路径，分别用于建模发音和 high-level style expressiveness
    3. 实验结果表明，方法可以提高自然度和表达性，尤其对 role 和 out-of-domain 场景有效

## 相关工作

基于 [Self-supervised Context-aware Style Representation for Expressive Speech Synthesis 笔记](Self-supervised%20Context-aware%20Style%20Representation%20for%20Expressive%20Speech%20Synthesis%20笔记.md) 和  [VQ-VAE- Neural Discrete Representation Learning 笔记](../../语音自监督模型论文阅读笔记/VQ-VAE-%20Neural%20Discrete%20Representation%20Learning%20笔记.md)。

## 方法

