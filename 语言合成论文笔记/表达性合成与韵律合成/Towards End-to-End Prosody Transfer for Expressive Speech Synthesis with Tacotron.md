> Google，ICML 2018

1. 提出 Tacotron 的拓展架构，从包含特定韵律的声学表征中学习韵律的 latent embedding 空间
2. 把学习到的 embedding space 作为 Tacotron 的条件可以合成匹配参考信号韵律的语音（即使两个语音来自不同的说话人）
3. 定义了一些定量的和客观的评价指标来评估韵律的迁移效果

## Introduction

1. 为了合成真实的语音，TTS 系统必须显式或隐式地包含多种非文本因子，如 intonation（音调音色）, stress（重音）, rhythm（节律、旋律） 和 style of the speech（风格），这些都被合称为 prosody（韵律）
2. 基于文本的语音合成是一个不确定性问题，因为文本所能表达的东西并非是具象的；为了表达不同的信息，语音中的不同词汇可能有不同的重音，或者为了表达情绪，有不同的音高。音调也会包含环境或者上下文的信息
3. 于是想出通过从 GT 真实语音中学习 latent prosody representation，且表明，可以通过空间的迁移捕获到有意义的 variation，这有点像是 “say it like this”  的任务
4. Tacotron 的韵律建模是隐式的，本文则采用显式的韵律控制，通过学习一个 ecoder 从语音信号中计算低维的 embedding，其包含的信息不是文本和说话人身份相关的，实验证明，prosody embedding 能被用于产生需要的韵律

> Prosody 的定义：Prosody is the variation in speech signals that remains after accounting for variation due to phonetics, speaker identity, and channel effects (i.e. the recording environment). 韵律是语音信号中的可变量，在考虑了语音学、说话人身份和信道效应（即录音环境）等因素的变化后，仍然存在。


## 相关工作

韵律和风格的建模在 HMM-based 的时代就开始研究了。

然后讲了一些非 DNN 时代的韵律建模。

Prosody transfer 和 voice conversion（style transfer） 任务相关。

一个和本文相似的方法是 [Style Tokens- Unsupervised Style Modeling, Control and Transfer inEnd-to-End Speech Synthesis 笔记](Style%20Tokens-%20Unsupervised%20Style%20Modeling,%20Control%20and%20Transfer%20inEnd-to-End%20Speech%20Synthesis%20笔记.md)，但是是以无监督的方式使用复杂的 AutoEncoder 来学习 style。

## 模型架构

用的是原始的 Tacotron，但是输入是通过 text normalization front-end 产生的 phoneme，而且也没用使用原始论文的 Bahdanau Attention，而是采用 GMM attention 来提高长语句的性能。

### Multi-speaker Tacotron