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

用的是原始的 [Tacotron- Towards End-to-End Speech Synthesis 笔记](../Tacotron-%20Towards%20End-to-End%20Speech%20Synthesis%20笔记.md)，但是输入是通过 text normalization front-end 产生的 phoneme，而且也没用使用原始论文的 Bahdanau Attention，而是采用 GMM attention 来提高长语句的性能。

### Multi-speaker Tacotron

Tacotron 没有显式建模 speaker identity，采用 [Deep Voice 2- Multi-Speaker Neural Text-to-Speech 笔记](../Deep%20Voice%202-%20Multi-Speaker%20Neural%20Text-to-Speech%20笔记.md) 相似的结构来建模多说话人。

Tacotron 基于自回归的 decoder，其输入为 encoder 生成的 $L_T \times d_T$ 维的 phoneme or grapheme 序列表征，$L_T$ 为长度，$d_T$ 为 embedding 维度，对于数据集中的每个说话人，一个 $\mathbb{R}^{d_S}$ 维的 speaker embedding vector 通过 Glorot initialization 进行初始化，然后将这个 $d_S$ 维的 embedding 进行广播，得到 $L_T\times d_S$ 维，然后和前面的 representation 进行拼接，得到特征维度为 $d_T+d_S$，作为 decoder 的输入。

### Reference Encoder

在 Tacotron 中添加了一个 reference encoder，输入为长 $L_R$ 的维度为 $d_R$ 的参考信号，从中计算 $d_P$ 维的 embedding，把这个固定维度的 embedding 作为 prosody space，在此空间中采样可以得到 diverse 和 plausible 的语音。

同理，此 prosody embedding 也会和 $L_T \times d_T$ 维的 representation 广播后进行拼接，加上前面的 speaker embedding，最后得到 $L_T \times\left(d_T+d_S+d_P\right)$ embedding 矩阵，如图：
![](image/Pasted%20image%2020230831161941.png)

训练时，前面的参考信号就是要被建模的 target audio sequence（就是文本对应的音频的某些表征，不是音频！），通过采用重构误差作为损失函数进行训练。

推理的时候，使用 reference encoder 来编码任意音频（也就是说不一定要和 speaker embedding 或 text 匹配），从而可以实现韵律迁移。

reference encoder 的架构如图：
![](image/Pasted%20image%2020230831162543.png)
简单使用 6 层的卷积，每个卷积包含 $3\times 3$ 的 filter 和 $2\times 2$ stride，SAME padding 和 ReLU 激活 + Batch normalization。
得到的序列通过 128-width Gated Recurrent Unit (GRU) 层，将最后的 128 维的输出再通过 FC 层+激活函数投影到 $d_P$ 维。
> 注：最后选的 $d_P$ 的维度还是 128。

### Reference signal feature representation

reference encode 的输入 $L_{R}\times d_R$ 的选取很大程度上影响韵律。例如，如果选 pitch track representation，则无法建模 prominence（因为不包含能量），如果选 MFCC，则无法建模 intonation，本文选择的是 mel-warped spectrum。

### Variable-Length Prosody Embeddings

使用 固定长度的  prosody embedding 会有一个 scaling bottleneck，从而难以拓展到长语句。因此，可以把前面 GRU 中的每一层都用起来而不是只用最后一个，每个 GRU 都通过 FC 层来得到相应的维度。

实现发现，variable-length prosody embeddings 可以适用非常长的语句但是相比于 fixed-length embeddings，其鲁棒性不强。

## 实验