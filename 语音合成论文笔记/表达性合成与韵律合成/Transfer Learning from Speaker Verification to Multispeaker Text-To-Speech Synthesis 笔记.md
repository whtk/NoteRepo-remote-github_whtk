> NIPS 2018，Google

1. 提出了一个可以生成不同说话人（即使是 unseen）声音的 TTS 系统，包含三个独立训练的部分：
	1. speaker encoder network，在独立的数据集下进行说话人验证任务，来生成一个固定维度的 embedding 向量
	2. 基于 Tacotron 2 的 seq2seq 合成网络基于 speaker embedding 从文本生成 mel 谱
	3. 自回归的 WaveNet-based vocoder 网络将 mel 谱 转为波形

> 本质就是把 d-vector、Tacotron 2 和 WaveNet 三个模型组合起来，但是这里的 speaker encoder（也就是 d-vector 提取模型）是独立训练的，且训练的数据更多更广，从而可以在 zero-shot 的情况下也能实现较好的合成。

## Introduction

1. 提出，把说话人建模从语音合成中解耦出来，独立训练一个 speaker-discriminative embedding 网络，embedding 网络训练的时候判断两句话是否来自同一个说话人，从而可以在无文本标签的数据下训练
2. 本文使用预训练的说话人验证模型进行迁移学习
3. 本文采用端到端的合成网络，不不依赖于中间的语言特征

## 多说话人语音合成

包含三个独立训练的模型，如图：
![](image/Pasted%20image%2020231022104308.png)

### speaker encoder

采用 [Generalized End-to-End Loss for Speaker Verification 笔记](../../声纹识别论文阅读笔记/Generalized%20End-to-End%20Loss%20for%20Speaker%20Verification%20笔记.md)，将从语音中提取的 log-mel 帧序列映射为一个固定维度的 embedding 向量，即 d-vector，训练的时候用的是 generalized end-to-end speaker verification loss，来自相同说话人的语音的 embedding 之间的余弦相似度高，不同说话人的 embedding 则尽可能远离。

尽管这个模型没有直接学习说话人特征，但是发现在 speaker discrimination 任务下得到的 embedding 可以作为合成网络的条件。

### Synthesizer

根据 [Deep Voice 2- Multi-Speaker Neural Text-to-Speech 笔记](../Deep%20Voice%202-%20Multi-Speaker%20Neural%20Text-to-Speech%20笔记.md) 拓展 Tacotron 2 使其支持多说话人。也就是把 synthesizer encoder 的输出的每个 time step 都拼接上 embedding 向量。

比较了两个模型，一个采用的是 speaker encoder 得到的 embedding，另一个 baseline 在训练的时候为每个说话人学习一个固定维度的 embedding。

synthesizer 的输入为 phoneme，网络以迁移学习的配置进行训练，采用一个预训练的 speaker encoder （不更新参数）提取 speaker embedding，即 speaker reference signal 和  target speech 是一样的。

## Neural vocoder

采用 [WaveNet- A Generative Model for Raw Audio 笔记](../WaveNet-%20A%20Generative%20Model%20for%20Raw%20Audio%20笔记.md) 作为 vocoder 。

### Inference and zero-shot speaker adaptation

推理的时候，模型可以基于任何无文本标记的音频作为条件而无需和预测的文本相匹配，而且也可以把域外的说话人作为条件。

推理过程的一个例子如图：
![](image/Pasted%20image%2020231022110530.png)


## 实验（略）