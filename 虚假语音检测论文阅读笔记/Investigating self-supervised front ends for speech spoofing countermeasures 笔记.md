>The Speaker and Language Recognition Workshop 2022

1. 本文使用预训练的自监督语音模型作为欺骗CM的前端
2. 主要研究了 与自监督前端结合的不同后端架构、微调前端的有效性、不同预训练的自监督模型的性能 三个方面的问题
3. 结果表明，好的前端+浅层的神经网络后端都可以显著优于 baseline

## Introduction

1. 大多数传统前端依赖于数字信号处理（DSP）算法来提取频谱、相位或其他声学特征，如 LFCC 和 CQCC
	2. 也有论文使用可训练的 DNN 前端的，如 [[FastAudio- A Learnable Audio Front-End For Spoof Speech Detection 笔记]]、[[../经典模型和算法/Speaker Recognition from Raw Waveform with SincNet 笔记]] 等
2. 但是基于 DSP 的前端对不匹配域的鲁棒性有待改进，基于 DNN 的前端需要大量的真实和虚假语音数据来训练
3. 所以使用自监督模型作为前端，以自监督的方式提取特征，可以在任何语音数据库中进行训练，本文的研究内容有：
	1. 适合自监督前端的后端架构
	2. 是否应该针对反欺诈任务进行 fine-tune
	3. 最好的自监督模型是哪个

## 方法

### 自监督语音模型

本文侧重于 Wav2vec 2.0 和 HuBERT 模型。具体原理见论文：
1. [[../语音自监督模型论文阅读笔记/wav2vec 2.0- A Framework for Self-Supervised Learning of Speech Representations 笔记]]
2. [[../语音自监督模型论文阅读笔记/HuBERT- Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units 笔记]]

### 基于自监督前端的 CM

将自监督模型的输出 $c_{1:N}$ 送到后端生成 score $s \in \mathbb{R}$，这个过程考虑以下因素。

#### 后端架构

浅层网络足够作为后端。本文比较了三种后端，如图：![[Pasted image 20221230101654.png]]
1. 来自 baseline 的 LCNN + 两层 Bi-LSTM + GAP + FC，称为 LLGF
2. 删掉 LCNN，称为 LGF
3. 删掉 LSTM，称为 GF

注：自监督前端的后面都接了一个 FC 来减少特征维度。

#### fine-tune 还是 freeze

说法不一，本文都试验了。

#### 不同预训练自监督前端

用了这几个：![[Pasted image 20221230102315.png]]

## 实验

训练集：ASVspoof 2019 LA

在多个测试集中进行评估：ASVspoof 2019 LA, 2015 , 2021 LA,  2021 DF

baseline：60维的 LFCC，其他训练参数见论文

没有进行 VAD 和 特征归一化。

训练了三轮，每次用不同的种子。

### 结果

总体结果如图：![[Pasted image 20221230105227.png]]

1. 最适合自监督模型的后端：LLGF
2. 无论使用哪种后端，前端经过微调的性能都与固定前端的CM相似或更优，且收敛更快
3. 微调之后，后端的差异变得不那么显著了
4. 微调前端有助于CM改进已知和部分已知攻击的EER
5. 最好的预训练自监督前端（后端固定 LLGF）：W2V-Large2和W2V-XLSR，因为他们都使用来自不同语料库的语音数据进行了训练

## 子带分析

使用带阻滤波器来mask掉一部分的频率范围然后输入到 CM 系统中，观察mask前后的得分分布情况来分析不同频带的信息利用情况。

结果如图：![[Pasted image 20221230110711.png]]
1. 使用 LFCC + GF 的baseline 对子带滤波很敏感，尤其是 5.6 − 7.2 和 7.2 − 8.0 频段（单位 KHz）
2. 使用自监督前端的则不太敏感，但是在 0−0.8或0.8−2.4 kHz 上的得分影响较大

这说明 ，自监督前端CM主要依赖0.1到2.4kHz之间的信息。这和 baseline CM 不同。可能的原因是，自监督模型在各种语音数据上进行了预训练，并倾向于提取因素或高级语言信息，而忽略了信道变化。低频带可以是寻找期望的语言信息的良好区域。因此，使用自监督前端的CM倾向于关注该频带。

