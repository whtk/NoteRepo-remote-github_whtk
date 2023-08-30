> Baidu 硅谷 AI Lab，2017，ICML

1. 提出 Deep Voice，只基于 DNN 的 production-quality TTS 系统，包含五个主要模块：
	1. segmentation model：定位音素的边界，使用 CTC loss 来检测 phoneme boundary
	2. grapheme-to-phoneme conversion model
	3.  phoneme duration prediction model
	4. fundamental frequency prediction model
	5. audio synthesis model，采用 WaveNet  的变体，只需要更少的参数且训练更快
2. 模型的推理速度比 real time 快

## Introduction

1. 当前的 TTS 基于complex, multi-stage processing pipelines，每一步都依赖于 hand-engineered feature
2. Deep Voice 受传统的 TTS pipelines 启发，但是把模块都替换成了神经网络：首先将 text 转换为 phoneme，然后使用 audio synthesis model 将语言特征转换为语音
3. 唯一用的特征是带有重音标注、持续时间和 F0 的 phoneme，从而可以适应新的数据集而不需要额外的特征工程
4. WaveNet 的速度很慢，作者使用 faster-than-real-time WaveNet inference kernel 来产生高质量的 16kHz 音频且比原始的 WaveNet 提高了 400 倍的推理速度


## 相关工作

## TTS 系统组成

如图：
![](image/Pasted%20image%2020230830114230.png)
包含五个模块：
+ grapheme-to-phoneme model：将文本转换为 phoneme
+ segmentation model：定位语音数据集中的 phoneme 边界，也就是每个 phoneme 对应的音频片段
+ phoneme duration model：预测每个 phoneme 的持续时间
+ fundamental frequency model：预测 F0
+ audio synthesis model：将上面四个模型的输出组合起来合成音频

推理的时候，文本通过 grapheme-to-phoneme model 生成 phoneme，然后将 phoneme 输入到 phoneme duration model 和 F0 prediction model 中，预测每个 phoneme 对应的 duration，生成 F0 曲线，最后 phonemes, phoneme durations, 和 F0 作为条件输入特征输入到 audio synthesis model 中生成音频。
> segmentation model 在推理的时候并没有用到，其主要是为了在训练的时候用于生成标签来训练 duration model 的

下面描述每个模型的细节。

### Grapheme-to-Phoneme Model

模型基于 encoder-decoder 架构，采用的是 带有 GRU 激活的 multi-layer bidirectional encoder 和 深度相同的 unidirectional GRU decoder。

模型采用 teacher forcing  进行训练，采用 beam search 进行解码，具体的细节为：
+ 3 层 bidirectional layers with 1024 units each in the encoder
+ 3 unidirectional layers of the same size in the decoder 
+ beam search with a width of 5 candidates

### Segmentation Model

模型训练用于输出给定语音和 phoneme 训练之间的 alignment，类似于语音识别中的语音和文本的对齐（用的是 CTC loss），采用 ASR 中的 SOTA 方法，即 convolutional recurrent neural network 来作为模型结构，也用 CTC 作为损失函数。

同时为了解决用 CTC loss 出现的尖峰问题，预测的是 phoneme pairs：
> 如对于单词 ‘Hello!’，其对应的 phoneme 为 ‘“sil HH EH L OW sil’（sil 为静音），则 phoneme pairs 为 ‘(sil, HH), (HH, EH), (EH, L), (L, OW), (OW, sil)’ 

同时采用 beam search 来计算 phoneme-pair error rate。模型细节见论文。

### Phoneme Duration and Fundamental Frequency Model

采用单个架构来联合预测 phoneme duration 和 time-dependent fundamental frequency，输入为 带有重音标注的 phoneme 序列（one-hot 编码），架构为 两个 FC 层+两个unidirectional recurrent layers with 128 GRU cells + FC 输出层，对每个 phoneme，最后一层产生三个输出：
+ phoneme duration
  + the probability that the phoneme is voiced (i.e. has a fundamental frequency)
  + 20 time-dependent F0 value

损失为 phoneme duration error、fundamental frequency error 和 negative log likelihood of the probability。

### Audio Synthesis Model

音频合成模型为 WaveNet 的变体，WaveNet 包含：
+ conditioning network，对语言特征进行上采样
+ autoregressive network，生成音频样本的分布
作者改变了 层数 $l$，residual channels 数 $r$ 和 skip channels 数 $s$，把 WaveNet 中的卷积分成两个矩阵乘法，$W_{prev},W_{cur}$，然后通过 $W_{skip}$ 将每一层的向量投影到 skip  channel。

WaveNet 使用 transposed convolutions 进行上采样，但是作者发现使用 bidirectional quasi-RNN (QRNN) layers 效果更好，速度更快。

其他细节见论文。

## 结果（略）