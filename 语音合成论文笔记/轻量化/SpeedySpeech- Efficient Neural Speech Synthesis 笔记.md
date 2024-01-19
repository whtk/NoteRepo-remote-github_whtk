> Interspeech 2020，Charles University

1. 提出 student-teacher 网络，可以实现高质量的、faster-than-real-time 的 spectrogram 合成
2. 表明 self-attention 层对于生成高质量的音频不是必须的
3. 在 student 和 teacher 模型上使用简单的带残差的卷积网络，teacher 模型中使用单层的 attention layer
4. 采用 MelGAN 作为 vocoder 时，合成的质量比 Tacotron 2 好很多

## Introduction

1. 本文关注在推理速度和硬件需求上提高 TTS 的效率，同时保持较好的合成质量
2. 提出全卷积网络，包含 teacher 和student 网络：
	1. teacher 网络是一个自回归的卷积网络，用于提取 phoneme 和 音频帧之间的对齐
	2. student 网络是非自回归的、全卷积的网络，编码输入 phoneme，预测 duration，解码得到 spectrogram
	3. 可以在  8GB GPU 下训练 40小时

## 相关工作（略）

## 模型

模型输入为 phoneme，输出为对数坐标下的 mel 谱。

### Teacher 网络——提取 duration

teacher 网络基于[Deep Voice 3- Scaling Text-to-Speech with Convolutional Sequence Learning 笔记](../Deep%20Voice%203-%20Scaling%20Text-to-Speech%20with%20Convolutional%20Sequence%20Learning%20笔记.md) 和 [DCTTS- Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention 笔记](DCTTS-%20Efficiently%20Trainable%20Text-to-Speech%20System%20Based%20on%20Deep%20Convolutional%20Networks%20with%20Guided%20Attention%20笔记.md) ，包含四个部分：
+  phoneme encoder
+ spectrogram encoder
+ attention
+ decoder

如图：
![](image/Pasted%20image%2020240118150538.png)

模型自回归地根据 phoneme 和上一个 frame 预测下一个 frame，具体来说：
+ phoneme encoder 输入为 embedding，结构为卷积+ReLU，还有一些 residual 模块由 dilated non-causal convolutions 组成，没有用 DCTTS 中的 highway 模块而是简单的卷积，但是性能没有显著下降
+ spectrogram encoder 提供上下文的 spectrogram frame。首先包含 FC + ReLU 层，然后一些 residual 模块由 dilated gated causal convolutions 组成
+ attention 用的是 dot-product attention，phoneme encoder 输出为 K，phoneme encoder 输出 + phoneme embedding 为 V，spectrogram encoder 输出为 Q，KQ 通过位置编码和  identical linear layer 来使得 attention 趋于单调
+ decoder 同样包含 residual 模型由 dilated causal convolutions 组成，最后通过 sigmoid 层

训练的时候，为了用上 sigmoid 层的输出，将对数 mel 谱 缩放到 $[0,1]$ 区间。

最小化 spectrograms 之间的 MAE loss + guided attention loss。

为了提高鲁棒性，还对 spectrogram 采用了数据增强：
+ 添加小的高斯噪声
+ 将输入 spectrogram 并行输入到模型（而非之前的 sequentially）
+ 通过随机替换一些帧为其他帧（让模型可以看到之前的稍远一点的帧，以避免过拟合）

推理时，以 teacher forcing 的方式进行（输入的是 GT frame），得到的 attention matrix 用于提取 duration。

### student 网络——spectrogram 分析

结构如图：
![](image/Pasted%20image%2020240118152931.png)

给定 phoneme，先预测 duration，然后再预测 mel 谱，包含  phoneme encoder, duration predictor 和 decoder。

三个模型都包含 dilated residual convolutional blocks。其中 phoneme encoding vectors 会根据 predicted duration 进行拓展来匹配 mel 谱的长度，中间添加了 positional encoding。

结构和 [FastSpeech- Fast, Robust and Controllable Text to Speech 笔记](../FastSpeech-%20Fast,%20Robust%20and%20Controllable%20Text%20to%20Speech%20笔记.md) 很像，但是把 attention 模块替换为卷积了。

训练的时候，对于对数 mel 谱 帧，用的是 MAE 和 structural similarity index (SSIM) loss；对于 duration 用的是 huber loss，训练的时候用的是 teacher 模型提取的 GT duration 进行拓展。duration predictor 的梯度不会传到 encoder 来避免过拟合。

## 实验（略）