> ASRU 2021，Apple

1. 现有的 TTS 可以将 Tacotron 和 WaveRNN 组合实现很好的性能，但是无法用于 real-world 语音合成的应用
2. 本文提出一些模型改进和优化策略，使得可以在手机设别上部署这些模型
3. 可以在手机上 3x faster than real time 合成 24KHz 音频

## Introduction

1. 本文关注于包含两个 TTS 系统的神经网络：
	1. 第一个是 sequence-to-sequence attention based 模型，类似于 Tacotron 从 phoneme 中预测 mel 谱
	2. 第二个是自回归的 RNN，类似于 WaveRNN，基于 mel 谱产生样本
2. 描述了 Tacotron 的一些 challenge，同时解释了一些 WaveRNN 的优化和性能提升方法

## 先前的工作（略）

## 模型架构

包含两个独立训练的自回归的网络，Tacotron 和 WaveRNN。

### Tacotron

将带有标点和单词边界的 phoneme 序列作为输入，减少 mispronunciations 问题，同时可以通过标点学习正确的韵律和停顿，输出为 mel 谱，每次预测两帧。

原始的 Tacotron 2 采用 location-sensitive attention 来生成输入和输出之间的 soft alignment，作者观察到，这个 attention 并不总是鲁棒的，尤其是输入句子有重复的时候，于是采用 location sensitive monotonic attention 和  location sensitive stepwise monotonic attention 来引入 locality 和 monotonicity 性质。

实验发现  stepwise monotonic attention 效果最好。

### WaveRNN

在原始的 [WaveRNN- Efficient Neural Audio Synthesis 笔记](../WaveRNN-%20Efficient%20Neural%20Audio%20Synthesis%20笔记.md) 上做了两点修改：
+ 将输出信号从 16bit 用 u-law 压缩到 8 bit
+ hidden state 维度从 896 减少到 512

还采用 split-state 来进一步优化模型。

分析 WaveRNN 发现，采用 CUDA 来实现生成 coarse 和 ﬁne bits 只能有 2x faster than real time，这么慢的原因在于，大部分时间消耗在同步 WaveRNN 中不同的 kernel 上。fine bit 和 coarse bit 需要互相等待对方预测完成。

于是采用 pre-emphasis ﬁlter，然后预测 8-bit µ-law 量化可以实现性能和质量之间的 trade-off。

实验也发现，GRU 层的 hidden state 的维度可以减少。

### 带宽拓展至 48K

对 24K 音频做上采样得到 48K，实际上就是将 0-12K 的频谱镜像到 12-24K。

## 实验

重点在于工程上的实现。