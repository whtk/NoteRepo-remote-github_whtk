> Baidu 硅谷 AI Lab，ICLR ，2018

1. 提出 Deep voice 3，一个 fully-convolutional attention-based TTS 模型
2. 模型在 800 个小时、多于两千个说话人的数据集中训练

## Introduction

1. 本文提出一个新的、fully-convolutional 的架构，将其拓展到超大的语音数据集中，且指出了在真实场景下部署 TTS 系统的一些问题
2. 具体贡献如下：
	1. 提出 fully-convolutional character-to-spectrogram 架构，可以实现完全的并行计算，速度比训练网络快
	2. 在 LibriSpeech ASR dataset 这种大数据集上训练很快（820小时，2484个说话人）
	3. 可以生成 monotonic attention，从而避免了一些 seq2seq 架构的问题
	4. 和 WORLD、Griffin-Lim、WaveNet 等的合成质量进行了比较
	5. 描述了 Deep Voice 3 的推理内核的实现，可以在单个 GPU 上每天可提供多达一千万次的 query

## 相关工作（略）

## 模型架构

提出的架构可以将很多种的文本特征（如 characters, phonemes, stresses）转换成不同的声码器参数（如 el-band spectrograms, linear-scale log magnitude spectrograms, fundamental frequency, spectral envelope, and aperiodicity parameters。

Deep Voice 3 的架构包含 3 个部分：
+ encoder：fully-convolutional encoder，将文本特征转换为表征
+ decoder：fully-convolutional causal decoder，将前面的表征通过 multi-hop convolutional attention mechanism 以自回归的方式解码到低维的声学表征（mel 谱）
+ converter：fully-convolutional 的后处理网络，用于从 decoder 的 hidden state 中预测最终的 vocoder 参数；和 decoder 不同，converter 是 non-causal 的，可以看到未来的信息

![](image/Pasted%20image%2020230902114710.png)

总的目标函数是 decoder 和 converter 的线性组合，且 decoder 和 converter 是分开训练的（多任务训练）。

在多说话人时，用了 [Deep Voice 2- Multi-Speaker Neural Text-to-Speech 笔记](Deep%20Voice%202-%20Multi-Speaker%20Neural%20Text-to-Speech%20笔记.md) 中的可训练的 speaker embedding。

### 文本预处理

直接把 characters 作为输入效果也还行，但是对一些稀有单词可能发音错误或者会跳过或重复某些单词，于是对输入的文本进行如下的归一化：
1. 所有的 character 都转为大写
2. 移除所有标点
3. 每段话都以句号或者问号结束
4. 单词之间的空格根据 duration 用四种 separator characters 来分割（快 -> 慢）：
	1. 连字符
	2. 标准空格
	3. 短停顿
	4. 长停顿
> 举个例子：Either way, you should shoot very slowly 可以转换为 Either way%you should shoot/very slowly%，其中 % 为长停顿，/ 为短停顿，以及普通的 空格

### CHARACTERS 和 PHONEMES 的联合表征

由于模型可以直接将 character 转为声学特征，从而可以隐式地学习 grapheme-to- phoneme 模型，但是这种隐式的转换在出错时很难纠正，于是除了 character 模型，还训练一个 phoneme 模型和 混合 character-and-phoneme 模型，从而使得 phoneme 可以显式的作为输入。

实验发现，支持 phoneme 表征的模型可以纠正使用 phoneme 词典导致的发音错误。

### 用于 SEQUENTIAL PROCESSING 的卷积模块

通过堆叠多层卷积网络实现足够大的感受野，能够充分利用长时的上下文信息，而不需要引入额外的依赖。

采用下图所示的卷积模块：
![](image/Pasted%20image%2020230902114950.png)

包含： 1-D convolution filter + gated-linear unit + residual connection。

同时 speaker-dependent embedding 作为 convolution filter 的 bias 项，采用 softsign 作为非线性函数，因为其可以限制输出的范围同时也可以避免指数的饱和问题。

卷积可以是 causal 和 non-causal 的。

### ENCODER

encoder 网络从 embedding 层开始，将 character  或 phoneme 转换为可训练的 向量表征 $h_e$，通过 FC 层从 embedding 维度投影到 target 维度，然后通过一系列的卷积模块提取文本信息，然后再投影回 embedding 维度得到 attention 的 key $h_k$，但是这里的 value 和 key 不是一样的，value 为 $h_v=\sqrt{0.5}(h_k+h_e)$，最终的 context vector 为 value vector 的加权和。

### DECODER

decoder 以自回归的方式生成音频，每次预测一组 $r$ 帧，由于是自回归的，其卷积也是 causal 的。采用 mel-band log-magnitude spectrogram 作为音频表征，。

decoder 首先采用 pre-net 处理 mel 谱 （pre-net 包含 FC+ReLU层），然后通过一系列的  causal convolution 和 attention blocks。convolution blocks 生成 query，最后通过 FC 层预测 $r$ 帧 和一个 二元的 final frame 预测（判断是不是最后一帧）。

mel 谱预测用的是 L1 损失，final frame 用的是交叉熵损失。

### Attention 模块

采用下图的 dot-product attention：
![](image/Pasted%20image%2020230904103508.png)
最终生成一个 context vector（一个 query 生成一个，多个 query 生成多个）。

而且发现，通过引入 inductive bias，可以实现 monotonic progression in time（其实就添加 positional embedding）。不过有一个不同就是，对于不同的 speaker，positional embedding 中的系数 $w_s$ 是不同的，具体可以见上图。

通过采用 monotonic attention mechanism（来自论文 [Online and Linear-Time Attention by Enforcing Monotonic Alignments 笔记](对齐/Online%20and%20Linear-Time%20Attention%20by%20Enforcing%20Monotonic%20Alignments%20笔记.md)），可以缓解一些 repeating or skipping words 的问题，但是会导致一个 diffused attention distribution.。原因是 soft alignment 的 attention coefficients 是没有归一化的，于是提出在仅在推理阶段约束 attention weights 为 monotonic，从而可以提高合成的质量。

### Converter

converter  把 decoder 的最后一个 hidden state 作为输入，通过一些 non-causal convolution blocks 预测下游 vocoder 的参数。
> converter 是 non-caual 和 non-autoregressive 的。

基于使用的 vocoder不同，这个模块的损失函数也不同。

## 结果（略）
