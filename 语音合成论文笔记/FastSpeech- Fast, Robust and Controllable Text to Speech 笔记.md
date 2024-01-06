> NIPS 2019，ZJU，MSRA

1. 提出新的基于 Transformer 的网络并行生成 mel 谱
2. 从 encoder-decoder based teacher model 中提取 alignment，用于 phoneme duration 的预测，phoneme duration 用于 length regulator 来匹配 phoneme 序列和 mel 谱 序列的长度
3. 模型可以 match 自回归模型，在一些 hard case 下可以几乎消除 word skipping 和 repeating 的问题，同时还可以平滑地调整合成速度

## Introduction

1. 现有的自回归的 TTS 存在的问题：
	1. mel 谱的推理速度慢
	2. 合成的语音不鲁棒，有words skipping and repeating 问题
	3. 合成的语音不可控，无法直接控制 speed 和 prosody
2. 提出 FastSpeech，输入 phoneme 序列，以非自回归的方式输出 mel 谱：
	1. 采用 Transformer 中的 基于自注意力机制的 feed-forward network 和 1D 卷积
	2. 采用 length regulator 来解决长度不匹配的问题，即根据  phoneme duration 对 phoneme 序列上采样

## 背景（略）

## FastSpeech

结构如图：
![](image/Pasted%20image%2020230917120730.png)

### Feed-Forward Transformer

FastSpeech 由基于自注意力机制的 feed-forward 结构 和 1D 卷积 组成。称为 Feed-Forward Transformer（FFT），图 a。

然后将多个 FFT 模块堆叠起来，在 phoneme 端有 $N$ 个，在 mel 谱 端有 $N$ 个。中间则是 length regulator。

每个 FFT 模块如图 b，包含自注意力和 1D 卷积。
> 之所以选 1D 卷积而不是 FFN，是因为在 phoneme 和 mel 谱 序列中，邻近的 hidden state 相似度更高。

其他都和 Transformer 一样。

### Length Regulator

解决长度不匹配的问题，且可以控制 speed 和 prosody。

定义  phoneme duration 为每个 phoneme 对应的 mel 谱帧 的长度（一个 phoneme 可能对应多个 mel 谱帧）。

假设 phoneme 序列为 $\mathcal{H}_{pho}=[h_1,h_2,...,h_n]$，$n$ 为其长度，phoneme duration 为 $\mathcal{D}=[d_{1},d_{2},\dots,d_{n}]$ ，所有的 $d_i$ 加起来长为 $m$，$m$ 为 mel 谱 帧的数量。定义 length regulator 为：
 $$\mathcal{H}_{mel}=\mathcal{LR}(\mathcal{H}_{pho},\mathcal{D},\alpha),$$
 其中 $\alpha$ 用于控制声音的速度。

### Duration Predictor

duration predictor 包含：
+ 两层 （1D 卷积 + ReLU + LayerNorm + Dropout ）
+ Linear Layer（输出 scale）

这个模块放在 phoneme side FFT 模块的顶端，采用 MSE loss 和 FastSpeech 联合训练。

duration predictor 只在推理阶段才用得上，因为在训练的时候 phoneme duration 是来自一个自回归的 teacher model：
+ 首先训练一个自回归的 encoder-attention-decoder based Transformer TTS 模型（Transformer-TTS）
+ 把 decoder-to-encoder attention 中得到的 alignment 提取出来，而且提出一个 focus rate $F=\frac{1}{S}\sum_{s=1}^{S}\operatorname*{max}_{1\leq t\leq T}a_{s,t}$，其中 $a_{s,t}$ 为 attention matrix 的第 $s$ 行第 $t$ 列，选择 $F$ 最大的那个 attention head 对应的 attention matrix 作为最终的 alignment（$S$ 为 mel 谱长度，$T$ 为 phoneme 序列长度）
+ duration sequence $d_i$ 则可以计算为：$\begin{aligned}d_i=\sum_{s=1}^S[\arg\max_ta_{s,t}=i]\end{aligned}$

## 实验

