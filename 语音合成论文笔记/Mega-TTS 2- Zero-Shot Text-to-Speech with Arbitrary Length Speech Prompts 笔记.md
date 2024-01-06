> ZJU，字节，renyi，2023 preprint

1. 之前的工作可以实现 10s 的 prompt 音频， 而 短的 prompt 则会限制性能
2. 本文提出 Mega-TTS 2，可以实现任意长度的 prompt 来合成未知说话人的语音：
	1. 设计了一个 multi-reference timbre encoder 来从多个参考语音中提取 timbre 信息
	2. 训练了一个可以实现任意长度 speech prompt 的 prosody language model
3. 同时还引入了一个 arbitrary-source prompts，采用 P-LLM 的输出的概览来产生 expressive 和 controlled prosody
4. 还提出了一个 phoneme-level 的自回归 duration 模型，来引入 in-context learning 能力

## Introduction

speaking style 包含多种元素：
![](image/Pasted%20image%2020231204204116.png)

每种元素都需要不同程度的 prompt，于是本文的目标是，实现可以支持任意长度的 prompt 的 zero-shot TTS。或者说，目标就是在给定长度的 prompt 中尽可能提取更多的信息。

本文贡献如下：
+ 为了拓展支持的 prompt 长度，训练 LM，从任意长度的 speech prompts 中自回归地生成 prosody codes
+ 为了得到 fine-grained timbre 信息，设计了一个 multi- reference timbre encoder，从而可以从多个参考语音中提取 timbre 信息
+ 提出 phoneme-level 的自回归 duration model，在这个模型中引入 in-context learning 能力，可以增强生成语音的自然度
+ 还提出了 arbitrary-source prompts 技术，可以从其他说话人中采用 prosody prompt 来生成 prosody，同时保留目标说话人的 timbre

在 LibriSpeech test-clean 数据集上实验，超过了现有的 SOTA zero-shot TTS。且发现，prompt 的长度越大，性能提升越大。

## 背景（略）

## 方法

整体架构如下：
![](image/Pasted%20image%2020231205155929.png)

采用 VQ-GAN based TTS 架构，包含：
+ content encoder
+ timbre encoder
+ VQ prosody encoder

然后引入 prosody information language model (PLM) 和  multi-reference timbre encoder (MRTE) 。

然后提出 phoneme-level auto-regressive duration model (ADM)，最后提出 prosody interpolation 技术。 

### 任意长度 prompt

Multi-reference timbre encoder：之前的模型的 timbre encoder 中都会有 temporal average pooling（也就是认为 timbre 是不随时间变化的）。本文引入 MRTE 可以从多个参考语音中捕获 fine-grained timbre 信息，如上图 b，包含 mel encoder、mel-to-phoneme attention 模块 和 length regulator。首先将 mel 谱编码到 acoustic hidden state $H_{mel}$，然后用 mel-to-phoneme attention 模块从 prompt 中提取语义相关的 timbre 信息，输入为 phoneme level content hidden state $H_{content}$ 作为 Query，$H_{mel}$ 作为 key 和 value。也用了一个 global timbre encoder（图中 GE） 来提取 time-invariant timbre 信息，然后拼接到 phoneme level hidden state。最后，用 length regulator 将 phoneme level hidden state 拓展以匹配 mel 谱 的长度，则输出 spectrogram-level hidden state $H_{CT}$ 包含 content 和 fine-grained timbre 信息。

Training PLM with arbitrary-length prompts：训练时，将同一个人的所有的句子在时间轴上拼接，每个 batch 最大的 mel 谱帧的长度为 32000，直接采用这个作为 speech prompt，然后用交叉熵训练 language model，模型把 speech prompt 的 prosody $\mathbf{u}$ 和 $H_{CT}$ 作为条件，自回归进行预测：
$$p\left(\mathbf{u}\mid H_{CT};\theta\right)=\prod_{t=0}^Tp\left(\mathbf{u}_t\mid\mathbf{u}_{<t},H_{CT};\theta\right)$$
推理的时候，通过增加 prompt 的长度就可以提高性能。

### 自回归 duration 模型

用 phoneme-level auto-regressive duration model (ADM) 来生成 duration，结构和 PLM 一样，但是用 MSE 损失。

### Prosody Interpolation

通过对多个不同说话人的 PLM 的输出的概率进行插值来进一步控制 prosody。

例如，如果一个人的 tone 过于平淡，如果我们想生成带有他的 timbre 的 prosody，方法为：
+ 从其他 expressive speaker 的语音中提取 $\mathbf{u}_{rhy}$，从 prompt 中提取 $\mathbf{u}_{flat}$
+ 分别用两个 LM 针对上面的 prosody 来解码
+ 在解码的每一步，两个 LM 的概率分布以权重 $\gamma$ 进行插值

用公式写为：
$$\tilde{p}\left(\mathbf{u}\right)=\prod_{t=0}^{T}\left(\gamma\cdot p\left(\mathbf{u}_{t}\mid\mathbf{u}_{<t},\mathbf{u}_{flat}\right)+\left(1-\gamma\right)\cdot p\left(\mathbf{u}_{t}\mid\mathbf{u}_{<t},\mathbf{u}_{rhy}\right)\right)$$
从而可以通过权重 $\gamma$ 来增强表达性或者选择保留原始韵律。

## 实验（略）