> ICASSP 2021，厦门大学

1. 提出一种新的轻量化的多说话人多语种语音合成，称为 LightTTS，可以快速合成 Chinese, English 或 code-switch 语音
2. 相比于 FastSpeech，mel 谱生成速度更快，参数更少

## Introduction

1. 提出 LightTTS，将轻量化卷积网络和 self-attention 结合起来减少参数
2. 把 x-vector 引入 LightTTS 来实现多说话人的 TTS
3. 引入 language ID 和 language embedding 来实现多语种

## 方法

设计了一个轻量化的 TTS，采用 sequence-level 的知识蒸馏来确保模型性能。然后引入 x-vector, language ID, and language embedding，可以实现多语言、多说话人的 TTS。

### Dynamic Convolution

DC 是一种轻量化的卷积网络，通过 depthwise convolution 和 group convolution 来减少参数，如图：
![](image/Pasted%20image%2020231228101700.png)

$K$ 为 kernel size，$d$ 为 embedding 的维度，$H$ 的 group 数量。DC 将输入的 embedding 在 channel 维度分为不同的 group，相同 group 的 参数 是通过 linear 层动态地从当前的 word embedding 中预测的，从而每个 group 可以并行计算。

### Relative positional encoding

原始的 transformer 采用绝对位置编码，只能编码有限长度的序列，而相对位置编码可以动态预测相对位置。

### LightTTS

采用 FastSpeech 的方法，如图：
![](image/Pasted%20image%2020231228102505.png)

encoder 和 decoder 采用 lightweight FF 模块。

对于 FF 模块，输入沿通道维度分成两个等长的部分，然后采用 DC 来提取 local context，用 self attention 来提取 global context，然后拼接起来以提高模型的表征能力。

引入 x-vector 从而合成不同说话人的语音，将 x-vector 归一化之后得到 z-vector，然后拼接到对应的 encoder 的输出中。

### Language embedding and language ID

还引入了 Language embedding 和 language ID，采用两个特殊的符号 < ZH > 和 < EN >，作为 language ID 来分别表示中文和英文。
![](image/Pasted%20image%2020231228103318.png)
然后上图的方式将 language IDs 插入到文本中，对于 code-switch 的情况，两个 language ID 分别插入到 switch point 中，然后采用 2 维的 one-hot embedding 作为 language embedding，和 text embedding 拼接。

### Low-rank approximation

multi-lingual TTS 的字典包含英文的 letter 和中文的拼音，导致 embedding 矩阵很大，于是采用 low-rank approximation 来减少参数，从 $M\times N$ 减到 $M\times k+k\times N(k\:<\:M,N)$。其中 $M$ 为字典大小，$N$ 为 embedding 的维度。

### Sequence-level knowledge distillation

非自回归模型比自回归的快，但是非自回归模型的并行输出会导致 decoder 的输出缺乏上下文依赖，于是采用 sequence-level knowledge distillation 来解决这个问题，和 FastSpeech 一样设计了一个 Transformer TTS 作为 teacher 模型，但是有一些不一样：
+ 引入 guided attention loss 来实现更好的 text-to-speech 对齐
+ 采用 multi-frame 输出策略，使 decoder 每次输出两个 mel 谱 帧，从而增强上下文的相关性
+ 引入 x-vector 来实现多说话人 TTS
+ 引入 pitch 和 energy 作为额外的韵律信息

## 实验（略）