> ACL 2023，ZJU、字节，renyi，huang rong jie

1. 提出 CLAPSpeech，一种跨模态的对比预训练框架，可以在不同上下文下学习相同文本的可变韵律特征：
	1. 在跨模态空间中，将 text context 和 对应的 prosody pattern 相联系
	2. 引入多尺度的预训练来捕获韵律
2. CLAPSpeech 可以提高现有 TTS 的韵律预测，而且还可以用于多语种和多说话人

## Introduction

1. 之前的 expressive TTS 采用外部的 variation predictor 或 variational generative model 来引入韵律；也有直接从文本表征中预测韵律的
2. 现有的方法主要基于 MLM 任务或者 MAM 任务，会导致两个缺点：
	1. 采用重构损失隐式学习韵律，很难提高韵律建模
	2. 没有解耦发音空间和韵律空间，训练效率低
3. 韵律可以看成是，在不同条件下相同 token 的 pitch 和 duration 的变化量
4. 本文从 CLAP 中启发，提出对比学习方法，将文本和韵律联系起来，称为 CLAPSpeech：
	1. 学习 text encoder 从文本中预测 韵律
	2. prosody encoder 从语音中提取 GT prosody
	3. 训练时，选择包含相同的发音 token 的 text-speech 对（不是整句，而是 token），对齐后 text encoder 用于从 text context 中提取 prosody

![](image/Pasted%20image%2020231219160038.png)

还提出 一种多尺度的预训练框架，学习两个 CLAPSpeech 模型，分别从 phoneme 和 word level 提取 prosody。

预训练完成后，可以看成是一种即插即用的 text encoder，用于从文本中提取 fine-grained prosody 表征。

## 相关工作（略）

## CLAPSpeech

包含 text encoder 和 prosody encoder。

### text encoder 和 prosody encoder

相同发音的 token 在不同的 text context 下不一样。目的就是要建模 text context 和 high-level prosody 之间的相关性。这里设计了一个 text encoder 和 prosody encoder 来构建这一多模态的韵律 embedding 空间。

![](image/Pasted%20image%2020231220102135.png)

如上图 a，text encoder 采用 phoneme 和 BPE 作为输入，网络结构包含几层的 FFT 层，学了两个独立的 FFT 来分别处理  phoneme 和 BPE 序列。但是由于两者的长度不一样，采用 word-level pooling 来处理 BPE 到 word level，然后拓展到 phoneme level。具体来说，如下图：
![](image/Pasted%20image%2020231220102546.png)
word-level pooling 根据单词边界在每个 word 里面对 phoneme hidden states 取平均。而 word2ph 操作根据单词边界重复 word hidden state。

融合两个序列之后，采用一个额外的 FFT 模块来得到最终的 phoneme-level text  encoding。由于预训练的时候，只有一个 token 被分析，所以 index 到这个 token 的位置得到最终的 token encoding。

prosody encoder 用于从语音段中提取被选中的 token 的韵律。然后根据单词边界将 mel 谱 分段，然后处理对应的 那个 token 得到一个 global encoding，这里 clip 之后得到的那段语音就只包含对应的 token 的局部的 prosody 信息但是又不会丢失任何的上下文信息。

得益于对比学习，提取的 global prosody encoding 可以从 phonetic 和 speaker 空间中解耦：
+ 因为正负样本都属于相同的发音的 token，可以消除 phonetic 信息
+ text encoder 并不包含说话人信息，导致 prosody encoder 会过滤掉说话人信息

采用 ResNet-50 作为 prosody encoder 的 backbone。

### 多尺度对比预训练

为了构造用于对比预训练的 mini-batch，随机选择一个 text token ，然后采样 batch 为 $N$ 的 text-speech 对（包含这个 token）。然后学习两个 CLAPSpeech 模型，分别用于 phoneme-level 和word level 的 text token。

具体而言，对于 phoneme-level CLAPSpeech 来说，假设选择的 text token 为 $X_{text}$，然后 phoneme token 对应的语音段为 $X_{speech}\in\mathbb{R}^{F\times T}$，$F$ 是 mel bin 的数量，$T$ 是帧数，下面用这两个分别表示 batch 为 $N$ 的 text-speech 对。

那么 text encoder 的输出 $f_{text}(X_{text})$ 为 phoneme-level 的 encoding，然后选择 phoneme token 对应的 token encoding $f_{text}(X_{text})_{i_{ph}}$，同理有 speech encoding $f_{speech}(X_{speech})$，然后进行归一化再线性投影到多模态的 embedding 空间：
$$\begin{gathered}
\begin{aligned}T_{ph}=L_{text}(LN(f_{text}(X_{text})_{i_{ph}}))\end{aligned} \\
S=L_{speech}(LN(f_{speech}(X_{speech}))), 
\end{gathered}$$
其中 $T_{ph}\in\mathbb{R}^{N\times C},S\in\mathbb{R}^{N\times C}$，$C$ 为 channel size。

CLAPSpeech 用于训练从 $N\times N$ 个 text-speech 对中选择正确的那些配对。也就是最大化 $N$ 个 real pair 之间的余弦相似度，最小化 $N^2-N$ 个负对之间的相似度。损失为：
$$\mathcal{L}_{ph}=0.5\times(l_{text}(\tau\cdot C_{ph})+l_{speech}(\tau\cdot C_{ph}))$$
其中 $C_{ph}\in\mathbb{R}^{N\times N}$ 为相似度矩阵，即 $C_{ph}=T_{ph}\cdot S^T$，$\tau$ 为 temperature parameter。

而 word-level 的 CLAPSpeech 类似，采用 word pooling 来处理 phoneme-level text encoding 到 word level，然后索引选中 $T_{word}$，其他的就和之前的一样，此时的 损失 为：
$$\mathcal{L}_{word}=0.5{\times}(l_{text}(\tau{\cdot}C_{word}){+}l_{speech}(\tau{\cdot}C_{word}))$$


### 将 CLAPSpeech 引入 TTS 中

以 [PortaSpeech- Portable and High-Quality Generative Text-to-Speech 笔记](../PortaSpeech-%20Portable%20and%20High-Quality%20Generative%20Text-to-Speech%20笔记.md) 为例，如图：
![](image/Pasted%20image%2020231221105237.png)

将 CLAPSpeech 预训练的 text encoder（红框里面的）作为一个额外的 encoder，然后把这部分得到的结果和原来的phonetic encoder 的输出相加。训练时，固定这 CLAPSpeech 的参数来避免过拟合。

## 实验（略）