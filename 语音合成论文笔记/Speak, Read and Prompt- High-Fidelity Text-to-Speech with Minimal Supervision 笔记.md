> Google，preprint

1. 提出 SPEAR-TTS，是一个多说话人 TTS 系统，其将两种离散语音表征结合，把 TTS 分解成两个 seq2seq 的任务：
	1. reading：从文本生成 high-level 的 semantic token
	2. speaking：从 semantic token 生成 low-level 的 acoustic token
2. 为了控制 speaker identity，采用 example prompting，从而允许 SPEAR-TTS 泛化到未知说话人（只需 3s 的音频样本），而不需要任何显式的说话人表征或者说话人 id

## Introduction

1. 把 TTS 看成是从文本到声学 token 的一种“翻译”，而 semantic token 则充当一种  pivot “language”，从而将 TTS 看成两个 seq2seq 任务：
	1. reading：将 text 翻译到 semantic token
	2. speaking：将 semantic token 翻译到 acoustic token
2. 好处就是，可以将两个 stage 解耦，reading stage 需要 text-audio 数据，而用于训练  speaking 模块的 audio token 快可以来自自监督语音模型，从而可以从无标签的数据中提取
3. 将 BART/T5-style pretraining 和 backtranslation 组合，减少训练 SPEAR-TTS 中所需的监督数据
4. 为了控制语音，采用 prompt 机制，将一段表示目标语音的 audio clip 作为 speaking 模块的条件，从而可以实现可控的多说话人 TTS

## 离散语音表征

1. semantic token 的作用是，为后面产生 acoustic token 提供一个 coarse、high-level 的条件，用于提供语言内容(从 phonetic 到 semantic)突出的语音表征，而副语言信息(如说话者身份和声学细节)则被删除
2. 于是基于 w2v-BERT 训练模型，将 MLM 和 对比学习组合来得到语音表征，完成训练后采用 k-mean 聚类，将聚类中心作为离散的 token
3. acoustic token 可以提供高保真的声学细节重构
4. 训练一个 SoundStream 来重构语音的同时将其压缩为离散的 unit

## SPEAR-TTS

拓展 [AudioLM- a Language Modeling Approach to Audio Generation 笔记](AudioLM-%20a%20Language%20Modeling%20Approach%20to%20Audio%20Generation%20笔记.md)，从而可以将文本作为一种条件：
![](image/Pasted%20image%2020231113203804.png)

stage 1：文本输入翻译为离散的 semantic 序列
stage 2：将 semantic token 映射到 acoustic token，然后通过 SoundStream 的 decoder 得到语音

好处在于：
+ semantic token 包含大量的 phonetic content，而 prosody 和 speaker 信息则受限，从而比直接从文本中学习 acoustic token 更容易
+ stage 2 可以采用 audio-only 的数据训练，而这种数据通常很多

两阶段的分离后，第一阶段也可以以一种 denoising pretext 的任务进行训练。

### stage 1

采用 parallel text-semantic tokens 数据来训练。

可以采用 encoder-decoder 或 decoder-only Transformer 结构来实现。

下面给出两种方法来减轻对于大量 parallel data 的限制。

#### 预训练

模型的输入为 corrupted version of an original semantic token 序列，目标是预测未被 corrupted token。

corruption 方法包括：
+ 随机替换
+ 删除
+ mask
发现删除的效果最好。

预训练完成后就进行 fine tune，冻住 encoder 的 upper layers 和 decoder 的所有参数，但是除了 decoder-encoder cross-attention layers 中的参数。

#### Back translation

文本到语音是一个 one-to-many 的映射，从而使得文本到语音的映射是非对称的。从而可以使用 back translation，即使用 parallel data 训练 speech-to-text model，然后用这些文本来生成合成语音。

用这个的好处在于：
+ 复杂度的降低，即不需要处理 raw audio
+ 可以实现 same semantic level 的 预训练（backward 阶段）

也是从预训练的模型开始，冻住 encoder 只 fine tune decoder，然后用生成的数据来训练 stage 1 的模型，整个过程如图：
![](image/Pasted%20image%2020231114102928.png)

### stage 2：控制生成过程

在 audio-only 的数据中，提取了一系列 semantic 和 acoustic token 序列，然后训练一个 transformer 模型来做 seq2seq translation，此过程可以生成 带有随机变化的 voice, tempo, and recording conditions 的语音。

为了控制说话人声音的特征，根据 AudioLM 中的发现：
+ 如果 speech prefix 仅基于 semantic token，可以生成连续的样本但是随机的声音
+ 如果还同时包含 acoustic token，则可以保留声音的特征

如图：
![](image/Pasted%20image%2020231114114218.png)
训练时，随机选两个不重叠的语音窗口，然后计算 semantic 和 acoustic token，把其中一个看成 prompt，另一个看成 target 输出，然后将序列以下述方式进行拼接：
+ prompt 中的 semantic token
+ target 中的 semantic token
+ prompt 中的 acoustic token
+ target 中的 acoustic token

训练时，前三个作为 prefix，模型用于预测第四个。

生成的样本可能存在一些噪声，提出两种方法来控制噪声等级：
+ 选择 cleaner speech 作为 prompt
+ 采用 stochastic sampling（如 temperature sampling）来生成多个样本，然后噪声最小的那个（用了 DNSMOS 来估计 MOS）

## 实验（略）
