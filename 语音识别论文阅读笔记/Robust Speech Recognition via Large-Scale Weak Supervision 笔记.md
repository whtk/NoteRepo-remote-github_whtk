> openai，whisper，ICML 2023

1. 用 680,000 小时的多语言和和多任务数据集训练模型，在 benchmarks 的任务上是 competitive 的，而 zero-shot 的效果很好
2. 模型可以得到人类的准确度和鲁棒性

## Introduction

1. 预训练的 audio decoder 很强，但是由于是纯无监督的，需要 fine tune 才能用于下游任务；而且 fine tune 会导致学习到一些没什么用的 dataset-specific 的特征
2. 语音识别系统应该可以在 out of the box 下也能工作，而不需要每次部署都进行一次有监督的 fine tune
3. 本文使用 680,000 小时的有标记数据集，提出 Whisper，同时还关注弱监督的预训练，在 680,000 小时中：
	1. 117,000 包含 96 种语言
	2. 125,000 包含任意语言到语音的语音翻译
4. 表明，弱监督预训练进行 scale 是有效的，可以不需要自监督训练

## 方法

### 数据处理

直接预测 raw text 而不做任何标准化，采用 sequence-to-sequence 模型的能力来学习音频和文本之间的映射。

文本多样性并不能使模型更鲁棒。且网络上数据的很多文本其实都不是人给的，而是用别的 ASR 得到，于是开发一些方法来去掉这些数据。

同时还使用 audio language detector，如果得到的语言不匹配的话也丢掉这部分数据。

分成 30s 的音频段，静音段也拿去训练了。

### 模型

采用已有的 encoder-decoder Transformer 架构：
![](image/Pasted%20image%2020231009102817.png)

### 多任务

语音识别系统其实应该还可以实现 voice activity detection, speaker diarization, 和 inverse text normalization 的功能。

为了在一个模型中实现 one-to-many mapping，需要一种格式来确定任务和输入。因为 decoder 是  audio-conditional language model，同时也把 history of text of the transcript 作为条件，其实就是以一定的概率把之前的文本也作为了条件输入（不仅是预测的音频）。

每次预测的开始是 <|startoftranscript|>（SOT） token，首先预测语种（一共99个），而如果音频段没有语音的话，预测的是 <|nospeech|> token，下一个 token 用于指定任务，<|transcribe|> or <|translate|>（语音识别 or 语音翻译），然后通过 <|notimestamps|> token 来制定是否预测 timestamps，此时定义完任务的格式，然后就可以开始输出了。

如果要预测 timestamp，会预测当前语音段中在哪些时间说话，然后依次预测 start time、caption（也就是文本）、end time。
> 其实就是多预测了每个文本的开头和结束时间，也就是在做 speaker diarization 的任务

最后添加 <|endoftranscript|> token。

### 训练细节

训练了四个版本的模型：
![](image/Pasted%20image%2020231009105042.png)
