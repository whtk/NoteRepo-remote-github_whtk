> ICASSP 2023，中科大、微软

1. 本文探索采用文本描述来引导语音合成
2. 提出 PromptTTS，输入为风格和内容的 prompt，输出合成的语音
3. 包含 style encoder 和 content encoder，分别从对应的 prompt 中提取；然后一个 speech decoder 根据这两个表征来合成语音
4. 还构造了一个数据集

> 首次提出用文本来描述风格，然后就有一个基于文本的 style encoder、一个用于编码要合成的文本的 content encoder，最后一个 speech decoder 得到波形，

## Introduction

1. 风格控制可以以隐式（采用诸如 pitch 这类的风格）或者显式（从参考语音中学习 style token）
2. 本文探索采用文本描述来引导语音合成
![](image/Pasted%20image%2020231024203706.png)

3. 如上表，输入 prompt 包含风格描述和内容描述
4. 是第一个用 prompt 引导的 TTS 系统，为这个任务设计了一个数据集、系统和评估指标：
	1. 数据集包含 style 和 content 信息 和 对应的 语音，prompt 包含五个 style factor： gender，pitch，speaking speed，volume，emotion
	2. 提出 PromptTTS，包含 style encoder，content encoder，speech decoder
	3. 计算输出语音的真实 style factor 预测的之间的准确度

## 方法

![](image/Pasted%20image%2020231024204455.png)
style encoder 将 style prompt 映射到语义空间来提取 style representation，用于引导 content encoder 和 speech decoder。

content encoder 输入 content prompt，提取 content representation。

speech decoder 将  style representation 和 content representation 进行拼接作为输入来产生语音。

### Style Encoder

Style Encoder 采用 BERT 模型提取 style representation。输入风格序列 $\begin{aligned}T&=[T_1,T_2,\cdots,T_M]\end{aligned}$，然后在前面添加一个 $\left[CLS\right]$ token，最后转化为 word embedding 然后喂入到 BERT 模型中，其中 $M$ 为  style prompt 的长度。然后把 $\left[CLS\right]$ 对应的 representation 作为 style representation。

同时为了更好地识别语义信息，BERT 采用了一个额外的分类任务来预测  gender, pitch, speaking speed, volume, and emotion。

### content encoder

content encoder 基于 style representation 提取 content representation。采用 grapheme-to-phoneme 转换工具将输入转换 phoneme 序列 $\begin{aligned}P=[P_1,P_2,\cdots,P_N]\end{aligned}$，$N$ 为 phoneme 序列的长度。把 style representation 添加到  transformer block 中的每个输入中。然后和 FastSpeech 2 中一样，content encoder 最顶端的模型作为 variance adaptor 来预测 duration, pitch, and energy。

### Speech Decoder

speech decoder 采用 style 和 content representation 来生成 mel 谱，两个 representation 拼接起来作为 decoder 的输出，同时 style representation 也添加到每个 transformer block 的输入之前（和 content encoder）。

## 数据集

## 实验
