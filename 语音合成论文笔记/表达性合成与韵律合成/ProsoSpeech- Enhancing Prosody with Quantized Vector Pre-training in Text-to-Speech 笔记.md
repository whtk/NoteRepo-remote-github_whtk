> 字节，浙大，ICASSP 2022，renyi

1. 韵律建模有以下几个挑战：
	1. 之前工作中提取的 pitch 有错，从而损害了韵律建模
	2. 韵律的不同属性（pitch、duration 和 energy）相互独立
	3. 由于韵律的高可变性和受限的高质量的数据集，韵律分布很难建模
2. 提出 ProsoSpeech，采用预训练的量化的 latent vector 来增强韵律
3. 引入 word level 的 prosody encoder，将语音的低频带部分进行压缩，然后将韵律压缩为 latent prosody vector（LPV）
4. 然后引入 LPV predictor，给定 word sequence 来预测 LPV

## Introduction

1. 之前的韵律建模存在一些问题：
	1. 一些工作采用外部的工具来提取 pitch contour，但是会导致一些不可逆的 error
	2. 一些工作从语音中提取韵律属性，但是是分开建模的
		1. 实际上，这些属性是相互依赖的，分开建模会导致一些 unnatural
	3. 韵律高可变性（person-to-person，word-to-word）
2. 提出 ProsoSpeech，基于 FastSpeech：
	1. 引入 word level prosody encoder 来从语音中解耦 prosody，根据 word boundary 将语音低频自带量化为 word level 量化的 LPV
	2. 提出自回归的 LPV predictor，基于 word level 的 text sequence 来预测 prosody
	3. 在大规模的文本和低质量的语音数据中预训练 LPV predictor，然后在高质量的 TTS 数据集上做 fine tune

## 方法

![](image/Pasted%20image%2020231213224240.png)

模型基于 FastSpeech，训练时，输入文本转为 phoneme 序列和 word 序列，通过 encoder 编码为 linguistic 特征，然后把 GT mel 谱 的 低频部分基于 这个 linguistic 特征，采用 prosody encoder 编码为量化后的 LPV。最后，把 linguistic 特征和 LPV 一起送到 decoder 来预测 mel 谱，采用 MSE 和 SSIM loss 来优化。

这个过程得到了 LPV，那为了预测 LPV，基于 word 序列 训练了一个自回归的 predictor。

推理时，采用 LPV predictor 得到 LPV，然后通过 decoder 从文本合成音频。

### Prosody Encoder

prosody encoder 采用 word-level VQ 层来从语音中解耦韵律。如图 b，包含两个 level，第一个 level 将 mel 谱压缩 为 word-level hidden state，第二个 level 后处理这些 state，最后通过 VQ 层得到 word-level LPV 序列。

但是，训练时发现要很多步才能很好的提取 prosody，而且可能出现 index collapse 的问题，模型只使用了很少部分的 code。

于是提出 warmup 策略 和 基于 k-mean 聚类的初始化策略：
+ 在前 20k step 移除了初始化层
+ 20k 之后，采用 k-mean 聚类来初始化 codebook
+ 此时再添加 VQ 层进行后续训练

### LPV predictor

如图 c，输入为 文本，预测 LPV 序列。采用 teacher forcing 的方式进行训练。

### 预训练和 fine tune

LPV predictor 并不能很准确地预测韵律，原因在于：
+ 训练的文本数据不够大
+ 训练的语音/韵律数据不够大

于是提出采用额外的纯文本和低质量的语音数据进行预训练，如图 d。

对于文本，采用 BERT-like mask prediction 来训练，对于低质量的音频数据，采用从噪声音频中编码的 LPV 序列进行预训练。

最后在高质量的数据集上进行 fine tune。

## 实验（略）

内部数据集下训练的。