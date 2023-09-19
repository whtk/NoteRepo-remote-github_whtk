> Google，2018，SLT

1. 在 GST 之后，提出 Text-Predicted Global Style Token（TP-GST），把 style embeddings 作为 Tacotron 中的 “virtual” speaking style labels
2. TP-GST 学习从文本中预测 stylistic renderings，在训练时不需要显示的 label，推理的时候也不要额外的输入
3. 实验表明：
	1. 生成的音频相比于 SOTA 有更多的 pitch and energy variation 
	2. TP-GST 可以合成移除背景噪声的音频
	3. 多说话人的 TP-GST 可以分解说话人信息和说话风格

## Introduction

1. Prosody 包含 low-level characteristics such as pitch, stress, breaks, and rhythm, and impacts speaking style
2. Tacotron 有一点韵律建模能力，如一段话以问号结尾的话会有 pitch 的上升，但是对于长语音很有挑战，因为语音越长，其韵律风格就越平均
3. GST [Style Tokens- Unsupervised Style Modeling, Control and Transfer inEnd-to-End Speech Synthesis 笔记](Style%20Tokens-%20Unsupervised%20Style%20Modeling,%20Control%20and%20Transfer%20inEnd-to-End%20Speech%20Synthesis%20笔记.md) 虽然可以区分  speaking style，但是在推理时需要额外的音频或者手动选取的权重
4. 本文改进 GST，从文本中预测 speaking style，给出了两种易于实现的方法且不需要额外的文本，最终可以捕获 speaker-independent factors of variation，包含 speaking style 和 background noise

## 相关工作（略）


## 模型

基于 Tacotron，用的是 GST 的模型结构，也就是加了 reference encoder、style attention module 和 style embedding 。

提出的 Text-Predicted Global Style Tokens 添加了两个额外的 text-
prediction 路径，通过以下两种方法使得模型可以在推理的时候预测 style embedding：
+ 通过从文本中预测的 combination weights 来插入 GST
+ 从文本特征中直接预测 style embedding，忽略 style tokens 和 combination weights

通过使用 stop gradient 算子，两条预测路径可以联合训练（个人感觉有点类似于多任务学习），推理时，可以用上面两种方法，也可以用 GST 原始的方法。

### 文本特征

两种路径都使用 Tacotron 的 text encoder 的输出作为特征。而由于这个输出是可变长度的，TP-GST 会将输出序列通过 64-unit time-aggregating GRU-RNN，输出一个固定长度的 text feature vector，然后作为两个路径的输入。

### 预测 Combination Weights
![](image/Pasted%20image%2020230905100035.png)

Tacotron 中，prosody embedding 是 query，然后对 token 进行 attention 得到概率，可以看成是 combination weights。

而这里就直接把 combination weights 看成是网络的预测输出，称为 “text-predicted combination weights”（TPCW-GST），因此为了预测这些权重，把前面固定长度的输出通过 FC 层得到 logits，然后计算这些值和 target（attention 的输出概率）的交叉熵。训练的时候使用 stop gradient 来确保只训练 FC 层和 64-unit time-aggregating GRU-RNN 层。

推理的时候，GST 向量是固定的，这条路可以直接通过文本得到 combination weights。

### 预测 Style Embedding
![](image/Pasted%20image%2020230905100831.png)

这条路称为 “text-predicted style embeddings”（TPSE-GST），其实就是把前面的向量通过一个或者多个 FC 层，然后输出的是 style embedding，训练的损失就是和 attention 得到的 style embedding 之间的 L1 loss，同样也需要 stop gradient。

推理的时候，可以直接从文本预测 style embedding，这条路完全和 style token 无关了（但是在训练的时候还是需要的，因为要作为 target，只是测试的时候不需要了）。

## 实验（略）