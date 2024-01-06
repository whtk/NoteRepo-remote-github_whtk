> ICML 2023

1. 提出一种用于语音识别的自监督学习方法
2. 采用随机投影（RQ）量化器（Quantizer）生成离散的标签，quantizer 基于一个随机初始化的矩阵来对语音输入进行投影，在随机初始化的 codebook 中进行最近邻查找。
3. 自监督学习期间，矩阵和码本都**不被更新**。随机投影量化器没有经过训练，并且与语音识别模型分离
4. 在LibriSpeech上，非流式模型下，提出的自监督学习实现了与先前工作类似的 WER，流式模型下，提供了比wav2vec 2.0和w2v BERT更低的 WER和延迟。在多语言任务方面，该方法也比wav2vec 2.0和w2v BERT有了显著的改进

## Introduction

1. 语音识别的自监督学习的一个常见设计原则是表征学习，如很多基于 BERT 的方法，但是有一个问题，连续语音和离散的文本token之间有差距，解决方法是学习语音表征或者离散表征
2. 但是将表征学习和自监督学习结合有两个限制：
	1. 模型架构：表征学习可能需要上下文，但是有些下游任务不允许上下文
	2. 复杂性：两者的目标并不总一样，需要找到一种平衡
3. 于是提出 BERT-based Speech pre-Training with Random-projection Quantizer (BEST-RQ)，对语音信号进行 mask，然后输入到 encoder，基于未被 mask 的部分预测 mask 的部分，而 mask 输出的目标是由随机投影量化器提供的 label。RPQ 将语音信号投影到随机初始化的矩阵，在随机初始化的 codebook 中寻找最近的 vector，其索引就是 label
4. 学习的过程中，投影矩阵和 codebook 都不会被更新
5. 进一步研究了 表征学习 和 自监督学习 的关系，证明两个目标其实内在不一致，从而可以在没有表征学习的情况下设计自监督学习

## 基于 RPQ 的自监督学习

Quantizer 随机初始化投影矩阵和 codebook，使用矩阵来投影输入语音信号，在 codebook 中找到最近的 vector。

训练时不更新矩阵和 codebook。

输入数据归一化为 0，1 的标准正太分布以避免投影后的向量发生 collapse（也就是只对应到很小的一部分的 codebook）。

如图：![](image/Pasted%20image%2020230521104337.png)
训练完之后，使用 encoder 对 ASR 任务进行 fine tune。

mask 的时候，以固定概率对每一帧判断是否 mask（伯努利分布），mask 的值是 0，1 的高斯噪声。

### RPQ

给定输入向量 $x$ 是一个从语音信号计算得到的 $d$ 维向量，RPQ 将 $x$ 映射为离散的 label：$$y=\underset{i}{\operatorname{argmin}} \| \operatorname{norm}_{l 2}\left(c_i\right)-\text { norm }_{l 2}(A x) \|$$
其中，$A$ 为随机初始化的 $h\times d$ 投影矩阵，$C=\{c_1,\dots,c_n\}$ 为随机初始化的 $h$ 维向量，$\operatorname{norm}_{l2}$ 是 $l2$ 归一化，$A$ 采用 Xavier 初始化，codebook $C$ 采用标准正态分布初始化，训练的时候这两个参数固定。

### 预训练

预训练时，在 encoder 顶端添加 softmax 层，学习 label。提出的方法可以适用于任何架构（因为 RPQ 是独立的），实验时采用 Conformer block 。

对于非流式的模型，上下文可知，BERT 直接可以用起来。

对于流式模型，提出两种兼容的方法：
+ 仅基于过去的来预测 mask
+ 对未来的上下文进行 mask（注意这个 mask 和预训练的 mask 是不一样的）

### fine tune

基于下游数据集进行有监督 fine tune，在预训练过程中的softmax层丢弃。具体的 ASR 模型采用 RNN transducers 这类的端到端模型，encoder 部分就是预训练的 encoder + 一层额外的投影层来适应下游任务，decoder 是 LSTM。
> fine tune 时，encoder 是会被更新的。

### RPQ 为什么有效

两个问题：
+ 量化质量
+ 量化器对 SSL 的影响

将 Quantizer 和 VQ-VAE 进行比较（它的 Quantizer 是会被更新的），实验表明，RPQ 的质量其实并不好，但是对 SSL 仍然有效，而且随着数据集的增加差距减小。

因为 SSL-ASR 的目的是训练模型学习上下文信息，而 RPQ 可以保留语音数据的分布，从而使得模型可以学习处理原始信号并推断语音数据中的上下文信息。

## 实验

实验基于 Lingvo 库。

### LibriSpeech

输入语音是 80 维 log-mel filter bank，stride 为 10ms，fine tune 的时候 vocab size 是1024。

![](image/Pasted%20image%2020230521213249.png)
算法在流式和非流式预训练都优于wav2vec 2.0和w2v BERT。

### 多语言任务

![](image/Pasted%20image%2020230521213354.png)
使用所提出的BEST-RQ，平均 WE R进一步降低了3%。这证明了一个简单的 RPQ 对于多语言预训练也是有效的。且有了更多的微调数据，BEST-RQ 的表现甚至比w2v BERT更好。