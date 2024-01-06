> Google，2020

1. 提出两种参数简化技术，提高 BERT 训练速度
2. 还使用了一种自监督损失，专注于对句子间连贯性的建模，并表明它有助于多句子输入的下游任务
3. 在GLUE、RACE和SQuAD benchmark 上实现了 SOTA

## Introduction

1. 设计了一种 Lite BERT （ALBERT）架构来解决 BERT 参数多的问题
2. 采用两种参数简化技术：
	1.  factorized embedding parameterization，将大的 vocabulary embedding 矩阵分解为两个小的矩阵
	2. 跨层的参数共享
3. 两种方法都可以极大减少参数而不严重影响性能，在 BERT-large 中，参数减少了 18 倍
4. 引入一种用于句子顺序预测（SOP）的自监督损失，SOP 主要关注句子之间的连贯性
5. 在 GLUE, SQuAD, 和 RACE benchmarks 的 NLU 任务中实现 SOTA 性能

## 相关工作

backbone 和 BERT 相似，GELU 的非线性激活+Transformer 结构。

设word embedding 维度为 $E$，encoder 层数为 $L$，hidden size 为 $H$。

ALBERT 的三个主要共享：

### Factorized embedding parameterization

BERT 中，有 $E\equiv H$，但是从建模的角度看，WordPiece 旨在学习上下文无关的表示，而隐藏层嵌入旨在学习与上下文相关的表示，两者大小不一样且 $H\gg E$ 能够更有效地使用建模需求所告知的总模型参数；从实践角度看，词表数量 $V$ 通常是很大的，$H$ 的增加会导致整个embedding matrices $V\times E$ 很大。

因此，在 ALBERT 中，将整个 embedding matrices 分解为两个矩阵，也就是首先将 one hot vector（$V$ 维）投影到 $E$ 维，然后在投影到 hidden space $H$ 维中，此时整个句子的参数维度从 $V\times H$ 减少到 $V\times E+E\times H$，当 $H\gg E$ 时减少量很显著。

### 跨层参数共享

跨层共享所有的参数
![](image/Pasted%20image%2020230508155214.png)
上图给出了 BERT-large 和 ALBERT-large 每一层的输入和输出 embedding 的 LA 距离和余弦相似度，发现 ALBERT从一层到另一层的过渡要比BERT平稳得多，也就说明权重共享对稳定网络参数有一定作用。

### 句间连贯性损失

BERT 里面除了用 MLM 损失，还有一种下一句预测（NSP）损失。NSP是一种二分类损失，用于预测原始文本中是否连续出现两个片段。但是研究发现 NSP 好像效果不太好，后来的研究都丢弃了它反而获得了性能的提升。

原因很可能是，相比于 MLM 任务，NSP 任务过于简单（以为负样本来自于不同材料的文本，所以从主题来看他们就不一致，更不用说上下文的连贯性），模型相当于什么都没学到。

所以句间的建模应该主要基于连贯性损失，于是在 ALBERT 中用 SOP 损失，其中，正样本的选择和 BERT 一样（同一个文本的两个连续句子），而负样本是两个连续句子交换顺序（而不是之前的从两个文本中任意选取），从而迫使模型学习语句层面的连贯性。

这一损失也使得 ALBERT 可以提高多语句编码的下游任务性能。

### 模型

ALBERT 和 BERT 的比较如图：![](image/Pasted%20image%2020230508160314.png)

## 实验和结果

和BERT 一样，使用 BOOKCOPRUS 和 English Wikipedia数据集，包含 16G 无压缩文本，输入格式为 $[CLS]\quad x_{1}\quad [SEP]\quad x_2\quad[SEP]$ ，其中 $x_1,x_2$ 为两个句子，采用 30000 大小的 SentencePiece 作为词表。

batch size 为 4096，LAMB 优化器，训练 125,000 steps，64 到 512 Cloud TPU V3 上进行训练。

和 BERT 对比的结果：![](image/Pasted%20image%2020230508162917.png)

消融实验：
embedding 大小的影响：![](image/Pasted%20image%2020230508163048.png)
$E=128$ 的结果最好。

参数共享的影响：![](image/Pasted%20image%2020230508163203.png)
all shared 都会损害性能，且性能的下降大部分来源于 FFN 层的共享。

SOP 的有效性：![](image/Pasted%20image%2020230508163320.png)
