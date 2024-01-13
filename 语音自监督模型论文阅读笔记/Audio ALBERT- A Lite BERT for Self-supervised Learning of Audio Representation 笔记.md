> 李宏毅团队，Amazon AI，2021

1. 提出 Audio ALBERT，在两个下游任务（说话人分类和音素分类）实现了与大规模预训练网络相当的性能，同时参数减少了91%
2. 相比于一些常用的表征，Audio ALBERT 得到的表征包含更多的音素和说话人信息

> 本质就是，共享 Transformer encoder 层的参数。

## Introduction

1. BERT-like 的自监督模型已经被用于语音，如语音识别和说话人识别
2. 但是这类模型通常很大，ALBERT 是 文本版的 BERT 的精简版减少了大量参数
3. 本文首先检查了 Mockingjay的每一层中的知识编码，发现所学习的参数在各层之间是多余的，因此将 ALBERT 共享参数的想法引入语音，提出了一种新的自监督模型，即音频ALBERT（AALBERT）
4. 表明，AALBERT在下游任务中产生的性能与其他预训练模型相当，但网络要小得多；同时还分析了从AALBERT的不同层提取的表征，发现中间层的表征比最后一层包含更多的语音和说话人信息

## 相关工作（略）

## 方法

### Mockingjay

具体原理见 [Mockingjay- Unsupervised Speech Representation Learning with Deep Bidirectional Transformer Encoders 笔记](Mockingjay-%20Unsupervised%20Speech%20Representation%20Learning%20with%20Deep%20Bidirectional%20Transformer%20Encoders%20笔记.md)，模型有三种，分别采用 3层、6层和12层 Transformer encoders（分别表示为Mockingjay-3L、Mockingjay-6L和Mockingjay-12L）。

通过研究 Mockingjay 中，使用 JS 散度来评估每个Transformer encoder 层的注意力分布之间的差异如图：![](image/Pasted%20image%2020230509110935.png)
发现各层之间的JS差异是显著的；而如果每层随机选取一个 attention head 进行同样的实验，发现 尽管有些 attention head 与其他层（深蓝色）的非常不同，但大多数层都是相似的（浅蓝色）。这说明，**对于 Mockingjay-6L 中特定的 attention head，通常在不同的层上有一些相似的注意力分布，也就是说，参数存在冗余**。

### AALBERT

输入还是为 声学特征，mask 的规则也和 Mockingjay 一样。

引入 weight tying 来减少参数：![](image/Pasted%20image%2020230509111416.png)
AALBERT 中每个 component 的参数都是共享的，如图：![](image/Pasted%20image%2020230509111609.png)
## 实验和结果

实验设置见原论文。

音素分类结果：![](image/Pasted%20image%2020230509113420.png)

说话人分类结果：![](image/Pasted%20image%2020230509113453.png)


