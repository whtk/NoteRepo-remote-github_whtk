> 2019 年

1. 提出 BERT 模型，即 Bidirectional Encoder Representations from Transformers，通过对模型层中的所有的上下文进行建模，来从未标记的数据中预训练深度表征
2. 预训练的 BERT 可以通过额外的 输出层 进行 fine tune，从而创建各种任务下的 SOTA 模型，而无需针对每个任务设计特定的架构

## Introduction
1. 有两种方法用于将预训练的 LM 用于下游任务：
	1. 基于特征的，如 ELMO 使用包含预训练表征的特定架构作为额外的特征
	2. fine tune：如 GPT，通过 fine tune 所有的预训练参数来针对下游任务进行训练
2. 作者任认为，这些技术限制了预训练表征的能力，因为他们的 LM 是单向的，每个 token 只能看到前面的 token。对于句子级别的任务来说，这种限制导致最后的结果并不是最优的，且基于 fine tune 的方法如果用于 QA 这类任务时反而有害，因为必须结合上下文。
3. 于是提出 BERT，使用双向的 Transformer encoder 来改善基于 fine tune 的 方法，使用 MLM 作为预训练目标，从输入中随机 mask 一些 token，目标是通过上下文预测被 mask 的 token，MLM 可以融合上下文，从而可以使用双向 Transformer
4. 此外还有一个 ”下一句预测“ 任务，联合预训练的文本对表征
5. 本文的贡献有：
	1. 证明了语言表征提取中双向预训练的重要性
	2. 预训练表征减少了大量工程化的特定任务架构的需求
	3. 在 11 个 NLP 任务中实现了 SOTA

## 相关工作

## BERT
分为两个步骤，预训练和 fine tune，预训练期间，模型在不同的预训练任务中基于未标记的数据进行训练，fine tune 时，使用来自下游任务的标记数据对所有参数进行调整，且每个下游任务都有单独的 fine tune 模型。

整个过程为：![[Pasted image 20230302222844.png]]

在不同的任务上架构都是统一的，在预训练和 fine tune 的架构上只有很小的区别。

### 架构

多层双向 Transformer encoder，层数为 $L$，hidden size 为 $H$，self-attention head 数为 $A$，实现了两个模型：
+ base：L=12, H=768, A=12, 参数量 110M，和GPT 的参数差不多
+ large：L=24, H=1024,A=16, 参数量 340M

### 输入输出表征

输入要可以表示一个句子或者一对句子，其中 ”句子“ 是指任意长度的连续文本（不是实际的语言句子），序列是 BERT 的输入，可能是一个句子或者两个句子的组合。

使用 vocabulary 大小为 30000 的 WordPiece embeddings，每个序列的第一个 token 是 CLS，并且这个 token 对应的输出用作分类。对于句子对，使用 SEP 来分隔，同时在每个 token 中添加一个可以学习的 embedding，表明其属于句子 A 还是 B（如上图中的左边）。

将输入 embedding 记为 $E$，CLS 对于的输出的那个 hidden vector 记为 $C\in \mathbb{R}^H$，第 $i$ 个输入 token 对于的输出 hidden vector 为 $T_{i}\in \mathbb{R}^H$。

对于给定的 token，其对应的输入表征为 token embedding+segment embedding+position embedding：![[Pasted image 20230302224734.png]]

### 预训练 BERT
使用两个无监督任务训练 BERT。

#### MLM（Mask LM）
传统的 LM 只能以回归的方式（要么是根据前面预测下一个，要么是根据后面预测上一个），因为如果使用双向条件会使得每个单词间接地看到自己，在多层的上下文中，模型其实很容易预测目标。

为了成功训练双向的模型，作者选择随机 mask 一定百分比的输入 token，称这种方法为 MLM，然后 masked token 对应的最终的 hidden vector 经过 softmax 得到预测 token。

和 DAE 不同，这里只会预测 masked token，而不会预测全部的输入。

但是这会造成预训练过程和 fine tune 过程不匹配，因为 MASK 这个 token 在 fine tune 期间不匹配，为了缓解这种情况，实际不总是选择 MASK 这个 token，而是首先确定要被 mask 的 15% 的 token 的位置，然后按照比例进行，如果第 $i$  个 token 被选中，则：
+ 以 80% 的概率把它替换成 MASK
+ 以 10% 的概率替换称一个随机的 token（不是 MASK ）
+ 以 10% 的概率保持不变

#### NSP（Next Sentence Prediction）
类似于 问答 或者 自然语言推理 这类的任务都需要理解两个句子之间的关系，但是一般的 LM 模型是无法建模这些的。

于是预训练一个二分的 NSP 任务，数据可以从任何单语种数据库中生成。具体来说，在选择句子的时候，一半的概率选择两个连续的句子 A-B，一半的概率选择两个随机的句子 A、B。

图中的 $C$ （ CLS 对应的输出的 token）就是用于 NSP 任务的，虽然看起来很简单，但是这样做确实可以提高在 QA 和 NLI 任务中的性能。

#### 数据
训练数据：BooksCorpus，800M 单词 ；English Wikipedia，2,500M 单词

### fine tune
对于每个特定的任务，将特定的输入输出放进 BERT 中然后端到端的 fine tune 所有的参数就行。

在输出短， token 表征送到输出层用于 token level 的任务，CLS 对应的输出 $C$ 送到另一个输出层用于 分类相关的任务。

## 实验（略）