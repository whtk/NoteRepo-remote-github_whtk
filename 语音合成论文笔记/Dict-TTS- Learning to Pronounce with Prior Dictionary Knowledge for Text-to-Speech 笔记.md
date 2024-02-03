> NIPS 2022，ZJU，Yi Ren

1. 多音字消歧（Polyphone disambiguation） 旨在从文本序列中捕获准确的发音知识
2. 本文解决多音字消歧问题，提出 Dict-TTS，是一个 semantic-aware 的 TTS 模型，包含 online website dictionary（先验信息）：
    1. 设计 semantics-to-pronunciation attention (S2PA) 模块，以匹配输入文本序列和词典中的先验语义之间的 semantic patterns，并获得相应的发音
    2. S2PA 模块可以和端到端 TTS 模型一起训练，而无需任何标注的音素标签

> 很不错的想法，用 online 字典来解决多音字消歧的问题，核心在于用当前的 token 和字典中的 token 之间做 attention，然后分别得到语义特征（attention 加权）和正确的发音（概率最大的那个发音作为当前 token 的发音）。

## Introduction

1. 现有的 polyphone disambiguation 仍存在一些挑战：
    1. rule-based 方法其语言知识受限
    2. network-based 方法采用 G2P 得到 phoneme，没有显示的语义建模
2. 本文采用现有的 worldwide 先验知识，如图：
![](image/Pasted%20image%2020240202153853.png)
字典可以看成是一种先验知识库，当人类对某个 polyphone 未知时会去查字典，本文则在架构中模仿此场景
3. 提出 Dict-TTS，无监督的 polyphone disambiguation 框架，显示地采用 online dictionary 来得到正确的语义和发音，具体来说：

    1. 采用 semantic encoder 得到输入文本的 semantic contexts，采用 S2PA 模块在字典中搜索匹配的 semantic patterns，以找到正确的发音，得到的 semantic 信息也作为 TTS 模型的韵律建模

    2. 将 S2PA 模块融入端到端 TTS 系统的训练和推理而无需音素标签

3. 在三个数据集上实验：Mandarin、Japanese、Cantonese，Dict-TTS 在发音准确性和韵律建模上均优于其他 SOTA polyphone disambiguation 模型

## 方法

Dict-TTS 整体架构基于 PortaSpeech，主要包括以下步骤：
+ self-attention based semantic encoder 得到输入字符序列的语义表示
+ 利用预训练的跨语言模型提取字典条目中的 semantic context 信息
+ 计算与输入 graphemes 最相关 entry，得到相应的发音序列
+ 将提取的 semantic context 信息和发音输入 linguistic encoder 进行特征融合

### 概览
整体架构如图：
![](image/Pasted%20image%2020240202170205.png)

Dict-TTS 主要包括：
+ 基于 Transformer 的 linguistic encoder
+ 基于 VAE 的 variational generator，采用 flow-based prior 生成多样的 mel 谱
+ 基于随机窗口的 multi-length discriminator（替换了 PortaSpeech 中的 post-net），用于提高单词发音的自然度
+ 由于没有音素输入，将 linguistic encoder 替换为：
    1. semantic encoder 提取 grapheme 序列的 semantic representations
    2. semantics-to-pronunciation attention module 匹配字典 entry 和 grapheme 表征之间的 semantic patterns，得到对应的 semantic embedding 和 pronunciation embedding
    3. linguistic encoder 融合 semantic embedding 和 pronunciation embedding

### phoneme-based 和 character-based TTS 对比

下面的描述都基于 logographic writing system，即一个字符代表一个完整的 grammatical word 或 morpheme。

如图：
![](image/Pasted%20image%2020240202170927.png)

phoneme-based TTS 的 linguistic encoder 输入为 phoneme 序列 $p$，一般通过 G2P 模块得到。其需要从 $p$ 推断出 semantic 和 syntactic representation $s$，然后从 $s$ 和 $p$ 推断出 pitch trajectory、speaking duration 和其他 acoustic features 以生成 expressive 和 natural 的 pronunciation hidden state $g$。由于 phoneme 序列 $p$ 是语音中最小的声音单元，其可能在语义上有歧义。
	如，“AE1, T, F, ER1, S, T” 可以很容易地分类为 “At first”，但 “W, EH1, DH, ER0” 可以分类为 “Whether” 或 “Weather”。Homophones（同音异形异义词）如 “to”, “too” 和 “two” 可以转换为相同的 phoneme 序列 “T, UW1”，但它们的 local speaking duration 和 pitch 是不同的。且包含在 word-based 输入序列中的 tree-structured syntactic information 也会缺失。导致 $s$ 的歧义会影响 phoneme-based TTS 系统的韵律建模。

对于 character-based TTS ，需要先预测正确的 phoneme 序列 $p$。不同于 phoneme-based TTS 系统，character-based TTS 系统在字符到达时并不知道 phoneme 序列 $p$，也不知道如何准确地发音。TTS 训练中 mel 谱重构损失会将 character representation $c$ 拉到 acoustic space。
	如，中文字符 “火” 和 “伙” 发音相同（“H UO3”），但有不同的语义。然而，由于 mel 谱重构损失，它们的 representations 会根据 acoustic pronunciation 分布，这会阻碍 polyphone disambiguation 和 prosody modeling。

综上，character representations $c$ 应该位于 semantic space，这样才能基于 context 捕获 $s$，基于 dictionary 和 $s$ 推断 $p$，最终得到 natural pronunciation hidden state $g$。

### Semantics-to-Pronunciation Attention

如上图，S2PA 模块用于显式的 semantics comprehension 和 polyphone disambiguation。

设 dictionary $D$ 包含字符序列 $C = [c_1, c_2, ..., c_n]$，其中 $n$ 是字符集的大小（字典可以来自 online website）。每个字符 $c_i$ 有可能的发音序列 $p_i = [p_{i,1}, p_{i,2}, ..., p_{i,m}]$，每个发音 $p_{i,j}$ 有对应的 dictionary entry $e_{i,j} = [e_{i,j,1}, e_{i,j,2}, ..., e_{i,j,u}] \in E$（如 definitions, usages, translations，以字符格式呈现），其中 $m$ 是 $c_i$ 的可能发音数，$u$ 是对应 entry 中字符的数量。
> 对于 polyphones $m > 1$，对于只有一个发音的字符，$m = 1$。

S2PA 的目标是通过衡量输入字符序列 $t = [t_1, ..., t_l]$ 和字典中对应的 gloss items 之间的 semantic similarity 来得到发音序列 $p$，其中 $l$ 是序列长度。如图：
![](image/Pasted%20image%2020240202213356.png)
+ 先用预训练的跨语言模型提取每个 entry $e$ 的 semantic context 信息 $\mathbf{k}$ 并将其存储为先验字典知识
+ 然后用 semantic encoder 得到输入字符序列 $t$ 的 semantic contexts $z$。对于每个字符 token $t_i$，其 semantic feature vector $z_i$ 作为 semantics-based attention 模块的 query。attention 学习 semantic feature vector $z_i$ 和 $k_{i,j,k}$ 之间的相似度：$$[\mathbf{a}_{i,1,1},...,\mathbf{a}_{i,m,u}]=\frac{[\mathbf{k}_{i,1,1},...,\mathbf{k}_{i,m,u}]\cdot\mathbf{z}_i^\top}d\:,$$其中 $d$ 是缩放因子，$[\mathbf{a}_{i,1,1},...,\mathbf{a}_{i,m,u}]$ 表示 $t_i$ 和 $[e_{i,1,1},...,e_{i,m,u}]$ 中每个 item 之间的 semantic similarity。提取的 semantic embeddings $s'_i$ 可以通过 $s'_i = \text{softmax}([\mathbf{a}_{i,1,1},...,\mathbf{a}_{i,m,u}]) \cdot [\mathbf{k}_{i,1,1},...,\mathbf{k}_{i,m,u}]$ 得到。$s'_i$ 中的丰富语言信息可以作为辅助信息来提高生成的语音的自然性和表现力。

得到的 attention weight $w_{i,j} = \sum_{k=1}^u \mathbf{a}_{i,j,k}$ 可以看作是发音 $p_{i,j}$ 的概率。但是由于特定句子中 polyphone 只有一个正确的发音，于是使用 Gumbel-Softmax 函数来采样最可能的发音 $p'_i$：
$$\begin{aligned}w_{i,j}&=\frac{\exp\left(\left(\log\left(\mathbf{w}_{i,j}\right)+g_{i,j}\right)/\tau\right)}{\sum_{l=1}^{m}\exp\left(\left(\log\left(\mathbf{w}_{i,l}\right)+g_{i,l}\right)/\tau\right)}\:,\\\\p_i'&=\sum_{j=1}^{m}w_{i,j}\cdot p_{i,j}\:,\end{aligned}$$
其中 $g_{i,1,...,i,m}$ 是从 Gumbel(0,1) 分布中抽取的 i.i.d 样本，$\tau$ 是 softmax temperature。然后得到的 pronunciation embeddings $p'_i$ 和 semantic embeddings $s'_i$ 被输入到 linguistic encoder 的其余部分进行特征融合和 syntax prediction。

S2PA 模块可以看成一个用于对 character representation、pronunciation 和 semantics 解耦的端到端的方法。通过 S2PA 模块，character representation 成功地分布在 semantic space，使得模型可以根据词典知识推断正确的发音和语义。

### 训练和预训练

训练阶段，S2PA 模块的权重（包括 character token embeddings 和 pronunciation token embeddings）通过 TTS decoder 的重构损失联合训练。所以 Dict-TTS 不需要任何显式的 phoneme labels。推理时，将文本序列输入 S2PA 模块得到预测的发音。

尽管 S2PA 模块可以显式地学习 semantics-to-pronunciation mappings，但可能不够准确，因为 text training data 不够大。于是用大规模 ASR 数据集中的低质量 text-speech pairs 进行预训练。

## 实验（略）
