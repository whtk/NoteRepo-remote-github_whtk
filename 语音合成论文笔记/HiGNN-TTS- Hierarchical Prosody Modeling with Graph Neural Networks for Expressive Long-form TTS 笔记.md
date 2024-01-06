> ASRU 2023，西工大、港中文

1. 基于 GNN 的 TTS 已经可以提高短合成语音的表达性，但是长语音仍具有一定的挑战
2. 提出采用 hierarchical prosody modeling，称为 HiGNN-TTS
	1. 在图中添加了一个 virtual global node 来强化 word node 之间的连接
	2. 引入 contextual attention 机制来加强韵律建模
3. 同时 采用 hierarchical supervision from acoustic prosody on each node of the graph to capture the prosodic variations with a high dynamic range

## Introduction

1. 传统的 TTS 主要是 单个句子 上的韵律建模，但是长语音通常包含多个语义上相关的句子，且每个句子的韵律会被其上下文影响，对于长语句的合成，既要确保整体韵律的一致性，又要建模局部的精细化的韵律和全局的韵律
2. 提出 HiGNN-TTS，通过 word、sentence 和 cross-sentence 不同 level 来实现层级韵律建模：
	1. 添加一个 virtual global node 来提高 word node 之间的联系
	2. 设计一个 hierarchical graph encoder 来提取三个 level 的结构
	3. 通过 GNN 中的信息传递引入 prosody supervision signals 

## 方法

![](image/Pasted%20image%2020231219102021.png)

总体架构基于 FaseSpeech 2，引入 hierarchical graph prosody encoder 和一个预训练的 mel encoder。Graph encoder 输入为前一个句子、当前句子和下一个句子的语法图，然后学习 word-level、sentence-level 和 cross-sentence 的context prosody representations。

预训练的 mel encoder 通过 coustic prosody signals 来监督韵律建模，从而可以捕获很高的韵律变化范围。

backbone 则基于层级韵律来产生 mel 谱，最终合成音频。

### 带有语义信息的图结构

从语法🌲中构造图，主要是在 word level 做 dependency parsing 来实现。
> 中文中，最小的韵律单位为词，大多数的词都由一个或者两个 character 组成。

通过引入语法信息，能够更有效地利用语法信息来捕获韵律变化。然后还引入了一个 virtual global node 来连接 🌲 中的所有节点：
![](image/Pasted%20image%2020231219102357.png)

语法🌲可以通过有向图来表示，$\mathcal{G}=(\mathcal{V},\mathcal{E})$，每个节点 $v\in \mathcal{V}$ 用于表示句子中的 word，每条边记为 $e=(v_i,v_{j)}\in \mathcal{E}$，用于表明节点之间的关系。本文同时引入 virtual global $v_g$ 用于在所有的节点之间建立联系，即 $e_{v_{g},i}=(v_{g},v_{i})\mid\forall v_{i}\in\mathcal{V}$。

采用从预训练的中文 BERT 模型中提取的 embedding 来初始化节点。
> 对于 word，就是提取每个 character 的 embedding 然后做平均。

对于 global node，用的是 CLS token 对应的 embedding。

### 基于图的层级韵律建模

![](image/Pasted%20image%2020231219104444.png)

提出基于 图 的层级韵律建模，输入前一个句子、当前句子和下一个句子的图，分别捕获句间和句内的 context。

首先采用 Gated Graph Transformer 将图 $\mathcal{G}$ 编码到 $\mathcal{G}^\prime$，包含 $N$ 层的 grap transformer，主要就是对 node feature 采用自注意力机制进行 加权聚合，其中第 $i$ 个特征的更新如下：
$$\mathbf{x}_i^{\prime}=\mathbf{W}_1\mathbf{x}_i+\sum_{j\in\mathcal{N}(i)}\alpha_{i,j}\left(\mathbf{W}_2\mathbf{x}_j+\mathbf{W}_3\mathbf{e}_{ij}\right),$$
其中，$\mathcal{N}(i)$ 表示其邻近节点，$\alpha_{i,j}$ 表示可学习的权重。

编码之后，通过拼接操作来将两个图融合为一个图（对应图中的 concat 模块）。
> 作者认为，编码后的图包含 acoustic prosody 信息。从而融合后的图既包含来自文本的语义信息，又包含 韵律信息。

此外，引入的 global node 可以使得模型更好地理解整句结构，，这里的 global node 可以看成是 sentence level 的 prosody representation，而其他的 node 则作为 word-level 的 prosody representation。

为了捕获跨句子之间的韵律，首先将 三个句子的 sentence-level 的表征进行拼接，得到所谓的 Context Aggregation Features (CAF)，把它作为 K 和 V 来进行 contextual attention，而当前的 sentence-level 标准作为 Q。
> 这样就可以计算句间的 attention。

为了用上这些韵律表征，将 word level 的表征对齐到 phoneme sequence 中，然后 repeat sentence-level 和 cross-sentence prosody representations 来得到和 phoneme 序列相同的长度。

### 来自 acoustic prosody 的层级监督

为了捕获韵律变化，采用预训练的 mel-encoder 来监督学习到的 prosody 表征。
> mel encoder 可以从 mel 谱 中提取 sentence-level acoustic prosody embedding。

然后采用 MSE loss 来监督 graph encoder 得到的 sentence-level
prosody representations。

## 实验（略）

