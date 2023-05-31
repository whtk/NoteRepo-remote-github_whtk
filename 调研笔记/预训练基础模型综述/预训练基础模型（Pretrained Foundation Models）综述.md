> 来自论文 - A Comprehensive Survey on Pretrained Foundation Models: A History from BERT to ChatGPT

1. PFM 是指在不同数据类型、不同下游任务中的基础模型，如 BERT、GPT-3、MAE、DALLE-E和ChatGPT，其基于大规模的数据进行训练，同时广泛应用于各类下游任务
2. 本文全面回顾了最新的进展、挑战和机遇，首先回顾一些基础和现有的预训练方法，然后讨论了其他高级的 PFM 和不同数据模态下的 PFM，研究了一些关于 PFM 的模型效率、模型压缩、安全和隐私的问题，最后阐述了关键意义、未来方向、挑战等

## Introduction
1. Fundation Model：来自 2021 的一篇研究报告，On the Opportunities and Risk of Foundation Models，简单来说就是大模型，翻译为基础模型或者基石模型
2. PFM 主要研究在三个领域：NLP、CV、图学习（Graph Learning）：![[Pasted image 20230301200722.png]]主要思想就是基于大量数据学习通用的语义表征。
3. 多模态 PFM 正在融合，组成一个所谓的 unified PFMs，本文也回顾了其中的一些进展
4. PFM 的优点在于：第一，只要进行轻微的 fine tune 就可以提高下游任务的性能；第二，PFM 在生成质量方面还不错

## 基本组成

PFM 通用的概念性架构：![[Pasted image 20230301201855.png]]

### Transformer
Transformer 是一个纯注意力的模型，不依赖于递归或者卷积网络。

在 NLP 中，Transformer 可以用以解决 顺序输入的数据中长期依赖性问题，如 GPT3。

在 CV 中，ViT 将图像表示为成一系列的 patch ，类似于词嵌入。

在 GL 中，Graph Transformer Networks（GTN）在没有领域知识的情况下学习新的图结构和节点表征。


### PFM 中的学习机制
1. 监督学习：给定训练数据和标签，最小化目标函数来学习一个 function。
2. 半监督学习：除了有标签数据，还有无标签数据，通过数据的类间距和代理任务从数据中产生伪标签
3. 弱监督学习：处于监督学习和自监督学习（SSL）之间的，数据集中存在部分不正确或者不精确的标签
4. 自监督学习：利用数据本身中的信息来学习不同任务的表征，如 VAE 或者 GAN 是两种生成式 SSL 方法，对比学习是一种判别式的方法等
5. 强化学习：将学习过程建模为代理和环境之间的有序交互，agent 的目标是每个 状态 的长期回报的期望

### PFM 的预训练任务

### NLP 中的预训练
1. 掩码语言建模：mask 一部分单词然后训练模型来预测这些单词
2. 去噪自编码器：对数据集添加噪声，然后用带噪的数据来重构原始数据
3. 替换单词检测：用于判断语言模型是否替换了当前的 token
4. 下一句预测：输入两个句子判断是不是上下句
5. 顺序预测：将两个连续的句子作为正样本，然后交换其顺序作为负样本来建模句子之间的相关性

#### CV 中的预训练
1. 特定代理任务：创建代理任务，借助代理任务的标签来训练，如 inpainting 用于修复图像中的缺失部分
2. 帧顺序学习：从视频中学习帧之间的顺序
3. 数据生成：通过 GAN 将数据映射到隐空间
4. 数据重构：借鉴 NLP 中的，对掩码进行预测
5. 其他：如基于对比学习

#### GL 中的预训练
1. 图信息补全：mask 图中的部分信息，基于剩下的来恢复这部分的信息
2. 图属性预测：通过挖掘图中的潜在特征来作为自监督的标签
3. 图一致性分析：最大化具有相似语义信息的样本之间的一致性
4. 其他：将多个代理任务整个到一个统一的框架中等等

总之：SSL 很有前途，RL 是一种很新颖的 fine tune 方法。

## 用于 NLP 的 PFM
首先介绍学习词嵌入的模型，包括 autoregressive language model、contextual LM、permuted LM。总结了模型增强、多任务学习和不同下游任务的增强方法，最后介绍了  instruction-aligning 技术。

### 词嵌入方法

#### Autoregressive Language Model
基于前面的单词预测下一个单词，或者基于后面的单词预测上一个单词。对于词序列 $T=\left[w_1, w_2, \ldots, w_N\right]$，序列出现的概率为：$$p\left(w_1, w_2, \ldots, w_N\right)=\prod_{i=1}^N p\left(w_i \mid w_1, w_2, \ldots, w_{i-1}\right)$$
GPT 就采用了 自监督预训练+有监督 fine tune 的方法，使用 Transformer 作为解码器。

GPT2 增加了 Transformer 的层数，同时引入多任务学习。其模型容量很大， 可以针对不同的任务进行调整而非 fine tune。不过特定的下游任务仍需特定的数据集进行 fine tune。

GPT3 进一步增加模型参数和训练数据，无需针对下游 任务进行 fine tune就可以实现很好的性能。

#### Contextual Language Model
Autoregressive 仅仅使用了 单边的信息，而不能同时使用上下文的。如 ELMO 仅使用 双向 LSTM。contextual LM 基于上下文单词进行预测，使用 Transformer encoder，上下层都通过注意力机制直接相连。此时的概率计算为：$$p\left(w_1, w_2, \ldots, w_N\right)=\prod_{i=1}^N p\left(w_i \mid w_1, w_2, \ldots, w_N\right)$$
BERT 就使用了堆叠的多层、双向、Transformer 作为基本结构，模型输入包含三个部分：词嵌入、段嵌入和位置嵌入。采用双向的 Tramsformer 提取特征，但是本质仍然是自编码模型，对计算资源较低的设备不友好。

RoBERTa 使用更大的未标记数据，训练时间更长，添加长序列训练，同时采用 BPE 进行分词。

#### Permuted Language Model
自编码模型在 NLG 的任务中性能较差。Permuted LM 结合自回归 LM 和自编码 LM 的优点，pLM 的目标函数形式为：$$\max _\theta \mathbb{E}_{z \sim Z_N}\left[\sum_{t=1}^N \log p_\theta\left(x_{z_{T=t}} \mid x_{z_{T<t}}\right)\right]$$
其中，$\theta$ 是在所有的 permutation 中共享的参数，$Z_N$ 表示输入序列 $T$ 的所有可能的 permutation 的集合，$z_{T=t}, z_{T<t}$ 分别表示permutation $z$ 中的第 $t$ 个元素和 $[1,2,\dots,t-1]$ 之间的元素。

因为 BERT 在训练的时候使用 mask，但是在 fine tune 的时候不使用，导致两个时期的数据不一致。pLM 则可以同时实现双向编码同时避免 mask 的问题，其不再对数据按顺序建模，而是给出数据的所有排列，然后最大化数据的对数似然，从而任何位置都可以利用来自所有位置的上下信息。

常见的模型是 XLNET 和 MPNet。

### 模型架构设计方法
ELMO 采用多层 RNN 架构，每一层都是双向的 LSTM。正向和反向的最大似然作为训练的目标函数，相比于 word vector，ELMO 引入上下文信息 可以改善 多义问题，但是整体能力偏弱。

PFM 应用研究有两个方向：
+ 带 fine tune 的，如 BERT
+ 有 zero/few-shot prompts 的，如 GPT

BART 采用 编码器-解码器结构的 seq2seq 模型构建降噪自编码器：![[Pasted image 20230302100319.png]]
预训练的时候，对文本添加噪声，然后使用 seq2seq 模型重建原始文本。encoder 是双向的 Transformer，decoder 根据 encoder 输出的编码表征和没有被 mask 的序列来重构原始序列。

#### mask 的设计方法
注意机制首先将重点单词聚合为句子向量，将重要句子向量聚合为文本向量，这允许模型对不同的输入给予不同的关注。

基于 RoBERTa 的 SpanBERT 采用 dynamic masking 和 single segment pretraining 技术，还提出了跨度掩码和跨度边界目标（SBO）来屏蔽一定长度的单词：![[Pasted image 20230302102253.png]]
BERT 和 GPT 只能在没有联合训练的情况下分离训练编码器和解码器。Song等人提出 掩码 seq2seq 预训练模型 MASS，在训练阶段，编码器的输入序列被随机掩码为长度为 $k$ 的连续段，这些段将通过 MASS 的解码器恢复。
UniLM 通过为输入数据中的两个句子设计不同的掩码来完成 NLG 模型的学习。



#### 提升方法

1. 提升模型性能：大多数预训练模型都需要大量数据，这往往很难实现。百度发布的ERNIE Tiny是一个小型化的ERNIE，将预测速度提高了4.3倍；Lan等人提出ALBERT 也减少了内存消耗，提高了训练速度，且没用性能损失。
2. 提升多任务学习的性能：ERNIE 主要由两部分组成，Transformer编码器和任务嵌入。Transformer编码器使用自注意力来捕获每个 token 的上下文信息并生成上下文表征，任务嵌入将不同特征应用于任务，通过引入多任务学习来实现词汇、语法和语义的预训练。UniLM 使用三个预训练任务：单向LM、双向LM 和 编码器-解码器 LM。通过自注意层的掩码机制在预训练阶段同时完成三种目标任务。
3. 提升不同下游任务的性能：预训练模型匹配下游任务是非常重要的。BERT-WWM 直接使用在中文中使用 BERT 来进行 MLM 训练，但是会导致语义信息的丢失。ZEN 是一种基于BERT的文本编码器，它采用N-gram来增强性能，并以快速的收敛速度和良好的性能集成了大量文本信息。Tsai等人 提出了一种面向序列标记任务的多语言序列标记模型。
4. 示例：![[Pasted image 20230302103558.png]]ChatGPT基于PFM GPT-3.5使用RLHF进行 fine tune。与InstructGPT相比，ChatGPT使用不同的数据收集设置：
	1. 收集包含提示的大型数据集用于通过监督学习 fine tune GPT-3.5
	2. 给定 fine tune 后的模型和 prompt，模型生成多个输出。labeler 对这些数据进行打分和排序，构成新的数据集，用于训练 reward模型
	3. PPO 算法根据 reward 模型 优化 chatGPT

#### 指令对齐
目的是让LM遵循人类意图产生有意义的输出。一般通过监督的方式用高质量的语料库 fine tune 预训练的LM。也有使用 RL 进一步提高性能的。监督和RL方法都可以利用 chain-of-thought 推理来提高 AI 决策的透明度。

1. 有监督的 fine tune（SFT）：SFT 由输入输出和指令构成
2. 基于反馈的RL：RL 已经被用于增强NLP任务中的各种模型，其有助于优化语言生成任务中不可微目标函数，将这些问题看成是顺序决策问题，不过存在过拟合的风险。如 InstructGPT，但是比较难做
3. Chain-of-Thoughts (CoT)：CoT是一系列中间推理步骤，可以显著提高大型LMs执行复杂推理的能力

## 用于 CV 的 PFM

## 用于 图学习的 PFM

## 其他数据模态

### 语音

### 视频

### 多模态

### SOTA Unified


## PFM 进阶

### 模型效率

### 模型压缩

### 安全和隐私

## 研究挑战和一些开放性问题

## 总结
