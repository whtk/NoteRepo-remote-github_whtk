> arxiv 2023，Google

1. 提出 AudioPaLM，实现语音理解和生成的 LLM：
+ 将文本 LLM PaLM-2 和语音 LLM AudioLM 融合为统一的多模态架构，实现文本和语音生成
+ AudioPaLM 可以保留音色信息，如说话者身份和语调，以及 PaLM-2 中的 linguistic knowledge
2. 表明使用 text-only LLM 的权重初始化 AudioPaLM，可以提高 speech processing 性能

> 从 AduioLM 中拓展多任务 + 多语种。

## Introduction

1. 将 text 和 audio vocabularies 合并为多模态的 single vocabulary，可以训练单个模型实现双向转换
2. 本文提出 AudioPaLM，多模态语音和文本生成模型：
1. 核心是 joint vocabulary，用数量有限的离散 token 表示语音和文本
2. 包含一个任务的 elementary markup description，可以在语音文本混合的任务上训练单个 decoder-only 模型，包括 ASR、TTS、STT
3. 用 Transformer 作为 backbone，可以用 text LLM 的权重初始化，从而获得语言和常识知识

## 相关工作（略）

## 方法

采用 decoder-only Transformer 建模 text 和 audio tokens 序列，一旦输入被 tokenized，输出被 detokenized，模型将 text 和 audio 视为整数序列。原则上纯文本的 decoder-only 几乎没有区别，只是这里的一些 token 代表 audio，一些代表 text。本文用预训练的 text-only 模型初始化多模态模型。
模型结构如图：
![](image/Pasted%20image%2020241012104202.png)

### Audio Embeddings 和 Tokenization

采用 [On generative spoken language modeling from raw audio 笔记](../语音自监督模型论文阅读笔记/On%20generative%20spoken%20language%20modeling%20from%20raw%20audio%20笔记.md) 和 [AudioLM- a Language Modeling Approach to Audio Generation 笔记](AudioLM-%20a%20Language%20Modeling%20Approach%20to%20Audio%20Generation%20笔记.md) 的方法，将原始波形转为 token，下面是几种方法：
+ 采用 AudioLM 中的 w2v-BERT 模型，但是用的是在 multilingual 数据上训练的（AudioLM 中的仅在英文数据训练）；同时不对 embeddings 进行归一化，因为在多语言中的归一化会导致性能下降；最终得到 25Hz 的 token，vocab 大小为 1024
+ 用 USM-v1 替代 w2v-BERT，从中间层提取 embeddings，得到 25Hz 的 token，vocab 大小为 1024
+ 用 USM-v2，用辅助 ASR loss 进行 finetune 从而在多语言上表现更好

### 修改 text-only decoder 以建模文本和语音

Transformer decoder 的第一层是 token embeddings 矩阵 $\mathbf{E}$，将整数 token 映射为 embeddings；$\mathbf{E}$ 是一个 $t \times m$ 的矩阵，$t$ 表示 token 数量，$m$ 表示 embeddings 大小。另一个 embeddings 矩阵 $\mathbf{E}'$ 出现在 softmax 层，用于计算每个位置上所有 token 的 logits；$\mathbf{E}'$ 是一个 $m \times t$ 的矩阵，与模型的 $m$ 维输出相乘得到 $t$ 维 logits。在 PaLM 架构中，这两个矩阵共享变量，即 $\mathbf{E}' = \mathbf{E}^T$。

为了将 text-only 模型转为 text 和 audio 模型，只需要扩展 embeddings 矩阵 $\mathbf{E}$ 的大小为 $(t + a) \times m$，其中 $a$ 是 audio token 数量（$\mathbf{E}' = \mathbf{E}^T$ 的大小相应变化）。

为了使用预训练的 text 模型，在 embeddings 矩阵 $\mathbf{E}$ 中添加新的行。具体来说，前 $t$ 个 token 对应 SentencePiece text tokens，接下来的 $a$ 个 token 代表 audio tokens。
> 这里虽然可以重用预训练模型的 text embeddings，但是新的 audio embeddings 还是要重新初始化训练。作者发现有必要训练所有模型参数，而不是固定之前的权重。

使用语音和文本混合数据的任务进行训练。

### 将 audio tokens 解码为原始音频

为了从 audio tokens 合成音频波形，尝试两种方法：
1. 自回归解码，类似 AudioLM 的方法
2. 非自回归解码，使用 [SoundStorm- Efficient Parallel Audio Generation 笔记](SoundStorm-%20Efficient%20Parallel%20Audio%20Generation%20笔记.md) 模型

AudioLM 中的 acoustic generation 分为两个阶段：
1. Stage 2：decoder-only Transformer 模型输入 audio tokens 和 voice conditioning，生成 SoundStream tokens，比特率低
2. Stage 3：重构 SoundStream 的 residual vector quantizer，增加比特率，提高音频质量

SoundStorm 提出了一种非自回归解码方案，采用迭代方法并行处理所有 token，生成的音频质量与 AudioLM 相同，速度快两个数量级。

两种方法都在 Multilingual LibriSpeech 上训练，voice conditioning 为 3 秒长的语音。通过给定原始输入语音的一部分作为 voice conditioning，模型可以在将其翻译为不同语言时保留原说话人的声音。

### 训练任务

任务包括 ASR、TTS 和 STT。所有数据集都是 speech-text 数据集，包含以下字段：
+ Audio：源语言的语音
+ Transcript：Audio 的文本
+ Translated audio：Audio 的翻译
+ Translated transcript：Audio 的文本翻译

任务有：
+ ASR：将 audio 转为 transcript
+ AST：将 audio 翻译为 translated transcript
+ S2ST：将 audio 翻译为 translated audio
+ TTS：将 transcript 读出为 audio
+ MT：将 transcript 翻译为 translated transcript

实验中发现，包含多个任务（如 ASR 和 AST）可以提高性能。

通过在输入前加上 tag 指定任务，让模型知道执行哪个任务，同样也指定输入和输出的语言。

例如，要让模型在法语上做 ASR，输入的 tokenized audio 前面加上 [ASR French]。对于 TTS，文本前面加上 [TTS English]。如果是 S2ST 从英语到法语，英语音频前面加上 [S2ST English French]。tag 用模型的文本 tokenizer 进行 tokenization，不引入特殊 token 表示任务或语言，并不会影响模型性能。而且在任务名称中加入语语种，对于低资源语言效果更好。


考虑直接任务和组合任务，组合任务要求模型输出复杂任务的中间步骤，类似于 chain of thought prompting。

例如，对于 S2ST，可以直接从英语音频 token 映射到法语音频 token，用 [S2ST English French] 表示。或者先输出英语文本，然后法语文本，最后法语音频 token，用 [ASR AST S2ST English French] 表示。模型将其作为单个自回归解码任务，而不是多个任务的分开调用。从而可以在每个阶段看到输入和之前的所有解码，而不是分开做 ASR、MT 和 TTS。
> 实验中发现组合任务可以提高性能。

### 训练混合

用的数据如下表：
![](image/Pasted%20image%2020241012114427.png)

有两个混合数据集：一个用于训练文本作为输出的 ASR 和 AST 任务的模型，另一个用于训练文本和语音同时作为输出的 TTS 和 S2ST 任务的模型。

### 训练设置

所有实验中，采用相同的 finetuning 设置，使用 Adafactor 优化器，学习率为 $5 \times 10^{-5}$，dropout 为 $0.1$，输入使用 loss masking。

## 数据和评估指标（略）

## 实验（略）
