> Google，TASLP，2023

1. 提出 AudioLM，一种高质量的音频生成框架，将输入音频映射为离散的 token 序列，然后在离散表征空间下，把音频生成任务看成是一种 language modeling 的任务
2. 提出一种 hybrid tokenization scheme 可以实现  reconstruction quality 和 long-term structure 之间的 trade-off
3. 即，利用音频预训练语言模型来捕获 long-term structure，然后利用 neural audio codec 产生的离散编码来合成高质量音频

## Introduction

1. 之前的模型只能生成一些 unstructured audio；而 language model 已经可以建模不同内容的 high-level、long-term structure
2. 本文提出 AudioLM，可以实现 high-quality audio generation with long-term coherent structure，将神经音频压缩、自监督表征学习和语言建模等方法结合：
	1. 使用预训练的语言模型从原始波形先构造 coarse semantic tokens
	2. 但是直接用这些 token 重构波形效果不好
	3. 于是使用 SoundStream 来生成 fine-level acoustic tokens 来捕获波形的细节
	4. 训练一个语言模型同时生成 semantic and acoustic token，从而可以实现 high audio quality and long-term consistency
3. 贡献如下：
	1. 提出 AudioLM，将  semantic and acoustic tokens 以 hierarchical 的方式进行组合
	2. 比较了来自 w2v-BERT 的 semantic tokens 和 来自 SoundStream 的 acoustic token，发两者是互补的
	3. 在没有文本标注的情况下，AudioLM 可以生成 honetics, syntax and semantics 一致的音频
	4. AudioLM 还可以用于音乐生成
	5. 训练了一个 classifier，可以区别 AudioLM 生成的音频

## 相关工作（略）

## 模型

### 组成

考虑单通道音频 $x\in\mathbb{R}^T$，然后通过以下三个组件处理：
+ tokenizer：将 $x$ 映射到序列 $h=\mathrm{enc}(x),h=(h_1,\ldots,h_{T^{\prime}})$ 为离散的 token 序列
+ decoder-only Transformer language model ：基于离散 token 来最大化似然 $\prod_{t=1}^{T'}p(h_t|h_{<t})$，推理时，模型自回归预测 $\hat{h}$
+ detokenizer：将 token 映射回波形 $\hat{x}=\det(\hat{h})$

tokenizer 和 detokenizer 是预训练的，只会训练 language model。

### 离散音频表征之间的 trade-off

![](image/Pasted%20image%2020230927172933.png)

采用 SoundStream 计算 acoustic tokens，配置其使得，对于 16KHz 的信号，产生 50 Hz 的 embedding。

采用  w2v-BERT 来计算 semantic token，选择 intermediate layer，计算其 embedding，然后采用 k-means 获得 $K$ 个聚类，把聚类中心作为 semantic token。
> 作者发现，提取的 token 和从 HuBERT 中提取的 token 很相似？

然后训练 SoundStream decoder 来重构波形，同时比较两种 token 的效果，指标是 ViSQOL。
实验表明，acoustic tokens 有较好的  reconstruction quality，但是  phonetic discriminability 较差，而 w2v-BERT 中的第 7 层的 semantic tokens 可以极大地改善 phonetic discriminability。

然后基于 acoustic tokens 训练 decoder-only Transformer，发现生成的语音可以保留 speaker identify，但是 linguistic content 不一致。

### Hierarchical modeling of semantic and acoustic tokens

首先在整个序列中建模 semantic tokens，然后把这些 semantic tokens 当作 condition 来预测 acoustic tokens，然后接下来有三个 stage，在所有的 stage，都使用独立的 ecoder-only Transformer 来预测下一个 token：
![](image/Pasted%20image%2020230927221221.png)

Semantic modeling：第一个 stage 建模 $p(z_t|z_{<t})$，自回归预测下一个 semantic token

Coarse acoustic modeling：仅仅预测来自 coarse $Q^\prime$ SoundStream quantizers 的 acoustic tokens（也就是第 $Q^\prime$ 个 RVQ 的输出，这一部分的量化比较粗糙）

Fine acoustic modeling：用 coarse $Q^\prime$ SoundStream quantizers 得到的 coarse tokens 作为 condition 来预测  fine token。

### 推理

基于不同的条件，有不同的生成方式。

无条件生成：无条件采样所有的 semantic tokens $\hat{z}$，然后把它作为 acoustic modeling 的 condition

Acoustic generation：使用从文本中生成的 semantic tokens $z$，其他的和无条件生成一样

连续生成：从一个短的 prompt $x$ 中生成后续音频，首先将 prompt 映射到 semantic tokens $z_{\leq t_s}$，然后生成部分 coarse acoustic tokens ，第一个 stage 就正常自回归生成，第二个 stage 把 全部的 semantic tokens 和前面部分的 coarse acoustic tokens 作为条件生成剩余的 coarse acoustic tokens。In the third stage, we process the coarse acoustic tokens with the fine acoustic mode（？？？）最后把 prompt 和 acoustic token 输入 SoundStream decoder 生成波形。

## 实验（略）
