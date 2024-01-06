> MIT、Google，ASRU 2021

1. 提出 w2v-BERT，将 MLM 用于自监督语音表征学习
2. 将对比学习和 MLM 结合，前者将语音离散化为 token，后者通过 masked prediction 任务学习语音上下文表征
3. 可以以端到端的方式进行训练
4. 也是在 ASR 任务中做的实验

## Introduction

1. 有两种主流的方法利用无标签数据解决半监督 ASR 任务：
	1. 第一种是 self-training（pseudo-labeling），用有标签的数据训练 teacher model，然后用 teacher model 标注无标签数据，最后得到的数据用于训练 student model
	2. 第二种是自监督预训练，模型训练用于完成代理任务
2. 本文关注于，提升无监督预训练，提出一种新的预训练框架，w2v-BERT，将 wav2vec 2 and BERT 结合起来

框架如图：
![](image/Pasted%20image%2020231116224120.png)

本文贡献如下：
+ 提出 w2v-BERT，可以同时优化对比损失和掩码预测损失
+ 在 LibriSpeech 上实现 SOTA 性能
+ 在真实场景下的识别性能超过 wav2vec 2.0
+ 经验上证明掩码预测在对比学习中的重要性

## 相关工作（略）

## 方法

### 模型架构

包含：
+ feature encoder，从原始声学输入中提取 latent 语音表征，由卷积层组成，输入为 log-mel 谱 ，输出为 latent 语音表征
+ 一个用于获取 离散 token 集合的 对比模块，包含 线性层+Conformer，目的是将 feature encoder 的输出离散为一堆 units，其中就会包含量化模块，feature encoder 的输出会通过两部分：
	+ 一方面 进行 mask 然后通过 线性+Conformer 层 来产生 context vector
	+ 另一方面不 mask，直接通过 quantizer 得到 quantized vector
+ 一个用于解决掩码预测任务的模块，Conformer block 堆叠，输入为 context vector，提取 high-level contextualized  语音表征

### 预训练

预训练时采用了无标签的语音数据。

#### 对比损失

用于训练 对比模块 和 quantizer，对比模块得到 context vector，用于作为 context vector 的输入，而 quantizer 产生 discriminative 的 离散 token 用于掩码预测 的 target。 用的是 [wav2vec 2.0- A Framework for Self-Supervised Learning of Speech Representations 笔记](wav2vec%202.0-%20A%20Framework%20for%20Self-Supervised%20Learning%20of%20Speech%20Representations%20笔记.md) 中的对比任务。
> 到这里其实都和 wav2vec 2.0 的一样，使得连续的 表征 和 其对应的离散的 token 尽可能相近，不对应的 尽可能远离。

但是和 wav2vec 2.0 中 mask 用的是 learnable vector 不一样，这里用的是随机向量。

具体而言，对于给定 context vector $c_t$（对应于 masked time step 的 $t$），模型需要判断对应的 GT quantized vector $q_t$，而 distracors 有 $K$ 个，是从相同语音的其他 masked time step 中随机选取的。定义此损失为 $\mathcal{L}_w$，然后还有一个 diversity loss $\mathcal{L}_d$ 使得 code 可以被均匀使用，总的对比损失为：
$$\mathcal{L}_c=\mathcal{L}_w+\alpha\cdot\mathcal{L}_d$$

#### 掩码预测损失

对比模块得到的 context vector 会通过 掩码预测模块 产生最终的 context vector 来实现掩码预测。

模块最后一层有一个 softmax 层，如果某个位置是 mask 的，softmax 的输入为该位置的 context vector，然后用于预测其对应的 token ID（即之前 quantizer 给出的），即这个过程的交叉熵损失为 $\mathcal{L}_{m}$。

此时，总损失为：
$$\mathcal{L}_p=\beta\cdot\mathcal{L}_c+\gamma\cdot\mathcal{L}_m$$

### fine tune

这个阶段有 有标签 的数据，在两个数据集（任务）上训练，LibriSpeech 和 voice search。

## 实验（略）