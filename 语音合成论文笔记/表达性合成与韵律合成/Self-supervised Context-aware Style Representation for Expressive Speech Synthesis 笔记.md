> Interspeech 2022，人大、微软 Azure

1. audiobook synthesis 中，从 reference audio 或从文本预测 style tags 需要大量标记数据，仍然很有挑战
2. 提出从大量的 plain text 中，以自监督的方式学习 style representation，采用了 emotion lexicon、对比学习和 deep clustering
3. 将 style representation 作为 conditioned embedding，整合到 multi-style Transformer TTS 中，可以在 in-domain 和 out-of-domain test set 上取得更好的结果
4. 在隐式的 context-aware style representation 下，长段落中合成音频的情感过渡更自然

> 主要还是在文本方面做的工作，通过文本数据来提取文本中的情感 embedding。

## Introduction

1. TTS 在 audiobook 合成中，有时会出现 over-smoothing prosody pattern 的问题，原因之一是很难建模高级特征，如情感和上下文变化
2. 两种解决方法：
    1. unsupervised joint training：基于 reference audio 和文本联合训练，得到隐式的 style representation space，推理时，从 reference audio 或联合训练过程中预测的 style 中推断 style representation
    2. supervised label conditioning：利用显式标签作为辅助信息，引导 multi-style TTS，不需要 reference audio，但需要大量标注数据，且简单的离散 tag 不能完全反映语音风格的差异
3. 本文不通过 reference audio 或者显示的 tag 来建模 style，提出从 plain text 中以自监督的方式学习 style representation。然后集成到端到端的 TTS 中：
    1. 先用对比学习，通过区分相似和不相似的 utterances 来预训练 style embedding
    2. 通过最小化 deep clustering loss、重构 loss 和对比 loss，对样本在 style embedding space 中进行聚类
    3. 从大量无标签的 plain text 数据中学习 style representation，构建 text sentiment embedding space，来指导 multi-style expressive audio 合成
4. 贡献如下：
    1. 提出了一个新的 framework，从无标签文本中建模 style representation，并将其整合到基于 style 的 TTS 模型中，不需要 reference audio 或显式 style labels
    2. 提出了一个新的两阶段 style representation 学习方法，结合了 deep embedded clustering 和 contrastive learning，通过 emotion lexicon 进行数据增强
    3. 实验结果表明效果优于 baseline，尤其在长音频生成中的情感过渡自然度上

## 相关工作（略）

> 主要和 对比学习 和 deep clustering 有关。

## 方法

### 问题描述、系统概述

给定文本数据集 $\mathcal{D}=\{U_i\}_{i=1}^D$，其中 $U_i= \{C_i^-,u_i,C_i^+\}$ 表示第 $i$ 个 utterance 和其上下文。$C_i^-=\{u_{i-m},...,u_{i-1}\}$ 是 $u_i$ 的前面 utterances，$C_i^+=\{u_{i+1},...,u_{i+m}\}$ 是 $u_i$ 的后面 utterances。目标是从 $\mathcal{D}$ 中学习 style encoding 模型 $s_i=g(U_i)$，为 utterance $u_i$ 生成一个 context-aware style representation。这个 style model 将应用到 TTS 系统中，以提高语音合成的表达性。

框架如图：
![](image/Pasted%20image%2020240421104718.png)

1. 先构建 $u_i$ 和其增强样本 $\tilde{u}_i$ 的正样本对，即：将 emotion arousal 最强的词（即最能体现情感的词）替换为其同义词。然后设计一个 style encoder 用对比学习进行预训练。
2. 采用对比学习来进一步训练 style encoder 以增强了 Deep Embedded Clustering 方法。最后学习 $g(U_i)$，然后将生成的 representation 称为 Context-aware Augmented Deep Embedded Clustering (CADEC) style
3. 将其作为 conditioning embedding 输入到 Transformer TTS 中，生成表达性音频

### 阶段 1：基于数据增强的对比学习

一个 utterance 可以用多种风格表达。utterance $u_i$ 的风格 与 上下文、语义内容和情感 相关。将 $u_i$ 和其上下文 $U_i$ 作为输入，结合 content 特征和 emotion 特征来建模风格。

使用预训练的 BERT 作为 backbone 来提取 content 信息，采用额外的 emotion lexicon 来提取 emotion 特征。
> lexicon 提供 word-level 的 emotion 特征，包含 VAD（valance, arousal, dominance）和 BE5（joy, anger, sadness, fear, disgust）。

提取初始的 style embedding $r_i$：
$$r_i=b(U_i)\oplus\frac1M\sum_{j=1}^Me(w_j)$$

其中 $\oplus$ 表示拼接，$b(U_i)$ 是将 $U_i$ 输入到 BERT 中的 CLS 对应的 embedding，$M$ 是 $U_i$ 中的单词数，$w_j$ 是 $U_i$ 中的第 $j$ 个单词，$e(w_j)$ 表示其归一化的 BE5 特征，是一个 5 维向量。

然后用 MLP 作为 encoder 来映射初始 embedding 到 hidden features，得到 output style embedding：
$$h_i=MLP(r_i)$$

采用数据增强和对比学习来预训练 encoder。

为了增强 $u_i$，得到类似风格的 $\tilde{u}_i$，先将 $u_i$ 分割成短片段，然后查找 emotion lexicon，获取每个片段中每个单词的 emotion arousal，并选择 top $k\%$，如 $20\%$，用其 WordNet 同义词替换。
> 例如，将 $u_i$ 分成两个片段，选择第一个片段中的 “lucky” 和 “happy”，第二个片段中的 “annoying” 和 “crazy”，然后用其同义词组成 $\tilde{u}_i$。

将长句子分成片段的目的是从不同片段提取情感词，避免只关注某个片段的主导情感词。

对于对比学习，从大型训练数据集 $\mathcal{D}$ 中，随机采样一个 minibatch 数据 $\mathcal{B}\:=\:\left\{U_i\right\}_{i=1}^N$，生成其增强数据 $\tilde{\mathcal{B}}\:=\:\{\tilde{U}_i\}_{i=1}^N$，其中 $\tilde{U}_i\:=\:\{C_i^-,\tilde{u}_i,C_i^+\}$。$U_i$ 和 $\tilde{U}_i$ 视为正样本，其他 $N-1$ 对 $\{<U_i,U_k>\}_{i\neq k}$ 都是负样本。为了最大化具有相似情感的文本之间的一致性，以及不同情感的文本之间的差异性，采用 SimCLR 中的方法，计算样本间的对比损失：
$$l_c^i=-log\frac{exp(cos(h_i,\tilde{h_i})/\tau)}{\sum_{k=1}^N\mathbb{1}_{k\neq i}exp(cos(h_i,\tilde{h_k})/\tau)}$$

其中 $\tau$ 是 temperature parameter，$\mathbb{1}_{k\neq i}$ 是指示函数。minibatch 的对比损失通过对 $B$ 和其增强数据 $\tilde{B}$ 中的所有样本求平均：
$$\mathcal{L}_{contrastive}=\frac1N\sum_i^N\ell_c^i$$

### 阶段 2：基于 Autoencoder 的深度嵌入聚类

采用 deep embedded clustering 和 autoencoder 来进一步训练 CADEC style encoder。聚类数 $K$ 是 prior，每个 cluster 由其质心 $\mu_k$ 表示。聚类损失定义为： 
$$\mathcal{L}_{clustering}=KL(P\|Q)=\sum_i\sum_kp_{ik}log\frac{p_{ik}}{q_{ik}}$$

其中 $P$ 是 $Q$ 的目标分布。采用 Student’s t-distribution 来计算将 $h_i$ 分配给第 $k$ 个 cluster 的概率 $q_{ik}$：
$$q_{ik}=\frac{\left(1+\left\|h_i-\mu_k\right\|_2^2/\alpha\right)^{-\frac{\alpha+1}2}}{\sum_{k^{'}=1}^K(1+\left\|h_i-\mu_{k^{'}}\right\|_2^2/\alpha)^{-\frac{\alpha+1}2}}$$

其中 $\alpha$ 表示 Student’s t-distribution 的自由度。这里设置 $\alpha=1$。目标分布 $p_{ik}$ 为：
$$p_{ik}=\frac{q_{ik}^2/\sum_iq_{ik}}{\sum_{k^{\prime}}(q_{ik^{\prime}}^2/\sum_iq_{ik^{\prime}})}$$

这种聚类会扭曲原始的表示空间，削弱隐式特征的表示能力，因此添加一个 autoencoder，通过重构损失来保留特征空间的局部结构，避免损坏：
$$\mathcal{L}_{reconstruction}=\sum_{i=1}^N\left\|r_i-r_i^{'}\right\|_2^2$$

阶段 2 的总的目标函数为：
$$\mathcal{L}_{total}=\mathcal{L}_{contrastive}+\beta\mathcal{L}_{clustering}+\gamma\mathcal{L}_{reconstruction}$$


### TTS 阶段：基于 style representation 的 Transformer TTS

拓展 [Transformer-TTS- Neural Speech Synthesis with Transformer Network 笔记](../Transformer-TTS-%20Neural%20Speech%20Synthesis%20with%20Transformer%20Network%20笔记.md) 模型，把从 style encoder 中生成的 CADEC style embedding 作为其条件，和 phoneme 序列一起作为模型的输入。

## 实验（略）
