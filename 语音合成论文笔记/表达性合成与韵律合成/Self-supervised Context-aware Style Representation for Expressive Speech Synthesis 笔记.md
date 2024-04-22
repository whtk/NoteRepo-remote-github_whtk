> Interspeech 2022，人大、微软 Azure
<!-- 翻译 & 理解 -->
<!-- Expressive speech synthesis, like audiobook synthesis, is still challenging for style representation learning and prediction. Deriving from reference audio or predicting style tags from text requires a huge amount of labeled data, which is costly to ac- quire and difficult to define and annotate accurately. In this paper, we propose a novel framework for learning style repre- sentation from abundant plain text in a self-supervised manner. It leverages an emotion lexicon and uses contrastive learning and deep clustering. We further integrate the style representa- tion as a conditioned embedding in a multi-style Transformer TTS. Comparing with multi-style TTS by predicting style tags trained on the same dataset but with human annotations, our method achieves improved results according to subjective eval- uations on both in-domain and out-of-domain test sets in au- diobook speech. Moreover, with implicit context-aware style representation, the emotion transition of synthesized audio in a long paragraph appears more natural. The audio samples are availableonthedemowebsite. 1 -->
1. audiobook synthesis 中，从 reference audio 或从文本预测 style tags 需要大量标记数据，仍然很有挑战
2. 提出从大量的 plain text 中，以自监督的方式学习 style representation，采用了 emotion lexicon、对比学习和 deep clustering
3. 将 style representation 作为 conditioned embedding，整合到 multi-style Transformer TTS 中，可以在 in-domain 和 out-of-domain test set 上取得更好的结果
4. 在隐式的 context-aware style representation 下，长段落中合成音频的情感过渡更自然

## Introduction
<!-- Although TTS models can synthesize clean and high-quality natural speeches, it still suffers from the issue of over- smoothing prosody pattern in some complex scenarios, as in audiobook synthesis. One of the reasons is the difficulty of modeling high-level characteristics such as emotions and con- text variations, which impact the overall prosody and speaking style. Being different with the low-level acoustic characteristics such as duration, pitch and energy, modeling high-level charac- teristics is more challenging and crucial in these complex sce- narios [1]. -->
1. TTS 在 audiobook 合成中，有时会出现 over-smoothing prosody pattern 的问题，原因之一是很难建模高级特征，如情感和上下文变化
<!-- There are two general approaches to deal with such tasks: unsupervised joint training and supervised label conditioning. The unsupervised approach models styles based on joint train- ing with both reference audio and text content [1, 2, 3, 4, 5]. By constructing an implicit style representation space in an unsu- pervised way, it infers a style representation from either the ref- erence audio or the predicted style by the joint training process. However, the joint training framework faces two challenges: 1) content information leaks into style encoder; 2) requiring a large number of audio and content pairs. Many recent studies in this area focus on these two issues [6, 7, 8]. In real applications, the supervised learning is more widely adopted by leveraging ex- plicit labels as auxiliary information to guide multi-style TTS [9, 10]. It does not require reference audio, but the definition of styles, which could be subjective. Predicting style tags also requires a large amount of annotated data. Moreover, a simple discrete tag cannot fully reflect the nuance in speech styles. -->
2. 两种解决方法：
    1. unsupervised joint training：基于 reference audio 和文本联合训练，得到隐式的 style representation space，推理时，从 reference audio 或联合训练过程中预测的 style 中推断 style representation
    2. supervised label conditioning：利用显式标签作为辅助信息，引导 multi-style TTS，不需要 reference audio，但需要大量标注数据，且简单的离散 tag 不能完全反映语音风格的差异
<!-- To address these problems, instead of modeling styles through reference audios or explicit tags, we propose a novel framework which learns the style representation from plain text in a self-supervised manner and integrates it into an end-to-end conditioned TTS model. First, we employ contrastive learning to pre-train style embedding by distinguishing between similar and dissimilar utterances. To this end, we create a similar ut- terance by replacing an emotional word by a similar one, deter- mined using an emotion lexicon. With the emotionally similar utterance as positive sample, all other dissimilar utterances in the randomly sampled minibatch are treated as negatives. Then training samples in style embedding space are clustered by min- imizing deep clustering loss, reconstruction loss and contrastive loss together. We learn the style representation from a large amount of unlabeled plain text data and construct a text senti- ment embedding space to guide the generation of multi-style ex- pressive audio in speech synthesis. Using it as a pre-training of style information, we can get rid of the dependence of matched audio and content. Our work has three main contributions: -->
3. 本文不通过 reference audio 或者显示的 tag 来建模 style，提出从 plain text 中以自监督的方式学习 style representation。然后集成到端到端的 TTS 中：
    1. 先用对比学习，通过区分相似和不相似的 utterances 来预训练 style embedding
    2. 通过最小化 deep clustering loss、重构 loss 和对比 loss，对样本在 style embedding space 中进行聚类
    3. 从大量无标签的 plain text 数据中学习 style representation，构建 text sentiment embedding space，来指导 multi-style expressive audio 合成
4. 贡献如下：
<!-- • Weproposeanovelframeworkformodelingstylerepresenta- tion from unlabeled texts and incorporate it into a style-based TTS model, without reference audio or explicit style labels.
• We propose a novel two-stage style representation learn- ing method combining deep embedded clustering with con- trastive learning based on data augmented via an emotion lex- icon.
• We demonstrate that with the same labeled text corpus and audiobook corpus, our speech synthesis outperforms the baseline, especially in naturalness of emotion transition in long audio generation. -->
    1. 提出了一个新的 framework，从无标签文本中建模 style representation，并将其整合到基于 style 的 TTS 模型中，不需要 reference audio 或显式 style labels
    2. 提出了一个新的两阶段 style representation 学习方法，结合了 deep embedded clustering 和 contrastive learning，通过 emotion lexicon 进行数据增强
    3. 实验结果表明效果优于 baseline，尤其在长音频生成中的情感过渡自然度上

## 相关工作（略）

> 主要和 对比学习 和 deep clustering 有关。

## 方法

### 问题描述、系统概述
<!-- Ui = and its context. Ci− = {ui−m,...,ui−1} is the preceding utterances of ui and Ci+ = {ui+1,...,ui+m} is the following utterances of ui. Our goal is to learn a style encoding model si = g(Ui) from D which can generate a context-aware style representation for ut- terance ui. This style model will be applied to a TTS system to  improve the expressiveness of speech synthesis.-->
给定文本数据集 $\mathcal{D}=\{U_i\}_{i=1}^D$，其中 $U_i= \{C_i^-,u_i,C_i^+\}$ 表示第 $i$ 个 utterance 和其上下文。$C_i^-=\{u_{i-m},...,u_{i-1}\}$ 是 $u_i$ 的前面 utterances，$C_i^+=\{u_{i+1},...,u_{i+m}\}$ 是 $u_i$ 的后面 utterances。目标是从 $\mathcal{D}$ 中学习 style encoding 模型 $s_i=g(U_i)$，为 utterance $u_i$ 生成一个 context-aware style representation。这个 style model 将应用到 TTS 系统中，以提高语音合成的表达性。
<!-- Figure 1 shows our proposed framework. First, we con- struct positive pairs of ui and its augmented sample u ̃i by re- placing the words having the strongest emotion arousal by their synonyms. Based on the data, we design a style encoder and pre-train it via contrastive learning. Second, to optimize the global distribution of our style representation, we further en- hance the improved Deep Embedded Clustering method [18] with contrastive learning to train our style encoder further. Through the two stages, we learn g(Ui) and denote the gener- ated representation as Context-aware Augmented Deep Embed- ded Clustering (CADEC) style. Finally, we feed it into Trans- former TTS as conditioning embedding to generate expressive audio by applying appropriate style to text. -->
框架如图：
![](image/Pasted%20image%2020240421104718.png)

1. 先构建 $u_i$ 和其增强样本 $\tilde{u}_i$ 的正样本对，即：将情感激动最强的词替换为其同义词。然后设计一个 style encoder 用对比学习进行预训练。
2. 采用对比学习来进一步训练 style encoder 以增强了 Deep Embedded Clustering 方法。最后学习 $g(U_i)$，然后将生成的 representation 称为 Context-aware Augmented Deep Embedded Clustering (CADEC) style
3. 将其作为 conditioning embedding 输入到 Transformer TTS 中，生成表达性音频

### 阶段 1：基于数据增强的对比学习
<!-- An utterance could be expressed by human in various styles. The appropriate style of utterance ui is highly related to context, its semantic content and conveyed emotion. We propose taking ui and its context together, i.e. Ui, as input and combine both content feature and emotion feature to model the best-fit style. -->
一个 utterance 可以用多种风格表达。utterance $u_i$ 的风格 与 上下文、语义内容和情感 相关。将 $u_i$ 和其上下文 $U_i$ 作为输入，结合 content 特征和 emotion 特征来建模风格。
<!-- We employ a pretrained BERT [23] as backbone to ex- tract content features, and an extra emotion lexicon [24] to ex- tract emotion features. The emotion lexicon starts from a man- ually annotated English source emotion lexicon. Combining emotion mapping, machine translation, and embedding-based lexicon expansion, the monolingual lexicons for 91 languages with more than two million entries for each language are cre- ated. The lexicon provides word-level emotion features includ- ing VAD (valance, arousal, dominance) on 1-to-9 scales and BE5 (joy, anger, sadness, fear, disgust) on 1-to-5 scales. Then, we extract our initial style embedding ri by: -->
使用预训练的 BERT 作为 backbone 来提取 content 信息，采用额外的 emotion lexicon 来提取 emotion 特征。
> lexicon 提供 word-level 的 emotion 特征，包含 VAD（valance, arousal, dominance）和 BE5（joy, anger, sadness, fear, disgust）。

提取初始的 style embedding $r_i$：
$$r_i=b(U_i)\oplus\frac1M\sum_{j=1}^Me(w_j)$$
<!-- where ⊕ denotes a concatenation operator, b(Ui) is the output
[CLS] embedding by inputting Ui into BERT, M is the total number of words in Ui and wj is j-th word in Ui while e(wj) denotes its normalized BE5 feature which is a 5-dimensional vector. -->
其中 $\oplus$ 表示拼接，$b(U_i)$ 是将 $U_i$ 输入到 BERT 中的 CLS 对应的 embedding，$M$ 是 $U_i$ 中的单词数，$w_j$ 是 $U_i$ 中的第 $j$ 个单词，$e(w_j)$ 表示其归一化的 BE5 特征，是一个 5 维向量。
<!-- Then we add a fully connected multilayer perceptron (MLP) as encoder to map the initial embedding into hidden fea- tures, which are our output style embedding: -->
然后用 MLP 作为 encoder 来映射初始 embedding 到 hidden features，得到 output style embedding：
$$h_i=MLP(r_i)$$
<!-- We propose augmenting data and using contrastive learning to pre-train the parameters of encoder. -->
采用数据增强和对比学习来预训练 encoder。
<!-- To augment ui to the utterance u ̃i that would have similar speech style, we first split ui into shorter segments not longer than a fixed length, e.g., 10 in our experiments. Then we look up the emotion lexicon to get emotion arousal for each word in a segment and select top k%, e.g., 20%, to be replaced by their WordNet synonyms [25]. Take the utterance ui in Figure 1 as an example. We split it into two segments, and select “lucky” and “happy” in the first segment and “annoying” and “crazy” in the second segment. We then replace them with their synonyms to compose u ̃i. The aim of splitting a long sentence into segments is to extract emotional words from different segments, thereby avoiding focusing on the dominant emotional words from some segment only. For example, although “fortunate” has higher arousal than “annoying” in the whole sentence of 20 words, we avoid choosing it for the whole sentence by the segment-based selection. This makes our concentrated emotional words more evenly distributed to ensure the expressiveness of the whole sen- tence. -->
为了增强 $u_i$，得到类似风格的 $\tilde{u}_i$，先将 $u_i$ 分割成短片段，然后查找 emotion lexicon，获取每个片段中每个单词的 emotion arousal，并选择 top $k\%$，如 $20\%$，用其 WordNet 同义词替换。
> 例如，将 $u_i$ 分成两个片段，选择第一个片段中的 “lucky” 和 “happy”，第二个片段中的 “annoying” 和 “crazy”，然后用其同义词组成 $\tilde{u}_i$。

将长句子分成片段的目的是从不同片段提取情感词，避免只关注某个片段的主导情感词。
<!-- As for contrastive learning, from a large training dataset D, we randomly sample a minibatch data B = {Ui}Ni=1, and generate its augmented data B ̃ = {U ̃i }Ni=1 , where U ̃i = {Ci− , u ̃i , Ci+ }. Ui and U ̃i are treated as positive pairs while the other N −1 pairs {< Ui , U ̃k >}i̸=k are all negative examples in one minibatch. To maximize the agreement between texts with similar emotions and disagreement between texts with differ- ent emotions, following simCLR [11], we calculate the sample- wise contrastive loss by -->
对于对比学习，从大型训练数据集 $\mathcal{D}$ 中，随机采样一个 minibatch 数据 $\mathcal{B}\:=\:\left\{U_i\right\}_{i=1}^N$，生成其增强数据 $\tilde{\mathcal{B}}\:=\:\{\tilde{U}_i\}_{i=1}^N$，其中 $\tilde{U}_i\:=\:\{C_i^-,\tilde{u}_i,C_i^+\}$。$U_i$ 和 $\tilde{U}_i$ 视为正样本，其他 $N-1$ 对 $\{<U_i,U_k>\}_{i\neq k}$ 都是负样本。为了最大化具有相似情感的文本之间的一致性，以及不同情感的文本之间的差异性，采用 SimCLR 中的方法，计算样本间的对比损失：
$$l_c^i=-log\frac{exp(cos(h_i,\tilde{h_i})/\tau)}{\sum_{k=1}^N\mathbb{1}_{k\neq i}exp(cos(h_i,\tilde{h_k})/\tau)}$$
<!-- Here τ is the temperature parameter and 1k̸=i is the indicator function. The contrastive loss for a minibatch is computed by averaging over all instances in B and its augmented data B ̃: -->
其中 $\tau$ 是 temperature parameter，$\mathbb{1}_{k\neq i}$ 是指示函数。minibatch 的对比损失通过对 $B$ 和其增强数据 $\tilde{B}$ 中的所有样本求平均：
$$\mathcal{L}_{contrastive}=\frac1N\sum_i^N\ell_c^i$$

### 阶段 2：基于 Autoencoder 的深度嵌入聚类
<!-- To optimize the global distribution of style representations, we apply deep embedded clustering with autoencoder to train the CADEC style encoder further. The number of clusters K is a prior and each cluster is represented by its centroid μk . Cluster- ing loss is defined as -->
采用 deep embedded clustering 和 autoencoder 来进一步训练 CADEC style encoder。聚类数 $K$ 是先验，每个 cluster 由其质心 $\mu_k$ 表示。聚类损失定义为： 