> TPAMI 2023，audEERING GmbH、University of Augsburg

1. 现有的 SER 方法没有考虑模型大小和预训练数据的影响，也没有关注 generalisation、robustness、fairness 和 efficiency
2. 本文在 MSP-Podcast 上 fine-tuned wav2vec 2.0 和 HuBERT，使用 IEMOCAP 和 MOSI 测试跨数据集的泛化性
3. 在没有显式语言信息的情况下，在 MSP-Podcast 上获得了最好的 valence 预测性能，CCC 为 0.638

## Introduction
1. SER 分为两种范式：discrete emotions 和 emotional dimensions
    1. discrete emotions：如 happy 或 sad
    2. emotional dimensions：arousal、valence 和 dominance
2. SER 通过 linguistic stream 或 paralinguistic stream 实现
    1. linguistic stream 更适合 valence recognition
    2. paralinguistics 更适合 arousal 和 dominance
    3. 两种范式可以结合在 bimodal architectures 中
    4. 本文的目标是在部署时只隐式使用 linguistic 信息流，不需要 ASR 和 NLP 前端
3. SER 的三个难点：
    1. 提高 valence 性能
    2. 解决 generalisation 和 robustness 问题
    3. 缓解个体和群体公平性问题
4. 本文评估 wav2vec 2.0 和 HuBERT 在 valence 上的效果，分析模型架构、预训练数据、泛化性、鲁棒性、公平性和效率

## 相关工作和方法

本文关注 emotional dimensions 而非 emotional categories
下表为最近在 IEMOCAP 数据集上基于 wav2vec 2.0 和 HuBERT 的方法：
![](image/Pasted%20image%2020241013105507.png)

结果按 anger、happiness、sadness 和 neutral 的 UAR/WAR 排序。大部分的方法使用 leave-one-session-out 交叉验证（5 fold），一些结论如下：
    1. fine-tuning 预训练权重提升 10%
    2. 额外的 ASR fine-tuning 对 SER 没有帮助
    3. 大模型通常优于基础模型
    4. HuBERT 优于 wav2vec 2.0
    5. fine-tuning transformer 层时，Wang et al. 的方法表现最好

部分研究在 arousal、dominance 和 valence 上 fine-tuned wav2vec 2.0 / HuBERT，结果表明预训练模型在 valence 预测上表现良好。但是用的是多模态的方法。

wav2vec 2.0 和 HuBERT 在 emotion recognition 上有很大潜力，本文对不同条件下的预训练模型进行了系统比较，并在多个数据集上进行了评估。

此外，本文还研究了 transformer-based 模型对噪声的鲁棒性。

公平性是机器学习模型的重要话题，本文也研究了 SER 模型的公平性。

## 实验设置

### 预训练模型

本文使用两种 transformer-based 模型：wav2vec 2.0 和 HuBERT。两种模型的网络架构相同，输入为归一化的原始波形，经过 7 层卷积层提取特征向量，维度为 512，时间步长为 20 ms。特征被投影到 768 或 1024 维，然后输入到 encoder 中。encoder 由 transformer layers 组成，每个包含 multi-head self-attention 模块和几个全连接层。为了注入时间信息，卷积层的输出被加到 encoder 的输入中。

两种模型的唯一区别在于预训练方式：
 + wav2vec 2.0 对一定比例的时间步进行 mask，然后最小化 encoder 的输出和 量化后的输入 之间的对比损失。
+ HuBERT 对 mask 的时间步最小化交叉熵损失，其中目标不与模型同时训练。预训练时，先用 MFCCs 的 k-means 聚类作为目标，后续步骤使用 transformer 层的输出的聚类。

wav2vec 2.0 和 HuBERT 有两种模型：
+ base 架构：12 个 transformer 层，每层 768 个 hidden units（95M 参数）
+ large 架构：24 个 transformer 层，每层 1024 个 hidden units（317M 参数）

通过给模型不同的预训练数据，得到以下模型：
+ wav2vec2-base、hubert-base-ls960、wav2vec2-large、hubert-large-ll60k
+ wav2vec2-large-robust：在电话语音上额外训练
+ wav2vec2-large-100k-voxpopuli：只在多种语言的议会演讲上训练
+ wav2vec2-xls-r-300m：在所有领域和多种语言上训练

### 结构

结构如图：
![](image/Pasted%20image%2020241013111942.png)

模型对最后一个 transformer 层的 hidden states 进行 average pooling，然后通过一个 hidden layer 和一个输出层。微调时使用 ADAM 优化器和 CCC 损失函数，学习率为  $1e^{-4}$，batch size 为 32，训练 5 个 epoch，保留在 development set 上表现最好的 checkpoint。

训练时冻结 CNN 层，微调 transformer 层。使用单个随机种子训练模型。

将结果与 14 层 Convolutional Neural Network (CNN14) 进行比较，CNN14 使用 log-Mel spectrograms 作为输入，有 6 个卷积块，每个块有两层卷积层，每层卷积层后接 max pooling。卷积层的 kernel 为 3 × 3，stride 为 1 × 1，max pooling 层的 stride 为 2 × 2。最后一个卷积层后，特征使用 mean 和 max pooling 池化，然后输入两个线性层。每个卷积块后应用 dropout，概率为 0.2。Log-Mel spectrograms 有 64 个 Mel bins，窗口大小为 32 ms，hop size 为 10 ms。CNN14 模型没有预训练，总是从头开始训练。训练 60 个 epoch，学习率为 0.01，batch size 为 64，使用 SGD 训练，Nesterov momentum 为 0.9。

### 数据集

使用 MSP-Podcast 数据集进行多任务训练，包含 arousal、dominance 和 valence 三个维度的标签，范围为 1 到 7，映射到 0 到 1。训练集包含 62 小时的录音，测试集包含 21 小时的音频，由 12,902 个样本（54% 女性 / 46% 男性）提供，60 个说话者（30 个女性 / 30 个男性）。每个说话者的样本数量在 42 到 912 之间。

在 IEMOCAP 数据集上进行跨数据集的实验，包含 12 小时的 scripted 和 improvised 对话，10 个说话者（5 个女性 / 5 个男性）。提供与 MSP-Podcast 相同的标签，范围为 1 到 5，映射到 0 到 1。由于只在评估时使用数据集，没有应用 speaker cross-validation，将整个数据集作为一个整体。包含 10,039 个样本（49% 女性 / 51% 男性）。

最后，在 Multimodal Opinion Sentiment Intensity (MOSI) 数据集上进行跨数据集的实验，包含 41 个女性和 48 个男性说话者的 YouTube 电影评论视频。标注了情感，范围为 -3 到 3，映射到 0 到 1。测试集包含 1 小时的音频，685 个样本（51% 女性 / 49% 男性），标注了情感。

情感和 valence 是不同的概念，情感对应于对特定对象的态度，而 valence 更一般地描述了一个人的感受。但是有证据表明，情感标注可以分解为两个部分：intensity 和 polarity，大致对应于 arousal 和 valence。因此，我们期望 valence 和 sentiment 之间有一定的相关性。

## 评估（略）
