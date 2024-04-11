> ICASSP 2022，Korea University

1. 现有的 TTS 在情感语音合成上的性能受限
2. 大多数方法使用情感标签或参考音频提取情感信息，但由于基于整个句子的情感条件，导致情感表达单调
3. 提出 EmoQ-TTS，把 phoneme-wise emotion 信息进行 fine-grained 的强度控制，来合成情感语音
    1. emotion 信息的强度通过 distance-based intensity quantization 实现，无需人工标注
    2. 也可以通过手动设置 intensity labels 来控制合成语音的情感

> 采用 GRL 来学习解耦的 emotion 信息，通过量化来控制不同的 emotion 的强度。

## Introduction

1. emotion 信息受到 pitch、tone 和 tempo 等 paralinguistic 特征的影响，情感语音合成很有挑战性
2. 传统方法使用全局的情感信息，导致合成语音表现单调，fine-grained 的情感表达应该在 phoneme-level 上：
    1. 一些研究尝试通过缩放或插值情感 embedding 来反映 fine-grained 的情感，但音质不稳定，参数难以确定
    2. 也有模型预测从学习的 ranking function 中提取的 phoneme-wise intensity scalar，但是很难稳定控制情感表达
3. 提出 EmoQ-TTS，通过 phoneme-wise emotion 信息进行 fine-grained 的情感强度控制，无需人工标注：
    1. 使用 intensity pseudo-labels 和 distance-based intensity quantization 来反映情感
    2. 可以从文本预测情感强度
    3. 可以手动设置 intensity labels 来控制情感表达

## EMOQ-TTS

### 模型架构

EmoQ-TTS 如图：
![](image/Pasted%20image%2020240409105927.png)

基于 FastSpeech2，包括 encoder、decoder 和 variance adaptor，但是做了两点修改：
+ 引入 emotion renderer，根据 fine-grained 的情感强度提供 phoneme-level 的 emotion，使得所有 variance 信息都可以受到 fine-grained 的情感强度影响
+ 将 duration predictor 移动到 variance adaptor 的之后，使所有 variance 信息在 phoneme-level 处理，比 frame-level 方法性能更好

### Emotion Renderer

emotion renderer 根据 fine-grained emotion intensity 提供 phoneme-level 的 emotion，如上图 b，emotion renderer 包括：
+ intensity predictor
+ intensity quantizer
+ intensity embedding table

给定 phoneme hidden sequence $H_{\text{pho}}$ 和第 $k$ 个 emotion category $\text{emotion}_k$ ，intensity predictor 预测 $\text{emotion}_k$ 的 phoneme-wise emotion intensity 标量序列，其值在 0 到 1 之间。intensity predictor 通过 mean absolute error (MAE) 损失进行优化，最小化预测的 intensity 和 ground-truth intensity 之间的差异。

intensity scalar 通过 emotion intensity quantizer 量化为 $N_I$ 个 emotion intensity pseudo-labels。$N_I$ 是量化的 intensity pseudo-labels 的总数。此外，引入一个 intensity embedding table。量化后的 intensity pseudo-labels 是每个 emotion 的 embedding table 的 entry index。最后，phoneme-wise intensity embedding sequence 与 phoneme hidden sequence 拼接（图中的 C）。

推理时，EmoQ-TTS 通过从预测的 intensity scalar（其以量化的 intensity embedding 为条件） 来合成  emotional speech。也可以通过手动设置 intensity labels 控制合成语音的情感表达。

## Emotion Intensity Modeling

采用 distance-based intensity quantization 来设计 emotion intensity 信息。如图 c，emotion intensity modeling 有两个阶段：
+ emotion 特征提取
+ emotion intensity 量化

### Emotion Feature Extraction

第一阶段，训练 reference encoder 从 mel-spectrogram 中提取聚类后的 emotion embedding。为了提取区分性的 emotion embedding，使用 emotion classifier 和 phoneme classifier，其中包含梯度反转层(GRL)。这两个 classifier 使得特征向量按照 emotion 聚类，而不影响 phoneme 信息。两个 classifier 都通过 softmax 层和 cross-entropy loss 进行优化。对于 phoneme classifier，在反向传播过程中通过梯度反转层乘以一个负标量来反转梯度。
> GRL 的存在让模型在训练时，可以让模型同时学习到 phoneme 无关的特征和 emotion 相关的特征。
> 正是因为 GRL 的存在，才可以学习到区分性的 emotion embedding。

然后，token-wise pooling 层基于音素边界进行平均，将 frame-level 序列转换为 phoneme-level 序列。添加两个辅助预测器，分别预测 pitch 和 energy。

emotion 特征提取模块与 TTS 模块一起训练。训练过程中，提取的 phoneme-wise emotion embedding 序列 $H_E$ 直接送到 emotion renderer，再与 phoneme hidden sequence 拼接。

### Distance-based Emotion Intensity Quantization

第二阶段，通过 intensity quantization 生成 emotion intensity pseudo-labels 和 intensity embedding table。如图 d，将第 $k$ 个 emotion embedding $E_{kj}$ 和 neutral embedding $E_{nj}$ 输入到 emotion intensity extractor，其中 $j \in \{1,2,...,N_k\}$，$N_k$ 是 $E_{kj}$ 的总数。
> 这里用了从 reference encoder 提取的每个 emotion 的所有的 emotion embedding。

为了提取合适的 intensity，引入 emotion distance，用于表示相对于 neutral emotion 的中心的相对距离，做两个假设：
+ neutral emotion 是所有 emotion 中最弱的
+ emotion intensity 随着离 neutral emotion 越远而增加

由于多维空间太不稳定，emotion intensity extractor 将 emotion embedding 投影到一个单一向量。选择 linear discriminant analysis (LDA) 作为最优投影向量 $w^*$，通过最大化二分类 LDA 的目标函数得到：
$$\mathcal{L}_{LDA}(w)=\frac{(m_k-m_n)^2}{s_k^2+s_n^2}=\frac{w^TS_Bw}{w^TS_Ww}.$$
其中，$m_k$ 和 $m_n$ 分别是 $E_{kj}$ 和 $E_{nj}$ 的均值，$s_k^2$ 和 $s_n^2$ 是 $E_{kj}$ 和 $E_{nj}$ 的方差，$w$ 是投影向量，$S_B$ 和 $S_W$ 是两个 cluster 的 between-class variance 和 within-class variance。得到 $w^*$ 后，将 $E_{kj}$ 和 $E_{nj}$ 投影到最优投影向量 $w^*$，得到投影的 emotion embeddings $E_{kj}^\prime$ 和 $E_{nj}^\prime$：
$$E_{kj}^{'}=\frac{E_{kj}\cdot w^*}{||w^*||_2^2}w^*$$
然后，通过投影的 neutral embeddings $E_{nj}^\prime$ 的中心和投影的 emotion embeddings $E_{kj}^\prime$ 之间的欧氏距离来衡量 emotion distance $d_{kj}$：
$$d_{kj}=||E_{kj}^{'}-M_n^{'}||_2\quad\mathrm{where}\quad M_n^{'}=\frac1{N_n}\sum_{j=1}^{N_n}E_{nj}^{\prime}$$

$M_n^\prime$ 表示投影的 neutral embedding 的中心。提取 emotion distance 后，使用 interquartile range technique 去除每个 emotion 的异常值。通过 min-max normalization 将 emotion distances 归一化为 0-1 之间的 intensity scalar。然后再量化为 $N_I$ 个 emotion intensity pseudo-labels。
> 类似于语音信号处理中对波形的 N 阶量化。

引入 intensity embedding table 来表征 intensity pseudo-labels。在 intensity embedding table 中，第 $l$ 个 intensity 的 embedding 由 reference encoder 对应的 intensity pseudo-label 的 emotion embedding 的平均值计算得到：
$$I_{kl}=\frac{1}{N_{kl}}\sum_{i_{kj}\in C_{kl}}E_{kj}$$

其中 $I_{kl}$ 表示 intensity embedding table 中 $\text{emotion}_k$ 的第 $l$ 个 intensity embedding，$i_{kj}$ 表示 $E_{kj}$ 的 intensity pseudo-label，$C_{kl}$ 是对应于第 $l$ 个 intensity 的 $i_{kj}$ 的组，$N_{kl}$ 表示 $i_{kj} \in C_{kl}$ 的 emotion embedding 的数量。intensity labels 是每个 emotion 的 intensity embedding table 的 entry index。这个 embedding table 用于 EmoQ-TTS 的 emotion renderer。
> 相当于 对于每个 emotion 的每个 intensity，都有一个与之对应的 embedding。

## 实验和结果（略）

数据集：
+ 单说话人：Korean Emotional Speech (KES)，包含 7 种情感（neutral, happy, sad, angry, surprised, fearful, and disgusted）的 21,000 段语音
+ 多说话人：EmotionTTS Open DB (ETOD)，包含 4 种情感（neutral, happy, sad,and angry）的 6,000 段语音