 1. 语音自监督有三个挑战：
	1. 每个输入语音中有多个声音单元
	2. 在训练的时候，没有有关声音单元的 lexicon 信息
	3. 声音单元的长度可变，无法进行确切的分割
2. 于斯提出用 HuBERT 用于语音的自监督学习，采用离线的聚类方法来提供对齐的标签
3. 关键是仅在 mask 区域计算预测损失，迫使模型在连续输入上学习声学和语言的组合模型，模型主要依赖无监督模型聚类得到的一致性，而不是聚类标签本身

## Introduction

1. 自监督语音表征学习的代理任务包括：区分邻近特征和上下文远处特征、音频特征预测、掩码预测等
2. 很多语言很少标签甚至没有标签（如一些方言）
3. 伪标签从有监督的数据开始，在特定的任务中训练教师模型，然后使用教师模型来为未标记的数据生成伪标签，然后使用这些数据来训练学生模型
4. 自监督的优势在于，不受教师模型的约束，直接学习特征的潜在表征；同时相比于伪标签的特征任务，自监督的泛化性更强
5. 本文引入 Hidden unit BERT，通过聚类生成带噪声的标签，然后用于 BERT-like 的模型进行预训练。BuBERT 模型从连续输入的语音中学习声学和语言模型。模型需要将没有被 mask 的输入建模为潜在表征（声学建模），同时为了减少预测误差，模型需要捕捉学习的表征之间的长时关系（灵感来自于 DeepCluster）
6. HuBERT 在Libri 的数据集上训练时，所有的数据集上的效果都优于 wav2vec2.0

## 方法
![[image/Pasted image 20230323182536.png]]
### 学习 hidden unit

简单的离散隐变量模型如 K-mean 或者 GMM 可以推断出 hidden unit，且这些 unit 和潜在的声学单元之间存在某种相关性，图模型或者神经网络可以更好地发现声学单元或者数据的分布。

基于此，提出 acoustic unit discovery models 来提取 frame level 的目标。设一段语音序列 $X=\left[x_1, \cdots, x_T\right]$，一共有 $T$ 帧，得到的 hidden unit 记为 $h(X)=Z=\left[z_1, \cdots, z_T\right]$，其中 $z_t \in[C]$ 是一个 $C$ 类的 categorical variable，$h$ 代表某种聚类模型如 k-means 聚类。
> 这个过程（对应上图顶部的红色模块）首先是对原始音频提取语音特征，如 MFCC 特征，然后对这些特征进行聚类，最后得到的 time step 要和后面 CNN encoder 得到的 time step 一样（都是 $T$）
> 然后聚类成 $C$ 类，聚类得到的结果经过 code embedding 层之后，就是后面提到的 codeword $e_c$，下标 $c$ 代表第 $c$ 个类别的聚类中心。

### 通过掩码预测进行表征学习

令 $M \subset[T]$ 表示要 mask 的 indices $\tilde{X}=r(X, M)$ 表示被 mask 之后的输入，其中 $x_t$ 被 $\tilde{x}$ 取代（当 $t\in M$）。掩码预测模型 $f$ 输入为 $\tilde{X}$，然后在每个 time step 预测 target 的分布。

mask 过程涉及两个问题，如何 mask 和 在哪计算 预测损失。

采用了 wav2vec 2.0 的方法进行 mask 生成，随机选择 $p\%$ 的 time step 作为初始 indices，mask 的长度为 $l$ step。 然后分开计算 mask 部分和没有 mask 部分的损失，mask 部分的损失 $L_m$ 为：$$L_m(f ; X, M, Z)=\sum_{t \in M} \log p_f\left(z_t \mid \tilde{X}, t\right)$$
没被 mask 部分的损失 $L_u$ 和上面的差不多，出了求和的范围是 $t \notin M$ ，最终的计算为两个的加权：$$L=\alpha L_m+(1-\alpha) L_u$$
极端情况下，$\alpha=0$，只计算没被 mask 部分的损失（类似于声学模型）。而 $\alpha=1$ 时只计算被 mask 部分的损失，这时候类似于语言模型。

### 聚类融合

将多个聚类方法组合起来以提高性能。例如，不同 codebook size 的 k-mean 算法组合可以获得不同粒度的目标。

令 $Z^{(k)}$ 为第 $k$ 个聚类模型生成的目标序列，$L_m$ 可以重新写成：$$L_m\left(f ; X,\left\{Z^{(k)}\right\}_k, M\right)=\sum_{t \in M} \sum_k \log p_f^{(k)}\left(z_t^{(k)} \mid \tilde{X}, t\right)$$
$L_u$ 的计算过程差不多。有点类似于多任务学习，但是任务是通过无监督的聚类创建的。

### refine

一种改进表征的方法是 refine cluster assignments，通过在学习到的潜在表征中创建新一代的聚类，然后获得新的 unit。
> 就是把 Transformer encoder 中的某一层的输出拿出来当成是 $X$，然后通过前面的 acoustic unit discovery models 得到新的聚类中心，作为新的 hidden unit。

### 实现

模型和 wav2vec2.0 差不多，首先是卷积特征编码，然后接 BERT 编码器+投影层+code embedding 层。

考虑了三种不同的配置：base，large 和 x-large，模型具体配置如下：![[image/Pasted image 20230323202045.png]]

CNN encoder 从 16KHz  的音频中生成 20ms 帧率的音频（也就是每秒得到 50 帧），然后根据前面的 mask 策略进行随机的 mask，然后输入到 BERT 中产生特征序列 $\left[o_1, \cdots, o_T\right]$ ，然后把问题看成一个分类问题，通过 softmax 得到：$$p_f^{(k)}(c \mid \tilde{X}, t)=\frac{\exp \left(\operatorname{sim}\left(A^{(k)} o_t, e_c\right) / \tau\right)}{\sum_{c^{\prime}=1}^C \exp \left(\operatorname{sim}\left(A^{(k)} o_t, e_{c^{\prime}}\right) / \tau\right)}$$
其中，$A$ 为投影矩阵，$e_c$ 为 codeword $c$ 的 embedding，$\text{sim}(\cdot,\cdot)$ 计算向量之间的余弦距离，$\tau$ 为缩放因子（temperature 系数，设置为 $0.1$），如果采用组合聚类的方法，每个聚类的投影矩阵 都不一样。
> 目的就是把 $e_c$ 当作 target，然后就和普通的 BERT 做 mask 预测差不多。
> 训练的时候就用最简单的交叉熵损失（因为本质是一个 $C$ 分类的问题），相比于 wav2vec2.0 的对比损失+多样性损失，简单了很多！

HuBERT 预训练完成之后，采用 CTC 损失在 ASR 任务中进行 fine tune（CNN 层不做fine tune，会 freeze），去掉投影层然后加上一个随机初始化的 softmax 层。

### 图解

预训练过程：
![[image/Pasted image 20230323205222.png]]
> 注：图中有些变量和论文中的命名不一致。


refine 聚类：![[image/Pasted image 20230323213137.png]]


## 实验细节

TODO