
1. 提出 vq-wav2vec 学习语音段的离散表征
2. 使用Gumbel Softmax或online k mean 聚类来量化表征

## Introduction

1. 学习离散单元的一种流行的方法是通过自编码器，有时也结合自回归模型
2. 本文通过上下文预测任务学习语音的离散表征，而非重建输入
3. 提出 vq-wav2vec 通过利用 wav2vec 损失和架构 学习音频信号的离散表征，类似于 VQ-VAE，为了选择离散表征，考虑使用 Gumbel-Softmax 和 online k-means 方法。然后在离散化的表征上训练 BERT，最后输入到声学模型中，结构如图：![[image/Pasted image 20221214111156.png]]
## 背景

### Wav2Vec

原理见 [[wav2vec- Unsupervised pre-training for speech recognition 笔记]]，其中对比损失定义为：$$\mathcal{L}_k^{\text {wav2vec }}=-\sum_{i=1}^{T-k}\left(\log \sigma\left(\mathbf{z}_{i+k}^{\top} h_k\left(\mathbf{c}_i\right)\right)+\underset{\tilde{\mathbf{z}} \sim p_n}{\mathbb{E}}\left[\log \sigma\left(-\tilde{\mathbf{z}}^{\top} h_k\left(\mathbf{c}_i\right)\right)\right]\right)$$
$T$ 为序列长度，$\sigma\left(\mathbf{z}_{i+k}^{\top} h_k\left(\mathbf{c}_i\right)\right)$ 表示样本 $\mathbf{z}_{i+k}$ 为真实标签的概率。最终的损失是所有的 step 的求和。

### BERT

原理见 [[BERT- Pre-training of Deep Bidirectional Transformers for Language Understanding 笔记]]。

BERT模型结合了两个任务进行训练：首先，masked language modeling 随机移除一些输入token，模型必须预测这些 token。其次，next sentence prediction 将两个不同的文本段落拼接成一个，模型需要预测这些段落是否来自同一文本。

### vq-wav2vec

vq-wav2vec 使用 未来时间步长预测任务学习音频的矢量量化（vq）表征。遵循与wav2vec 相同的架构选择，使用两个卷积网络 $f: \mathcal{X} \mapsto \mathcal{Z}$ 和 $g: \hat{\mathcal{Z}} \mapsto \mathcal{C}$ 进行特征提取和聚合，以及一个新的量化模块 $q: \mathcal{Z} \mapsto \hat{\mathcal{Z}}$ 来生成离散表征。

量化模块将大小固定的 codebook $\mathbf{e} \in \mathbb{R}^{V \times d}$ （包含 $V$ 个维度为 $d$ 的表征）中的表征 $\hat{\mathbf{z}}=\mathbf{e}_i$ 替换原始的表征 $\mathbf{z}$ ，Gumbel-Softmax 是 argmax 的可微近似，用来计算 one hot 表征，同时也使用了 online k-means 聚类，如图：![[image/Pasted image 20221214112901.png]]最后，在 $\mathbf{z}$ 的不同部分执行多个矢量量化，以减轻模式崩塌问题。

#### GUMBEL-SOFTMAX

Gumbel-Softmax 使得选择 codebook entry 这个过程可微，同时也使用了 straight-through estimator。给定连续的表征 $\mathbf{z}$ ，依次通过 linear、ReLU、linear 层得到 $l \in \mathbb{R}^V$ ，也就是所谓的 logits，在推理阶段，选择 logits 最大的那个 index。

在训练时，选择第 $j$ 个变量的概率为：$$p_j=\frac{\exp \left(l_j+v_j\right) / \tau}{\sum_{k=1}^V \exp \left(l_k+v_k\right) / \tau}$$
其中，$v=-\log (-\log (u))$ 且 $u$ 为均匀分布。

#### k-means

在从 codebook 选择 entry 时，选择距离最近的那个，其索引计算为：$$i=\operatorname{argmin}_j\left\|\mathbf{z}-\mathbf{e}_j\right\|_2^2$$
在 forward 过程中，选择 $\hat{\mathbf{z}}=\mathbf{e}_i$ ，则 encoder 的梯度是通过反向传播 $\mathrm{d} \mathcal{L}^{\text {wav2vec }} / \mathrm{d} \hat{\mathbf{z}}$ 计算得到的，最终的损失包含了两个额外的项：$$\mathcal{L}=\sum_{k=1}^K \mathcal{L}_k^{\text {wav2vec }}+\left(\|\operatorname{sg}(\mathbf{z})-\hat{\mathbf{z}}\|^2+\gamma\|\mathbf{z}-\operatorname{sg}(\hat{\mathbf{z}})\|^2\right)$$
其中，$sg$ 为梯度停止算子。第一项为 future prediction 任务，BP 时不改变 codebook 中 entry 的值（因为是 straight-through gradient estimation）。第二项使得 codebook vectors 尽可能地接近 encoder 的输出，第三项使得 encoder 的输出尽可能地接近中心。

#### VECTOR QUANTIZATION WITH MULTIPLE VARIABLE GROUPS

目前只使用 codebook 中单个 entry 作为离散特征，但是这很容易发生 模式崩塌 现象（只有一部分 entry 被使用，从而输出表征的多样性较弱），本文提出独立 量化 $\mathbf{z}$ 的一部分（类似于乘积量化）以提高性能。

首先将 $\mathbf{z} \in \mathbb{R}^d$ 分成多个 group $\mathbf{z}^{\prime} \in \mathbb{R}^{G \times(d / G)}$ ，然后用 整数索引 表示每一行，因此索引向量 $\mathbf{i} \in[V]^G$ 就可以表示一个完整的离散特征 ，对于每个组，都使用上面提到的两种方法进行量化。

## 在量化后的表征使用 BERT 

采用 BERT 进行预训练，任务是基于上下文的编码来预测 masked input token。

训练完成后，输入到声学模型中来改进语音识别等任务。

单个 token 预测太简单，实际 mask 连续语音（$M=10$ 个 连续 token）的 token，相比于预测单个 token 可以提升性能。

## 实验及结果（略）