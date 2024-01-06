  
1. 本文表明，从音频中学习表征，然后进行微调，可以超过最好的半监督方法。
2. 提出 wav2vec 2.0 ，mask 潜在空间中的语音输入，然后联合学习量化潜在表征及其定义的对比任务

## Introduction

本文提出了一个从原始音频中自监督学习表征的框架。

通过多层卷积神经网络对音频进行编码，然后以类似于BERT的方法 mask 一段潜在语音表征。

潜在表征通过 Transformer 网络来生成上下文表征，并且通过对比任务来训练模型，即区分真正的潜在表征和噪声。

在训练时，使用 gumbel softmax 得到对比任务中的潜在表征，这种方法得到的潜在表征比非量化法得到的更好。

预训练完成后，使用 CTC 损失 fine tune 模型以将其用于下游的语音识别任务。

[[vq-wav2vec- Self-Supervised Learning of Discrete Speech Representations 笔记]] 解决了数据量化的问题，并使用自注意力模型学习上下文表征，本文则使用端到端的方法来实现。

结果表明，联合学习的上下文表征能够获得更好的性能，同时在低资源情况下效果特别好。

## 模型

模型由多层卷积 feature encoder $f: \mathcal{X} \mapsto \mathcal{Z}$ 组成，其输入为原始音频 $\mathcal{X}$ ，输出 $T$ 个 time step 的表征 $\mathbf{z}_1, \ldots, \mathbf{z}_T$，然后这些表征输入到 Transformer $g: \mathcal{Z} \mapsto \mathcal{C}$ 来生成捕获了整个序列的上下文表征 $\mathbf{c}_1, \ldots, \mathbf{c}_T$，feature encoder 的输出通过量化模块 $\mathcal{Z} \mapsto \mathcal{Q}$ 生成 $\mathbf{q}_t$ 以作为自监督模型的 target（或者说label）。

整个模型如图：![[image/Pasted image 20221205113309.png]]

相比于 vq-wav2vec，提出的模型在连续的语音表征上构建上下文表征，并使用自注意模块端到端地捕获整个潜在表征序列上的依赖性。

### Feature encoder

encoder 由：
+ 时间卷积（temporal convolution）
+ 层归一化
+ GELU激活函数
组成，输入到 encoder 的原始波形被归一化为零均值和单位方差。encoder 的 stride 决定了输入到 Transformer 的 time step $T$。

### Transformer 提取上下文表征

Feature encoder 的输出进到由 Transformer 组成的 context network。使用卷积层作为相对位置嵌入。卷积的输出通过 GELU 后应用层归一化。

### 量化模块

为了实现自监督训练，通过 product quantization 将 Feature encoder的输出 $\mathbf{z}$ 转化为一组有限的离散语音表征。

Product quantization 是从多个 codebooks 中选择量化表征并将它们串联起来。给定 $G$ 个 codebooks，每个 codebook 有 $V$ 个 entry，$e \in \mathbb{R}^{V \times d / G}$ ，每个 codebook 选择一个 entry 然后拼接 $e_1, \ldots, e_G$ 得到结果，最后使用线性变换 $\mathbb{R}^d \mapsto \mathbb{R}^f$ 得到最终的量化结果 $\mathbf{q} \in \mathbb{R}^f$。

使用 Gummel softmax 使得在选择离散的 codebook entries 是可微的。使用 straight-through estimator，设置 $G$ 个 Gumbel softmax 操作，使得 feature encoder 的输出 $\mathbf{z}$ 映射到 $l \in \mathbb{R}^{G \times V}$，即 logits，同时选择第 $g$ 个codebook 的第 $v$ 个 entry 的概率为：$$p_{g, v}=\frac{\exp \left(l_{g, v}+n_v\right) / \tau}{\sum_{k=1}^V \exp \left(l_{g, k}+n_k\right) / \tau}$$
其中，$\tau$ 为非负 temperature 系数，$n=-\log (-\log (u))$ 且 $u$ 为从$[0,1]$ 的均匀分布中采样得到。

在 forward 阶段， entry $i$ 通过 $i=\operatorname{argmax}_j p_{g, j}$ 选中；
在 backward 阶段，使用 Gumbel softmax 输出的真实梯度。

## 训练  

模型训练时，在feature encoder 潜在空间中 mask 了一定数量的 time step，类似于BERT。

目标函数使得，在一堆噪声样本中正确识别每个 mask 的 time step 的量化表征。

### Masking

在将 feature ecoder 的输出送到 context network 之前对其进行 mask；但在量化的时候不进行 mask。

为了实现 mask，随机采样总 time step 的某一比例 $p$ 作为起始索引，然后从每个采样索引中 mask 随后的 $M$ 个连续的 time step（可能重叠）。

### 目标函数

训练的时候，通过对比任务来学习表征，对应的损失为 $\mathcal{L}_m$ 。
同时对于量化模块，还有 codebook diversity $\mathcal{L}_d$，总损失为：$$\mathcal{L}=\mathcal{L}_m+\alpha \mathcal{L}_d$$
#### 对比损失

给定以被 mask 的 time step $t$ 为中心的 context network 对应的输出 $\mathbf{c}_t$，模型需要在 $K+1$ 个候选的 quantized representations $\tilde{\mathbf{q}} \in \mathbf{Q}_t$ （$\mathbf{Q}_t$ 包含 $\mathbf{q}_t$ 和 $K$ 个噪声样本，且噪声样本是在其他被 mask 的 time step 中随机均匀采样的）中识别 true quantized representation $\mathbf{q}_t$ ，这个过程的损失为：$$\mathcal{L}_m=-\log \frac{\exp \left(\operatorname{sim}\left(\mathbf{c}_t, \mathbf{q}_t\right) / \kappa\right)}{\sum_{\tilde{\mathbf{q}} \sim \mathbf{Q}_t} \exp \left(\operatorname{sim}\left(\mathbf{c}_t, \tilde{\mathbf{q}}\right) / \kappa\right)}$$
这里计算了 quantized representations 和 context representations 之间的 余弦相似度 $\operatorname{sim}(\mathbf{a}, \mathbf{b})=\mathbf{a}^T \mathbf{b} /\|\mathbf{a}\|\|\mathbf{b}\|$。

#### Diversity Loss

对比任务依赖 codebook 来表示正负样本，Diversity Loss $\mathcal{L}_d$ 用于increase the use of the quantized codebook。通过最大化 utterances 中每个 codebook $\bar{p}_g$ 的 entries 上的averaged softmax distribution $l$ 的熵，鼓励在 $G$ 个 codebooks 中平等使用 $V$ 个 entry；softmax分布不包含 gumbel 噪声，也不包含temperature：$$\mathcal{L}_d=\frac{1}{G V} \sum_{g=1}^G-H\left(\bar{p}_g\right)=\frac{1}{G V} \sum_{g=1}^G \sum_{v=1}^V \bar{p}_{g, v} \log \bar{p}_{g, v}$$

### fine tune

预训练的模型在语音识别任务中进行 fine tune，在顶端添加一个随机初始化的 linear projection 层，linear 层的输入维度为表征的维度，输出维度为类别 $C$，用于分类（其实就是词表大小为 $C$），对于 librispeech，采用 29 个字符作为 token（$C=29$），模型通过 CTC loss 进行优化，采用了 修改的 SpecAugment 进行增强。

### 图解

预训练过程：![[image/Pasted image 20230323205011.png]]

量化：![[image/Pasted image 20230323205101.png]]

## 实验

### 数据集

采用 960 小时的 Librispeech （LS-960）或者  LibriVox (LV-60k)，后者预处理得到 53.2k 小时的音频。在五种情况下进行 fine tune：
+ 960 hours of transcribed Librispeech
+  the train-clean-100 subset comprising 100 hours (100 hours labeled)
+ Libri-light limited resource training subsets，train-10h (10 hours labeled)
+ train-1h (1 hour labeled)
+ train-10min (10 min labeled)

### 预训练

mask 的概率为 $p=0.065$，mask 的长度为 $M=10$，大约 $49\%$ 的 time step会被 mask。

feature encoder 有 7 个 block，channel 都是 512，stride 为 $( (5,2,2,2,2,2,2))$，kernel 为 $(10,3,3,3,3,2,2)$，最后大概 20 ms 得到一个 time step，感受野为 400 个样本点，25 ms（有点类似于谱特征提取的 win size 和 shift）。

base 模型包含 12 Transformer Encoder，维度 768，FFN 的维度为 3072，num head 为 8，通过裁剪 250k 个样本点（15.625s）得到一条音频的长度，每个 gpu 不超过 1.54m 个样本点，在 64 台 V100 上训练 1.6 天，总的 batch 是1.6 小时（6分钟一台gpu？）。

large 模型包含 24 层的 Transformer Encoder，维度 1024，FFN 维度 4096，num head 为 16，裁剪 320k 个样本点（20s）得到一条音频的长度，对于 LibriSpeech，在 128 V100 GPUs 训练 2.3 天，总的 batch size 是 2.7 小时，Transformer 的 dropout 是 0.1。

采用 Adam 优化器，前 8% 的step 进行 warm up 到 $5\times 10^{-4}$（base），large 是到 $3\times 10^{-4}$，然后进行线性衰减。

large 训练了 250K step，base 训练 400K。量化模块中，$G=2,V=320$。

gumbel softmax temperature $\tau$ ，对于 base，从 2 衰减到 0.5，对于 large 从 2 衰减到 0.1，衰减因子 0.999995。

对比损失中的 $\kappa=0.1$，对于小的 Librispeech 数据集，通过对 feature encoder 的最后一层的激活 采用 L2惩罚 来进行正则化，并将编码器的梯度缩小10倍。还使用稍微不同的 encoder 架构，即不使用 layer norm，也没有对原始波形进行归一化，而是对编码器层的第一层的输出进行归一化。对比损失中，$K=100$，在 valid 时选择训练损失 $\mathcal{L}_m$ 最低的那个 checkpoint。

### fine tune（略）

