> Google，NIPS，2017

1. 本文提出 VQ（vector quantization）-VAE，和 VAE 的区别在于：
	1. encoder 的输出离散而非连续的 code
	2. 先验分布是可以学习的而非一开始就确定的
2. 结合 矢量量化的思想来学习离散表征，同时也可以避免模型的 “posterior collapse” 问题
3. 将 representation 和 autoregressive prior 配对，模型可以生成高质量的表征

> 关于 VAE 的 “**Posterior Collapse**” 问题见：[https://zhuanlan.zhihu.com/p/389295612](https://zhuanlan.zhihu.com/p/389295612)
## Introduction

1. 最大似然和重构误差是在像素域中训练无监督模型的两个常见目标。但是其有效性取决于特征的应用场景
2. 本文的目标是，实现模型使其在潜在空间中保留数据的重要特征。同时优化最大似然
3. 先前工作的重点是学习连续特征，本文则专注于离散表征学习。因为语言本质上是离散的，语音通常表示为符号序列。同时离散表征法适合复杂的推理、规划和预测学习
4. 本文提出的 VQ-VAE，训练方便、没有 large variance，同时避免VAE中的 “posterior collapse” 问题。
5. VQ-VAE 可以有效利用潜在空间，可以对数据空间中跨纬度特征进行建模（如跨图像中的像素、语音中的音素、文本中的消息等），而非专注于建模局部噪声或细节。

## 相关工作（略）


## VQ-VAE

VAE 由 encoder（以 后验分布 $q(z \mid x)$ 建模）、先验分布 $p(z)$ 和一个以分布 $p(x \mid z)$ 建模输入数据 decoder ，通常，VAE 中的先验和后验分布假定为对角方差的正态分布。

本文引入 VQ-VAE，使用离散的潜在变量，采用了一种新的训练方式，灵感来源于 VQ，先验和后验分布是 Categorical分布，the samples drawn from these distributions index an embedding table。这些 embedding 用作 decoder 网络的输入。

### 离散潜在变量

定义 latent embedding space $e \in R^{K \times D}$（所谓的 codebook），$K$ 为 discrete latent space 的大小（即 $K$ 为 categorical分布的大小），$D$ 为 每个 latent embedding vector $e_i$ 的维度。一共有 $K$ 个 embedding vectors $e_i \in R^D, i \in 1,2, \ldots, K$，如图：![[image/Pasted image 20221201202930.png]]
模型输入为 $x$，通过encoder之后输出 $z_e(x)$，discrete latent variables  $z$ 通过最近邻 look-up 来计算。

posterior categorical distribution 分布 $q(z \mid x)$ 定义为 ：$$q(z=k \mid x)= \begin{cases}1 & \text { for } \mathrm{k}=\operatorname{argmin}_j\left\|z_e(x)-e_j\right\|_2 \\ 0 & \text { otherwise }\end{cases}$$
> 其实就是 one-hot 向量，而且是 deterministic 的。
> 对于 VAE，loss 为：$$L_{VAE}(\phi, \theta) = -\mathbin{E}_{z\sim q_{\phi}(z|x)}[\text{log }p_\theta(x|z)] + D_{KL}(q_\phi(z|x)||p_\theta(z))$$
> 其中的 $q(z \mid x)$ 不是确定的（是一个分布），而 VQ-VAE 因为是 one-hot vector，所以说是 deterministic 的。

  将此模型视为一个VAE，其中可以将 $\log p(x)$ 与ELBO绑定。由于前面分析的，分布 $q(z=k \mid x)$ 是确定的，同时作者在论文中假设 $z$ 的先验分布是均匀分布，结果计算得到 KL 散度等于常数 $\log K$，在计算损失和优化函数的时候就不用管了。

decoder 的输入是下式中对应的 embedding vector $e_k$，$$z_q(x)=e_k, \quad \text { where } \quad k=\operatorname{argmin}_j\left\|z_e(x)-e_j\right\|_2$$
表征 $z_e(x)$ 通过 discretisation bottleneck，然后映射到 上式中最邻近的 embedding $e$ 中。


### 学习过程

最近邻映射没有真正的梯度，但是类似 straight-through estimator 来近似梯度。直接把 decoder 的输入 $z_q(x)$ 的梯度复制到encoder 的输出 $z_e(x)$ 中。

在 forward 计算时，最近邻的 embedding $z_q(x)$ 被送到 decoder中，在 backward 中，梯度直接从decoder传到 encoder 而保持不变。这个过程中，梯度包含 encoder 是如何改变其输出来降低重构损失 的有用的信息。

VQ-VAE 的总损失为：$$L=\log p\left(x \mid z_q(x)\right)+\left\|\operatorname{sg}\left[z_e(x)\right]-e\right\|_2^2+\beta\left\|z_e(x)-\operatorname{sg}[e]\right\|_2^2$$
包含三个部分：
+ 第一项是 重构损失，由于是 straight-through ，embedding $e_i$ 不会从离散映射中接受到任何 loss 和 梯度，也就是不会进行更新（encoder 的输入是量化后的 embedding，但是计算 loss 的时候是直接计算输入输出之间的差异，这个过程中更新的是 encoder 和 decoder 的参数，目测代码实现起来比较困难？？）
+ 第二项则解决上述问题，使用  Vector Quantisation 算法，采用 $l_2$ 损失使得 embedding $e_i$ 尽可能地靠近 encoder 输出 $z_e(x)$（$\operatorname{sg}$ 算子的作用就是，让这一部分的模型的参数不更新）
+ 最后，为了确保 encoder 和 embedding 相关，且输出不会增长和频繁波动，添加 commitment loss 也就是第三项

其中，$sg$ 表示 stop gradient 操作，也就是训练的时候不更新里面的参数。  

decoder 仅通过损失的第一项优化，encoder 通过第一和第三项，embedding 通过第二项进行优化。

完整模型的对数似然计算为，$$\log p(x)=\log \sum_k p\left(x \mid z_k\right) p\left(z_k\right)$$

### 先验

如前文所述，在训练 VQ-VAE 的时候，先验为均匀分布。训练完成后，在 $z$ 上拟合一个 auto regressive 分布，所以可以通过 ancestral sampling 来生成 $x$。

对于图像，采用 PixelCNN，对于音频，采用 WaveNet，auto regressive 的模型训练的时候是和 VQ-VAE 联合训练的。

> 这个怎么理解呢，就比如说，现在已经训练好了 encoder 、decoder 和 cookbook，但是我们的根本的目的还是做生成，以图像生成为例，就是首先得到一个 $N\times N$ 的 矩阵，然后输入到 decoder 中就可以得到生成的图形，这个矩阵中的每个元素是一个值 $i, i=1,2,\dots,K$ 代表对应的像素在 codebook 中的索引，而生成这个矩阵就是用 auto regressive 的方法，也就是给定一个 initial token，采用 auto regressive 的方法直到生成完整的 $N\times N$ 的矩阵。具体到图像中，每个像素的生成是采用 PixelCNN 的方法。


