> Hinton 提出的

# Dynamic Routing Between Capsules
capsule 是一组神经元，其激活向量表示特定类型的实体的实例化参数，对于一个向量，最重要的就是两个点：
+ 向量的长度，用来表示实体存在的概率
+ 向量的方向：用来表示实例化参数

一层的 capsule 通过转换矩阵对高层的 capsule 的实例化 参数进行预测，当多个预测一致时，高层的 capsule 将被激活。

本文实现了一种协议迭代路由机制，A lower-level capsule prefers to send its output to higher level capsules whose activity vectors have a big scalar product with the prediction coming from the lower-level capsule.

## Introduction

使用向量的长度表示实体出现的概率，使用向量的方向表示实体的属性（通过非线性转换来保持向量的方向不变但是长度不大于1）。

由于 capsule 的输出是 vector，因此可以使用 动态路由机制 确保 capsule 的输出送到合适的下一层中。一开始会送到所有的下层，但是通过 耦合系数进行 scale，对于每个可能的下一层，capsule 通过加权矩阵计算 预测向量 ，如果预测向量和某个下一层的输出的内积较大，反馈机制会使得对应这个 parent 的耦合系数增加，同时减少其他的 parent。从而进一步增加了这个 parent 的贡献。这种路由机制比 max pooling 要更有效。

## capsule 输入输出计算过程

使用 squashing 函数来对向量进行非线性压缩：$$\mathbf{v}_j=\frac{\left\|\mathbf{s}_j\right\|^2}{1+\left\|\mathbf{s}_j\right\|^2} \frac{\mathbf{s}_j}{\left\|\mathbf{s}_j\right\|}$$
其中，$\mathbf{v}_j$ 为 capsule $j$ 的输出，$s_j$ 为输入。

对于每层（除第一层）的capsule，输入 $s_j$ 计算为预测向量 $\hat{\mathbf{u}}_{j \mid i}$ 的加权和，预测向量计算为 上一层的输出 $\mathbf{u}_i$ 乘以加权矩阵 $\mathbf{W}_{i j}$，整体表述为：$$\mathbf{s}_j=\sum_i c_{i j} \hat{\mathbf{u}}_{j \mid i}, \quad \hat{\mathbf{u}}_{j \mid i}=\mathbf{W}_{i j} \mathbf{u}_i$$
这里的 $c_{ij}$ 为 耦合系数，在动态路由过程中迭代确定。

capsule $i$ 和上一层中所有的 capsule 之间的耦合系数之和为 1，并且是通过 “routing softmax” 决定的。其初始 logits $b_{ij}$ 为对数先验概率：$$c_{i j}=\frac{\exp \left(b_{i j}\right)}{\sum_k \exp \left(b_{i k}\right)}$$
对数先验概率 $b_{ij}$ 初始化为 0，并且通过以下公式进行更新：$$b_{i j} \leftarrow b_{i j}+\hat{\mathbf{u}}_{j \mid i} \cdot \mathbf{v}_j$$

这张图可能可以更好地解释动态路由地过程：![](./image/Pasted%20image%2020221227214127.png)

![](./image/Pasted%20image%2020230210124502.jpg)
更多细节见博客 https://shelleyhlx.blog.csdn.net/article/details/83058667

## 损失函数
capsule network 的损失是：$$L_k=T_k \max \left(0, m^{+}-\left\|\boldsymbol{v}_c\right\|\right)^2+\lambda\left(1-T_k\right) \max \left(0,\left\|\boldsymbol{v}_k\right\|-m^{-}\right)^2$$
这里的 $T_c$ 非 0 即 1，表明是不是这个类，其实就是一个二分类的问题。$T_k=1$ 表明类别 $k$ 存在，$m^+,m^-$ 代表存在和不存在的 margin 值。

# Matrix capsule with EM routing

## capsule 的工作原理

capsule network 包含几层 capsules。设第 $L$ 层的 capsule 的集合为 $\Omega_L$，每个 capsule 都包含一个 $4\times 4$ 的姿态矩阵 $M$ 和一个激活概率 $a$。
> 类似于标准神经网络中的激活，依赖于当前的输入，且不会被存储

在第 $L$ 层的每个 capsule $i$ 和第 $L+1$ 层的每个 capsule $j$ 之间，都有一个 $4\times 4$ 的可训练的变换矩阵 $W_{ij}$。这些矩阵是唯一存储的参数并且是有区分性地进行学习的。

capsule $i$ 的姿态矩阵通过 $W_{ij}$ 进行变换，以对 capsule $j$ 的姿态矩阵进行投票：$V_{i j}=M_i W_{i j}$。第 $L+1$ 层的所有的 capsule 的姿态和激活都是通过非线性的路由过程进行计算的，计算过程的输入是 $V_{ij},a_i$。

这里指的非线性路由过程是 EM 过程。通过迭代来调整 $L+1$ 层 capsule 的 均值、方差和激活概率和调整 $i \in \Omega_L, j \in \Omega_{L+1}$ 之间的分配概率。

## 使用 EM 进行 协议路由

假设已经有了一层中的所有的 capsule 的姿态矩阵和激活概率，想要决定在高层激活哪些 capsule，以及如何将每个 low level 的 capsule 分配给 high level 的 capsule。且高层中的每个 capsule 都对应一个高斯分布，而低层中的每个激活的 capsule 都对应一个数据点（如果 capsule 是部分激活的，则对应于数据点的一个比例）。

如何决定要不要激活高层的 capsule：
+ 如果不激活，设定一个固定的损失
+ 如果激活，设一个固定的损失

## EM routing

### 直接套用 EM 

此时就相当于把 上一层的 capsule 作为 GMM 的训练数据，把下一层的 capsule 作为高斯分布的均值（其实就是软加权的聚类中心）：
$$\text { 新动态路由1: }\left\{\begin{array}{l}
p_{i j} \leftarrow N\left(\boldsymbol{P}_i ; \boldsymbol{\mu}_j, \boldsymbol{\sigma}_j^2\right) \\
R_{i j} \leftarrow \frac{\pi_j p_{i j}}{\sum_{j=1}^k \pi_j p_{i j}}, r_{i j} \leftarrow \frac{R_{i j}}{\sum_{i=1}^n R_{i j}} \\
\boldsymbol{M}_j \leftarrow \sum_{i=1}^n r_{i j} \boldsymbol{P}_i \\
\boldsymbol{\sigma}_j^2 \leftarrow \sum_{i=1}^n r_{i j}\left(\boldsymbol{P}_i-\boldsymbol{M}_j\right)^2 \\
\pi_j \leftarrow \frac{1}{n} \sum_{i=1}^n R_{i j}
\end{array}\right.$$
这里的 $\boldsymbol{M}_j$ 代表下一层的 capsule，其实本质上就是 GMM 的 $\boldsymbol{\mu}_j$。$\boldsymbol{P}_i$ 为上一层的输入 capsule，本质就是 训练数据 $\boldsymbol{x}_i$。

但是上面的公式没有涉及到激活值 $a_j$。

### 添加激活值

不能选择 $\pi_j$ 作为激活值 $a_j$ 的原因：
+ $\pi_j$ 是归一化的，但是激活值表现的是一个特征的显著程度，不是那种非你即我的排斥关系

论文给的公式是：$$a_j=\operatorname{logistic}\left(\lambda\left(\beta_a-\beta_u \sum_i R_{i j}-\sum_h\operatorname{cost}_j^h\right)\right)$$
在 GMM 中，第 $j$ 个高斯分布（的均值表示了聚类中心），而分布的 ”不确定性程度“ 表示了类的紧凑程度，如果类越紧凑（越”团结“），则激活值越大。

通常，不确定性是用信息熵来表述的，则第 $j$ 个分布的信息熵计算为：$$\begin{aligned}
S_j & =-\int p(\boldsymbol{x} \mid j) \ln p(\boldsymbol{x} \mid j) d \boldsymbol{x} \\
& =-\frac{1}{p(j)} \int p(j \mid \boldsymbol{x}) p(\boldsymbol{x}) \ln p(\boldsymbol{x} \mid j) d \boldsymbol{x} \\
& =-\frac{1}{p(j)} E[p(j \mid \boldsymbol{x}) \ln p(\boldsymbol{x} \mid j)] \\
& =-\frac{1}{n \pi_j} \sum_{i=1}^n R_{i j} \ln p_{i j} \\
& =-\frac{1}{\sum_{i=1}^n R_{i j}} \sum_{i=1}^n R_{i j} \ln p_{i j} \\
& =-\sum_{i=1}^n r_{i j} \ln p_{i j}
\end{aligned}$$
如果多维高斯分布的维度为 $d$（其实也就是向量或者矩阵的维度，例如在论文中就是 $16$），则上面的信息熵计算为：$$S_j=\frac{d}{2}+\left(\sum_{l=1}^d \ln \boldsymbol{\sigma}_j^l+\frac{d}{2} \ln (2 \pi)\right) \sum_i r_{i j}$$
对应于论文中给的公式就是，$$\begin{aligned}
\operatorname{cost}_j^h & =\sum_i-r_{i j} \ln \left(P_{i \mid j}^h\right) \\
& =\frac{\sum_i r_{i j}\left(V_{i j}^h-\mu_j^h\right)^2}{2\left(\sigma_j^h\right)^2}+\left(\ln \left(\sigma_j^h\right)+\frac{\ln (2 \pi)}{2}\right) \sum_i r_{i j} \\
& =\left(\ln \left(\sigma_j^h\right)+\frac{1}{2}+\frac{\ln (2 \pi)}{2}\right) \sum_i r_{i j}
\end{aligned}$$
不过这里的 $\operatorname{cost}_j^h$ 只是求了一维，也就是令前面的 $d=1$，然后再对上标 $h$ 表示维度来进行求和（相当于是多个 $1$ 维的进行相加，而不是一个多维的，最终得到的结果都是一个标量）。

### 用上这个激活值

将激活值 $a_j$ 用在 EM routing 中，得到的结果如下：$$\text { 新动态路由3: }\left\{\begin{array}{l}
p_{i j} \leftarrow N\left(\boldsymbol{P}_i ; \boldsymbol{\mu}_j, \boldsymbol{\sigma}_j^2\right) \\
R_{i j} \leftarrow \frac{a_j p_{i j}}{\sum_{j=1}^k a_j p_{i j}}, r_{i j} \leftarrow \frac{a_i^{l a s t} R_{i j}}{\sum_{i=1}^n a_i^{l a s t} R_{i j}} \\
\boldsymbol{M}_j \leftarrow \sum_{i=1}^n r_{i j} \boldsymbol{P}_i \\
\boldsymbol{\sigma}_j^2 \leftarrow \sum_{i=1}^n r_{i j}\left(\boldsymbol{P}_i-\boldsymbol{M}_j\right)^2 \\
\operatorname{cost}_j \leftarrow\left(\beta_u+\sum_{l=1}^d \ln \boldsymbol{\sigma}_j^l\right) \sum_i r_{i j} \\
a_j \leftarrow \operatorname{sigmoid}\left(\lambda\left(\beta_a-\operatorname{cost}_j\right)\right)
\end{array}\right.$$

### 实际论文的思路

![](./image/Pasted%20image%2020230119110130.png)
有好多解释不清楚的。。。。

## 损失

为了使训练对模型的初始化和超参数不太敏感，使用 spread loss 来直接最大化目标类 $a_t$ 的激活和其他类的激活之间的差距，如果激活了一个错误的类 $a_i$，且和 $a_t$ 的距离小于 margin 值 $m$，通过以下平方损失来进行惩罚：$$L_i=(\max \left(0, m-\left(a_t-a_i\right)\right)^2, \quad L=\sum_{i \neq t} L_i$$
开始 margin 为 0.2，后面线性增加到 0.9。


# Stacked Capsule Autoencoders

> object 和 part 之间的变换矩阵 $OP$ 是视角不变的，而 viewer 和 object 之间的变换矩阵 $OV$ 是视角等变的。


SCAE 可以包括三个部分：
+ CCAE：使用二维的点作为 part，坐标作为系统的输入。CCAE 将点建模为相似集群，每个集群都经过独立的相似性变换，CCAE 在未知集群数量情况下，将每个点分配给对应的集群
+ PCAE 学习从图像中推理 part 及其 pose
+ OCAE 和 CCAE 很像，堆叠在 PCAE 上得到 SCAE

## CCAE
> CCAE 相当于 toy example，目标是对二维平面的点进行无监督聚类

设 $\left\{\mathbf{x}_m \mid m=1, \ldots, M\right\}$ 为一系列二维输入点，每个点都会属于一个集群（**事实上，这里的每个点就相当于是 part**），使用 Set Transformer 将所有的点 encode 成 $K$ 个object capsule，第 $k$ 个 capsule 包含一个特征向量 $\boldsymbol{c}_k$、存在概率 $a_k\in[0,1]$ 和一个 $3\times 3$ 的 object-viewer-relationship 即 OV 矩阵，用于表示集群和 viewer 之间的仿射变换，每个 object capsule 只能包含一个 object。
decode阶段，使用独立的 MLP 从特征向量 $\boldsymbol{c}_k$ 中预测 $N$ 个候选（$N\leq M$），每个 候选（part）包含
+ 条件概率 $a_{k, n} \in[0,1]$ 表示某个候选 part 的存在，
+ 标量标准差 $\lambda_{k,n}$（高斯分布的方差）
+ $3 \times 3$ 的 object-part-relationship 即 OP 矩阵，用于表示 object 和 part 之间的仿射变换
候选预测 $\mu_{k,n}$ 通过 OV 和 OP 矩阵的乘积给出（也就是高斯分布的均值）。
> 从 object 预测 part 的候选就相当于是 decode 过程，decode 过程中把每个 part 都看成一个高斯分布来建模，所以最后的目标函数的最大化重构数据的似然。

更为正式的表述如下：$$\begin{aligned}
& \mathrm{OV}_{1: K}, \mathbf{c}_{1: K}, a_{1: K}=\mathrm{h}^{\mathrm{caps}}\left(\mathbf{x}_{1: M}\right) \\
& \mathrm{OP}_{k, 1: N}, a_{k, 1: N}, \lambda_{k, 1: N}=\mathrm{h}_{\mathrm{k}}^{\text {part }}\left(\mathbf{c}_k\right) \\
& V_{k, n}=\mathrm{OV}_k \mathrm{OP}_{k, n} \\
& p\left(\mathbf{x}_m \mid k, n\right)=\mathcal{N}\left(\mathbf{x}_m \mid \mu_{k, n}, \lambda_{k, n}\right)
\end{aligned}$$
训练时是无监督的，通过最大化以下似然函数实现：$$p\left(\mathbf{x}_{1: M}\right)=\prod_{m=1}^M \sum_{k=1}^K \sum_{n=1}^N \frac{a_k a_{k, n}}{\sum_i a_i \sum_j a_{i, j}} p\left(\mathbf{x}_m \mid k, n\right)$$
$M$ 代表点的数量，$K$ 代表 object 的数量（论文中的例子是 3 个），$N$ 代表候选的数量。最后分类结果为：$k^{\star}=\arg \max _k a_k a_{k, n} p\left(\mathbf{x}_m \mid k, n\right)$。

### Set Transformer

本文使用的 set transformer 将 part 编码成 object，其输入为 $n$ 个顺序无关的样本集合 $X\in\mathbb{R}^{n\times d}$，输出为 $k$ 个样本集合 $O\in\mathbb{R}^{k\times d}$。

Transformer 的编码器是下面的其中一种：$$\begin{gathered}
\mathrm{Z}=\operatorname{Encoder}(\mathrm{X})=\operatorname{SAB}(\operatorname{SAB}(\mathrm{X})) \in \mathbb{R}^{\mathrm{n} \times \mathrm{d}} \\
\mathrm{Z}=\operatorname{Encoder}(\mathrm{X})=\operatorname{ISAB}_{\mathrm{m}}\left(\operatorname{ISAB}_{\mathrm{m}}(\mathrm{X})\right) \in \mathbb{R}^{\mathrm{n} \times \mathrm{d}}
\end{gathered}$$
解码器为：$$\mathrm{O}=\operatorname{Decoder}(\mathrm{Z} ; \lambda)=\operatorname{rFF}\left(\operatorname{SAB}\left(\operatorname{PMA}_{\mathrm{k}}(\mathrm{Z})\right)\right) \in \mathbb{R}^{\mathrm{kxd}}$$
整个架构如图：![](./image/Pasted%20image%2020230120114949.png)
 



## PCAE

如果要将图像解释为一些 part 的几何排列需要知道：
+ 图中有哪些 part
+ 推断 part 相对于 viewer 的关系（姿态 pose），例如 在 CCAE 中，part 就是2维的点，但是在 PCAE 中，每个 part capsule $m$ 包含：
	+ 6 维的pose（两维旋转，两维平移，一维缩放一维裁剪）$\boldsymbol{x}_m$
	+ 存在 presence 概率 $d_m\in[0,1]$
	+ unique identity $\boldsymbol{z}_m$（part 的特征）
将 part discovery 的问题看成是 auto-encoding：
+ encoder 学习推断不同 capsule 的 pose 和 presense
+ decoder 从每个 part 中学习一个 图像模板 $T_m$，如果 part 存在，对应的 模板通过 pose $\boldsymbol{x}_m$ 进行仿射变换得到 $\widehat{T}_m$。最后从这些转换后的 template 生成图像
PACE 是在 OCAE 后面的，OCAE 有点像 CCAE。

设 $\mathbf{y} \in[0,1]^{h \times w \times c}$ 为图像，part capsule 的数量为 $M$，使用 encoder 推理每个 capsule 的 $\boldsymbol{x}_m,d_m,\boldsymbol{z}_m\in\mathbb{R}^{c_z}$，这里的 $\boldsymbol{z}_m$ 可以用来修改 template。

模板 template $T_m \in[0,1]^{h_t \times w_t \times(c+1)}$ 比图像小，但是有一个额外的 alpha 通道，允许其他的 template 对它进行 mask，使用 $T_m^a$ 表示 alpha 通道，$T_m^c$ 表示 RGB 通道。

encoder 是基于 CNN 的，使用了 attention-based pooling，图像是通过 spatial Gaussian mixture 建模的，整个 AE 的过程为：$$\begin{aligned}
& \mathbf{x}_{1: M}, d_{1: M}, \mathbf{z}_{1: M}=\mathrm{h}^{\text {enc }}(\mathbf{y}) \quad \text { predict part capsule parameters, } \\
& \boldsymbol{c}_m=\operatorname{MLP}\left(\mathbf{z}_m\right) \quad \text { predict the color of the } \mathrm{m}^{\text {th }} \text { template, } \\
& \widehat{T}_m=\text { TransformImage }\left(T_m, \mathbf{x}_m\right) \quad \text { apply affine transforms to image templates, } \\
& p_{m, i, j}^y \propto d_m \widehat{T}_{m, i, j}^a \quad \text { compute mixing probabilities,} \\
& p(\mathbf{y})=\prod_{i, j} \sum_{m=1}^M p_{m, i, j}^y \mathcal{N}\left(y_{i, j} \mid \boldsymbol{c}_m \cdot \widehat{T}_{m, i, j}^c ; \sigma_y^2\right)\quad \text {calculate the image likelihood.}
\end{aligned}$$

## OCAE

在识别了 part 和其参数之后，下一步就是发掘由这些 part 组成的 object 。

> OCAE 的 encoder 将 part 合并成 object，encoder 的主要架构是 Set Transformer，用来建模置换不变性和得到高阶的 object capsule，这个 object capsule 包含前面 CCAE 提到的三个部分：
> + 特征向量 $\boldsymbol{c}_k$
> + 存在概率 $a_k\in[0,1]$ 
> + $3\times 3$ 的 object-viewer-relationship 即 OV 矩阵 ${OV}_k$

将 $\boldsymbol{x}_m,\boldsymbol{z}_m,T_m$ 进行 拼接，作为 OCAE 的输入，和 CCAE 的不同在于：
+ 将 part capsule 的 $d_k$ 输入 到 OCAE 的 encoder 中，使得 Set Transformer 不考虑缺失的点
+ $d_m$ 也被用于加权 part capsule 的似然，从而也可以不用考虑缺失点的似然（将 $d_m$ 作为似然计算的幂，当 $d_m$ 为 $0$ 时，下面的似然为 $1$，对最终的优化没有影响）
+ 每个 part pose 都看成是来自于 object capsule 的独立混合预测，每个 object capsule 选择 $M$ 个候选预测 $V_{k, 1: M}$
最终，part-capsule 的似然为：$$p\left(\mathbf{x}_{1: M}, d_{1: M}\right)=\prod_{m=1}^M\left[\sum_{k=1}^K \frac{a_k a_{k, m}}{\sum_i a_i \sum_j a_{i, j}} p\left(\mathbf{x}_m \mid k, m\right)\right]^{d_m}$$
通过最大化上式的似然来训练 OCAE。

> 因为 OCAE 要根据 part 生成 object，所以损失是关于 part 的，这个时候已经和图像没有关联了。
> OCAE 通过优化（最大化）上面的似然进行训练。

SCAE 的整个过程如图：![](./image/Pasted%20image%2020230119171349.png)

还有一张图解释得更为清楚：![](./image/Pasted%20image%2020230127104958.png)

# Siamese Capsule Networks
> 好像没找到发表的期刊或者会议？

1. 在受控和非受控环境进行人脸验证试验，引入 Siamese capsule network，用于 pairwise learning 的任务
2. 使用对比损失训练 $l2$ 归一化的 capsule 特征，在 pairwise learning 中可以得到和 baseline 相竞争，在 few shot learning 中可以得到最好的结果
## Introduction（略）

## capsule network
原理见前面的。

一些 capsule 的相关工作：
+ SegCaps，使用局部连接的动态路由方案来减少参数数量
+ Spectral Capsule Networks，收敛性更快，用于医学诊断

## Siamese Capsule Network
![](./image/Pasted%20image%2020230130213635.png)
用于人脸验证的 Capsule Network 旨在识别面部特征及其姿态的部分-整体关系，这反过来通过在成对图像中对齐 capsule 特征来提高相似性度量。

第一层是卷积，输入图像对 $\left\langle x_1, x_2\right\rangle \in \mathbb{R}^{100 \times 100}$，第二层为 primary capsule 层，第三层为 face capsule 层，为面部特征路由后的结果。然后把所有的输出进行拼接后通过全连接层，然后通过 sigmoid 函数来控制 dropout，且把 squash 函数换成了 tanh。

capsule 表征：每张输入图片最后得到的是 32 个capsule，每个 capsule 为 512 维，然后通过 20 个激活单元，为了确保每个 capsule 都可以被激活且 dropout rate 是从每个capsule 中学习的，采用 sigmoid 函数进行学习。

损失函数：原始的损失函数是 margin loss 和 spread loss，本文使用的是 contrastive margin loss，前面 capsule 得到的表征使用相似度计算函数 $d_w$ 计算了相似度得分，对比损失的目标是使得 pose 相同的表征接近而不同的远离，定义为：$$L_c(\omega)=\sum_{i=1}^m\left(\frac{1}{2}\left(1-y^{(i)}\right) D_\omega^{(i)}+\frac{1}{2} y^{(i)} \max \left(0, m-D_\omega^{(i)}\right)\right)$$
其中，$D_w=\left\|f^\omega\left(x_1\right)-f^\omega\left(x_2\right)\right\|_2^2$ 为两个表征之间的欧式距离，$y \in[-1,1]$，$m$ 为 margin。
> 没用使用重构损失，因为压根没有重构的过程。

优化：采用 AMSGrad 加快参数更新。

## 实验和结果（略）
