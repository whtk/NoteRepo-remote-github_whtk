> ICASSP 2019 的论文

1. 提出了一种重放攻击检测系统 —— Attentive Filtering Networks
2. 由基于注意力的过滤机制和 ResNet 分类器组成，attention-based filtering 增强了 时频域的特征表征
3. ASVspool 2017 2.0 数据集中，AFN 相比于 baseline 获得了较大的改进

## Introduction

先前的工作分为三类：
1. 基于GMM 和 i-vector
2. 基于 DNN
3. 融合模型

[[Audio replay attack detection with deep learning frameworks 笔记]] 采用 LCNN ，并且将 LCNN 和 RNN 层叠来进行欺诈检测。[[An end-to-end spoofing countermeasure for automatic speaker verification using evolving recurrent neural networks 笔记]] 开发  evolution RNN  进行欺诈检测。

本文目标是开发 DNN 系统，利用时频域中的辨别特征进行欺诈检测。其动机是，欺骗攻击的线索可能是随时间变化的，需要一种可以自动获取并增强有区分性的时频特征的系统，于是通过在基于ResNet的分类器之前设计基于注意力机制的过滤器 来消除或增强特征。

ResNet 之前已经被用在 欺诈检测中了（[[ResNet and model fusion for automatic spoofing detection 笔记]]），在设计网络时，受到了 Stimulated training 、convolutional attention network  和 residual attention network 影响。

在分类器端，Dilated Residual Network 使用卷积层而非全连接层，同时通过增加 dilation factor 来修改 residual unit 。

两者一起组成 AFN 模型。

## AFN

### 特征工程

从 logspec 中通过重复所有话语的特征图将其扩展到最长话语的长度创建 unified time-frequency map，如图：![[Pasted image 20221129202022.png]]


### Dilated Residual Network

DRN 结构如图：![[Pasted image 20221129202307.png]]
在 DRM 中添加 dilated convolution layer 的动机是，大多模型都很难泛化到未知的攻击，因此防止过拟合是获得良好性能的一个重要因素，在不损害模型容量的情况下减少小数据集过拟合的影响的一个方法是使用 dilated convolution layer，这个操作 $*_d$ 计算为：$$\left(F *_d G\right)(\mathbf{n})=\sum_{\mathbf{m}_{\mathbf{1}}+d \mathbf{m}_{\mathbf{2}}=\mathbf{n}} F\left(\mathbf{m}_{\mathbf{1}}\right) G\left(\mathbf{m}_{\mathbf{2}}\right), \forall \mathbf{m}_{\mathbf{1}}, \mathbf{m}_{\mathbf{2}}$$
其中，$F$ 为 feature map，$G$ 为 kernel，$d$ 为 dilation rate，加粗的都是向量。  

通过 dilated convolution，网络的感受野随着层深度呈指数增长，从而整合了更广泛和全局背景的信息。

### Attentive Filtering

AF 有选择地累积频域和时域中的辨别性特征。结构为：![[Pasted image 20221129204132.png]]

具体来说，AF 使用  attention heatmap $\mathbf{A}_s$ 来增强输入的  feature map $\mathbf{S}$，得到增强后的 feature map $\mathbf{S}^*$ ，计算为：$$\mathbf{S}^*=\mathbf{A}_{\mathbf{s}} \circ \mathbf{S}+\overline{\mathbf{S}}$$
其中，$\mathbf{S}, \mathbf{S}^* \in \mathbb{R}^{F \times T}$，$F,T$ 分别代表时域和频域的维度。$\circ$ 代表 element-wise 乘法，$+$ 也是 element-wise 的加法，$\overline{\mathbf{S}}$ 是 residual $\mathbf{S}$，本文中，设置 $\overline{\mathbf{s}}=\mathbf{S}$。$\mathbf{A}_{\mathbf{s}}$ 的计算包含 bottom-up and top-down 的操作：$$\mathbf{A}_{\mathbf{s}}=\phi(U(\mathbf{S}))$$
$\phi$ 为非线性变换（如 sigmoid 或 softmax），$U$ 是 U-net 类似的结构，由一系列下采样和上采样操作组成，其输入为 $\mathbf{S}$ 。使用 max pooling 进行下采样，使用双线性插值进行上采样，此外，还添加了相应的skip connection，以帮助学习注意力权重。  

下图为使用和不使用 $\overline{\mathbf{S}}$ 学习到的 attention heatmap：![[Pasted image 20221129204323.png]]
可以看到，用 $\overline{\mathbf{S}}$ 训练 attention heatmap 时，注意力更多集中在语音段的高频分量上。直观的解释是，通过向DRN提供 $\overline{\mathbf{S}}$ 的全部信息，AF可以专注于学习注意力权重。还可以看到，使用 $\overline{\mathbf{S}}$，AF 不仅可以选择性地关注和增强高频段，还可以关注和增强任何时间和频率段。

### 优化

AF和DRN是 AFN 的两个组成部分，网络是端到端训练的。使用Xavier初始化初始化 AF 网络，使用Adam优化。

## 实验

baseline：CQCC-GMM、i-vector、LCNN

数据集：ASVspool 2017 2.0

使用 BOSARIS工具包进行分数融合。

单系统结果：![[Pasted image 20221129211435.png]]
带有 Sigmoid 的 AF 效果最好。模型也还是有点过拟合。

融合结果：![[Pasted image 20221129211645.png]]
进一步降低了 EER。

单系统的结果表明，系统的性能取决于AF中使用的非线性激活函数。Softmax-T和Softmax-F强制稀疏激活，而Sigmoid显示了多个时间和频率bin中的激活：![[Pasted image 20221129212252.png]]