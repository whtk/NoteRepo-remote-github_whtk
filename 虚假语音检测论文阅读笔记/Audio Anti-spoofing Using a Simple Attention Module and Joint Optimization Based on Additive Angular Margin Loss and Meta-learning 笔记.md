
1. 本文引入了一个简单的注意力模块来推断卷积层中特征图的三维注意力权重，然后优化能量函数以确定每个神经元的重要程度
2. 提出了一种基于加权加性角裕度损失（Additive Angular Margin Loss）的联合优化方法，用于二分类，并使用元学习训练框架来开发系统，系统对欺骗攻击具有鲁棒性，可用于模型泛化增强

## Introduction

1. 本研究的重点是LA场景中的欺骗攻击
2. [[End-to-End Spectro-Temporal Graph Attention Networks for Speaker Verification Anti-Spoofing and Speech Deepfake Detection 笔记]] 提出基于RawNet2-like 的编码器和图注意力网络的端到端系统，使用两个并行图来同时建模频谱和时间信息。
3. Squeeze-and-excitation（SE） 和 convolutional block attention（CBAM） 模块被用于 ASV 系统也实现大幅提高
4. 有研究受神经科学启发，创建了一个简单的注意力模块，通过优化能量函数来利用每个神经元的注意力权重
5. 本文中，基于附加角裕度（AAM）损失，为每个类别分配不同的权重和裕度，以解决数据不平衡和过拟合问题，同时基于度量的元学习用于自适应地学习不可见攻击样本和已知类之间的共享度量空间
6. 总的来说，本文研究了基于RawNet2的端到端欺骗检测系统的三个扩展，并分析了每个注意力模块在提高系统性能方面的有效性。此外，利用加权AAM损失替代交叉熵损失进行二分，鼓励类内相似和类间可分。此外，我们提出了AAM损失和元学习训练框架的联合优化，该框架集成了 episodic 和 global 的分类。

## 方法

### 基于 RawNet2 的 encoder

采用 [[RawNet2]] 中的结构的变体，最终从输入波形中生成 high-level 的 feature map $F \in \mathbb{R}^{C \times F \times T}$ ，然后 sinc-convolution 层转换成 单通道的二维 $F \times T$ 特征图。

### Attention 模块

SE 模块聚合所有的 feature map 的全局频率信息得到 attention map $\mathbf{M}_f \in \mathbb{R}^{1 \times F \times 1}$，CBAM 顺序推理得到 channel-wise 的 attention map $\mathbf{M}_c \in \mathbb{R}^{C \times 1 \times 1}$ 和 时频 attention map $\mathbf{M}_{f t} \in \mathbb{R}^{1 \times F \times T}$ 。

simple attention module （SimAM）通过优化能量函数来获得每个神经元的重要程度：$$e_t\left(W_t, b_t, \mathbf{y}, x_i\right)=\left(y_t-\hat{t}\right)^2+\frac{1}{M-1} \sum_{i=1}^{M-1}\left(y_o-\hat{x}_i\right)^2$$
给定特征图 $\mathbf{x}$ 中某个通道的 目标神经元 $t$ 和其他神经元，$\hat{t}=W_t t+b_t$，$\hat{x}_i=W_t x_i+b_t$ 为 $t, x_i$ 的线性转换，$M=F\times T$ 为每个 channel 的神经元数量，当 $\hat{x}_i=W_t x_i+b_t,\hat{x}_i=y_t$ 时上式取最小，最终的能量函数（带正则化）中，$y_o,y_t$ 分配一个标签（如 $1,-1$）：$$\begin{aligned}
&e_t\left(W_t, b_t, \mathbf{y}, x_i\right)=\frac{1}{M-1} \sum_{i=1}^{M-1}\left(-1-\left(W_t x_i+b_t\right)\right)^2 \\
&+\left(1-\left(W_t t+b_t\right)\right)^2+\lambda W_t^2
\end{aligned}$$
使用 SGD 进行优化计算量太大，权重可以通过最小化能量函数来获得：$$e_t^*=\frac{4\left(\hat{\sigma}^2+\lambda\right)}{(t-\hat{u})^2+2 \hat{\sigma}^2+2 \lambda}$$
其中，$\hat{u}=\frac{1}{M} \sum_{i=1}^M x_i$ 和 $\hat{\sigma}^2=\frac{1}{M} \sum_{i=1}^M\left(x_i-\hat{u}\right)^2$ ，基于神经科学理论，能量 $e_t^*$ 越低，表示神经元 $t$ 的权重越大。

最终有：$$\tilde{\mathbf{x}}=\sigma\left(\frac{1}{\mathbf{E}}\right) \otimes \mathbf{x}$$
其中，$\mathbf{E}$ 为所有能量值 $e_t^*$ 在 channel 和 时频维度的组合。

### 联合优化 AAM 和 元学习

AAM 损失可以增强类内紧凑性和类间可分性，定义为：$$L_{A A M}=-\frac{1}{B} \sum_{i=1}^B \log \frac{w_{y_i} e^{s\left(\cos \left(\theta_{y_i}+m_{y_i}\right)\right)}}{e^{s\left(\cos \left(\theta_{y_i}+m_{y_i}\right)\right)}+e^{s \cos \theta_{\left(1-y_i\right)}}}$$

元学习使用任务驱动模型通过优化每个子任务（即 episode）来提高其学习能力。
元学习的 support set 和 query 的构建见论文，最终 support set 为 $\mathcal{S}=\left\{\mathbf{x}_i^s\right\}_{i=1}^{(N-1) \times K} \cup\left\{\mathbf{x}_i^g\right\}_{i=1}^K$ ，query set 为 $\mathcal{Q}=\left\{\mathbf{x}_j^s\right\}_{j=1}^K \cup\left\{\mathbf{x}_j^g\right\}_{j=1}^K$ 。

使用 relation network 来 比较 support 和 query set 中的样本，通过神经网络来参数化 comparison metric。  

具体地说，relation network 在一组子任务上同时学习特征表示和 metric，这些子任务可以推广到以前看不见的欺骗攻击。给定表示输入样本及其对应标签的 $(\mathbf{x}, y)$，来自 $\mathcal{S}$ 和 $\mathcal{Q}$ 的样本通过编码器 $f_\theta$ 产生特征图。然后，为了形成一对，将来自 support set 的特征映射 $f_\theta\left(\mathbf{x}_i\right)$ 与来自 query set 中的特征映射 $f_\theta\left(\mathbf{x}_j\right)$ 连接起来。考虑到 $\mathcal{S}(|\mathcal{S}|=N K)$ 和 $\mathcal{Q}(|\mathcal{Q}|=2 K)$ 中的样本数量，每个 episode（相当于一个小批量）都有 $2 N K^2$ 个排列。然后将每个 pairs 输入到 relation 模块 $f_\phi$ 中，该模块产生指示特征图对之间的相似性的标量 relation score：$$r_{i, j}=f_\phi\left(\left[f_\theta\left(\mathbf{x}_i\right), f_\theta\left(\mathbf{x}_j\right)\right]\right)$$
网络将 relation score 作为 similarity measure：$$r_{i, j}= \begin{cases}1, & \text { if } y_i=y_j \\ 0, & \text { otherwise }\end{cases}$$
$f_\theta$ 和 $f_\phi$ 通过 MSE 联合优化：$$L_{M S E}=\frac{1}{2 N K^2} \sum_{i=1}^{N K} \sum_{j=1}^{2 K}\left(r_{i, j}-1\left(y_i==y_j\right)\right)^2$$

整个模型框架如图：![[Pasted image 20221128121427.png]]

## 实验和结果

模型参数配置见 论文。

消融实验结果：![[Pasted image 20221128124249.png]]
说明每个组件都很重要。

和 SOTA 的比较：![[Pasted image 20221128124419.png]]
