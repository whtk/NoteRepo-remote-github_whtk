> [IEEE Signal Processing Letters](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=97)，2021，University of Rochester

1. 提出使用 one class learning 来检测 未知的合成语音的欺诈攻击
2. 关键思想是 compact 真实语音表征，并在 embedding 空间中 加入 angular margin 来分离欺骗攻击
3. 不使用数据增强时，在 ASV spoof 2019 LA 中优于所有的单系统


## Introduction

1. 传统的方法专注于特征工程，好的特征如 CFC-CIF、LFCC、CQCC；后端则一般用 GMM
2. [[An Investigation of Deep-Learning Frameworks for Speaker Verification Antispoofing 笔记]] 研究了 DNN 和 RNN 模型用于欺诈检测可以提高鲁棒性
3. [[A Light Convolutional GRU-RNN Deep Feature Extractor for ASV Spoofing Detection 笔记]] 采用 light convolutional gated RNN 提高对欺诈检测中的长期依赖性
4. [[Light Convolutional Neural Network with Feature Genuinization for Detection of Synthetic Speech Attacks 笔记]] 提出 一种基于 Feature Genuinization 的LCNN系统，在检测合成攻击方面优于其他单系统
5. [[Audio Spoofing Verification using Deep Convolutional Neural Networks by Transfer Learning 笔记]] 探索了基于 ResNet 的迁移学习方法
6. 为了进一步提高性能，[[Spoofing Attack Detection using the Non-linear Fusion of Sub-band Classifiers 笔记]] 提出基于子带建模的特征融合
7. 但现有方法通常很难泛化到未知攻击，作者认为这种问题来自于二分类的假设，因为有一类是假的，因此在训练和测试的时候存在不匹配现象
8. 而在 one class 分类中则不存在上述问题，one class 分类方法的关键在于获得目标类的分布同时设定一个较紧的分类边界，使得所有非目标类都在界外
9. 本文提出 one-class softmax 的损失函数学习特征空间，其中真实语音嵌入具有紧凑的边界，而欺骗数据与真实数据保持一定的距离

## 方法
> 本节首先分析了二分类的损失函数，然后提出了 one class 的学习损失

softmax、am-softmax、oc-softmax 损失边界对比：![[Pasted image 20221210195220.png]]

### 二分类损失

原始的 二分类 softmax 为：$$\begin{aligned}
\mathcal{L}_S & =-\frac{1}{N} \sum_{i=1}^N \log \frac{e^{\boldsymbol{w}_{y_i}^T \boldsymbol{x}_i}}{e^{\boldsymbol{w}_{y_i}^T \boldsymbol{x}_i}+e^{\boldsymbol{w}_{1-y_i}^T \boldsymbol{x}_i}} \\
& =\frac{1}{N} \sum_{i=1}^N \log \left(1+e^{\left(\boldsymbol{w}_{1-y_i}-\boldsymbol{w}_{y_i}\right)^T \boldsymbol{x}_i}\right)
\end{aligned}$$
其中，$\boldsymbol{x}_i \in \mathbb{R}^D,y_i \in\{0,1\}$ 分别为 embedding vector 和 第 $i$ 个样本的标签，$\boldsymbol{w}_0, \boldsymbol{w}_1 \in \mathbb{R}^D$ 为两个类的权值向量。

AM-softmax 通过引入 angular margin 来使得两个类的 分布更为紧凑：$$\begin{aligned}
\mathcal{L}_{A M S} & =-\frac{1}{N} \sum_{i=1}^N \log \frac{e^{\alpha\left(\hat{\boldsymbol{w}}_{y_i}^T \hat{\boldsymbol{x}}_i-m\right)}}{e^{\alpha\left(\hat{\boldsymbol{w}}_{y_i}^T \hat{\boldsymbol{x}}_i-m\right)}+e^{\alpha \hat{\boldsymbol{w}}_{1-y_i}^T \hat{\boldsymbol{x}}_i}} \\
& =\frac{1}{N} \sum_{i=1}^N \log \left(1+e^{\alpha\left(m-\left(\hat{\boldsymbol{w}}_{y_i}-\hat{\boldsymbol{w}}_{1-y_i}\right)^T \hat{\boldsymbol{x}}_i\right)}\right)
\end{aligned}$$
其中，$m$ 为 margin，$\hat{\boldsymbol{w}}, \hat{\boldsymbol{x}}$ 为归一化的向量。

### one class 损失

AM-softmax 使用相同的 margin，$m$ 越大，embedding 越紧凑（夹角越小）。

对于真实语音，当然是紧凑一点好，但是对于 虚假的语音，如果 embedding 过于紧凑，可能会过拟合已知攻击从而无法泛化到未知攻击。因此提出了两个 margin ，定义 one-class softmax 如下：$$\mathcal{L}_{O C S}=\frac{1}{N} \sum_{i=1}^N \log \left(1+e^{\alpha\left(m_{y_i}-\hat{\boldsymbol{w}}_0 \hat{\boldsymbol{x}}_i\right)(-1)^{y_i}}\right)$$
> 这里的向量也都是归一化的。

公式里面只有权重 $\boldsymbol{w}_0$，表明只需要优化目标类（也就是真实类）的参数。两个边界满足 $m_0, m_1 \in[-1,1], m_0>m_1$ 分别对应真实语音和虚假语音。
> $m$ 越大，代表边界夹得越紧。

## 实验

数据集：ASVspoof 2019 LA

训练参数：60 维 LFCC，采用论文 [[Generalized end-to-end detection of spoofing attacks to automatic speaker recognizers 笔记]] 的网络架构，OC-Softmax 损失函数中，有$\alpha=20,m_0=0.9,m_1=0.2$，采用 Adam 优化器。

结果：

不同损失函数的效果：![[Pasted image 20221210201613.png]]
1. 在dev集上的效果差不多，但是在评估的时候，对于未知攻击，oc-softmax 能提升性能

和其他系统比较：![[Pasted image 20221210201641.png]]
结论：提出的系统显著优于所有其他单系统。

