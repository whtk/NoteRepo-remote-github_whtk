
1. 提出 wav2vec，在大量未标记的音频数据上进行无监督训练
2. 模型基于多层卷积神经网络，使用 noise contrastive 二分类任务进行优化


## Introduction

本文使用无监督预训练来改进有监督语音识别。提出 wav2vec模型，本质是一个卷积神经网络，输入原始音频，输出语音表征。目标函数是对比损失，通过区分真实的未来音频样本（正样本）和负样本

wav2vec依为全卷积的结构，与先前工作中的递归模型相比，可以在现代硬件上轻松并行化。

WSJ基准测试的实验结果表明，基于约1000小时未标记语音的预训练表征可以显著改进基于字符的ASR系统。

## 预训练方法

给定音频信号作为输入，模型预测给定上下文时的未来样本。这种方法的一个常见问题是需要对数据分布 $p(\mathbf{x})$ 进行精确建模，但是通常很难。

本文通过以较低的时间分辨率将原始语音样本 $\mathbf{x}$ 编码成特征 $\mathbf{z}$，然后隐式地建模 density ratio $p\left(\mathbf{z}_{i+k} \mid \mathbf{z}_i \ldots \mathbf{z}_{i-r}\right) / p\left(\mathbf{z}_{i+k}\right)$，这一过程和 [[Representation Learning with Contrastive Predictive Coding 笔记]] 相似。

###  模型

模型输入为原始音频信号，然后用了两个网络：
+ encoder network 将音频信号嵌入到潜在空间中
+ context network 组合encoder的多个 time step 的输出以获得上下文表征。
模型如图：![[image/Pasted image 20221204215354.png]]

给定原始音频样本 $\mathbf{x}_i \in \mathcal{X}$、 使用 encoder network 为 $f: \mathcal{X} \mapsto \mathcal{Z}$ ，网络由五层卷积网络组成。这五层的kernel size为（10、8、4、4、3），strides 为（5、4、2、2、3）。

encoder 的输出是低频特征表示 $\mathbf{z}_i \in \mathcal{Z}$，每个向量 $\mathbf{z}_i$ 编码了16kHz音频的30ms，间隔为 10ms。

然后对 encoder network 的输出使用 context network $g: \mathcal{Z} \mapsto \mathcal{C}$ ，将多个潜在表征 $\mathbf{z}_i \ldots \mathbf{z}_{i-v}$ 转化为一个上下文向量 $\mathbf{c}_i=g\left(\mathbf{z}_i \ldots \mathbf{z}_{i-v}\right)$。

context network 有九层，kernel size 都是3，stride 都是1。总的感受野为210ms。

网络中的层 由512个channel 的因果卷积、组归一化（group normalization）层和 ReLU 组成。

训练时对每个样本的特征和时间维度进行归一化，相当于使用单个归一化组进行组归一化。

作者发现，选择一个对输入的缩放和偏移保持不变的标准化方案非常重要，能够产生通用的表示。

对于大数据和大模型，使用了两个额外的线性变换和 skip connection。

### 目标函数

训练模型时，通过最小化每个 step $k＝1,\dots,k$ 的对比损失，将未来为 $k$ 步的样本 $\mathbf{z}_{i+k}$ 与从 proposal 分布 $p_n$ 中提取的负样本 $\tilde{\mathbf{z}}$ 区分开来：$$\mathcal{L}_k=-\sum_{i=1}^{T-k}\left(\log \sigma\left(\mathbf{z}_{i+k}^{\top} h_k\left(\mathbf{c}_i\right)\right)+\underset{\tilde{\mathbf{z}} \sim p_n}{\mathbb{E}}\left[\log \sigma\left(-\tilde{\mathbf{z}}^{\top} h_k\left(\mathbf{c}_i\right)\right)\right]\right)$$
其中，Sigmoid 函数定义为 $\sigma(x)=1 /(1+\exp (-x))$，$\sigma\left(\mathbf{z}_{i+k}^{\top} h_k\left(\mathbf{c}_i\right)\right)$ 表示 $\mathbf{z}_{i+k}$ 为正样本的概率。考虑每个 step 特有的仿射变换 $h_k\left(\mathbf{c}_i\right)=W_k \mathbf{c}_i+\mathbf{b}_k$，通过对 $\mathcal{L}_k$ 求和来优化总的损失 $\mathcal{L}=\sum_{k=1}^K \mathcal{L}_k$ 。实际中，通过从音频序列中基于均匀分布采样噪声（负样本）来近似期望，即 $p_n(\mathbf{z})=\frac{1}{T}$，$T$ 为序列长度，$\lambda$ 为负样本的数量。

> 这里的loss 被修改以考虑二分类预测任务，因为 InfoNCE 的损失为：$$\mathcal{L}_{\mathrm{N}}=-\underset{X}{\mathbb{E}}\left[\log \frac{f_k\left(x_{t+k}, c_t\right)}{\sum_{x_j \in X} f_k\left(x_j, c_t\right)}\right]$$


## 实验 & 结果（略）