> IEEE TRANSACTIONS ON INFORMATION FORENSICS AND SECURITY，2024 年收录，广东省信息安全重点实验室、中山大学等

1. 现有深度学习虚假检测很难泛化到未知攻击，且 class imbalance 问题使得学习过程偏向于已知攻击
2. 提出端到端的 One-Class Neural Network + Statistics Pooling 的方法：
    1. 采用 feature cropping 来削弱高频成分，减少过拟合
    2. 采用 directed statistics pooling 来提取更有效的特征
    3. 采用 Threshold One-class Softmax loss，通过减少 spoofing 样本的优化权重来缓解 class imbalance
3. 在 ASVspoof 2019 LA 上 EER 为 0.44%，minDCF 为 0.0145
4. 在 reproducible 的 ensemble models 上保持 SOTA

## Introduction 

1. 传统的模型通常采用 binary loss functions 如 Softmax 或 weighted cross entropy (WCE) 来优化网络，但是会限制其检测未知攻击的能力
2. ASVspoof 2019 LA 数据集中，训练集和测试集中的 spoofing 样本分布差异很大，传统的二分类方法很难解决这个问题
3. [One-class Learning Towards Synthetic Voice Spoofing Detection 笔记](One-class%20Learning%20Towards%20Synthetic%20Voice%20Spoofing%20Detection%20笔记.md) 提出了一种 one-class learning 的方法，通过引入两个不同的 margin，来 compact bonafide 样本和 push away spoofing 样本
4. 但是 class imbalance 问题会导致 biased training，即 spoofing 样本在更新模型时梯度贡献更多，导致模型偏向于将 spoofing 样本推出边界
5. 本文提出 Thresholding One-class Softmax (TOC-Softmax) loss，通过引入 thresholding 来减少 spoofing 样本的优化权重
6. 然后提出 One-Class Network with Directed Statistics Pooling (OCNet-DSP) 模型，基于 [AASIST- Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks 笔记](AASIST-%20Audio%20Anti-Spoofing%20using%20Integrated%20Spectro-Temporal%20Graph%20Attention%20Networks%20笔记.md)，但是不使用图神经网络而是用更简单的模型来同时编码 spectral 和 temporal 特征
7. 还提出了 feature cropping 操作，减少 spectrogram 的高频部分的影响
8. 且发现 bonafide 和 spoofing 之间存在时域和频域特征的统计差异，因此引入了 directed statistics pooling (DSP) 层，计算 statistical vector 来捕获特征图中频谱特征的时域变化

## 模型

由于检测的语音通常会受到各种未知攻击，而 spoofing 样本的数量通常远远超过 bonafide 样本，因此使用常用的二分类损失函数会过拟合或泛化能力差。提出的 One-Class neural network with directed statistics pooling for spoofing speech detection 模型如图：
![](image/Pasted%20image%2020240127154834.png)
包含五个模块：
+ 频谱提取
+ 特征提取
+ feature cropping
+ directed statistics pooling 
+ 全连接层

每个模块的配置如下：
![](image/Pasted%20image%2020240127154951.png)

### 频谱提取

模型输入为长为 $L$ 秒的语音，采样率为 $f_s$，总样本书为 $f_s \times L$。实验中 $f_s = 16KHz$，$L = 8$。通过 sinc layer 对原始语音进行时域卷积。sinc layer 包含一系列时域带通滤波器函数 $h$，如下：
$$\begin{aligned}h\left(n;\:f_1,\:f_2\right)=&2f_2\frac{\sin\left(2\pi nf_2\right)}{2\pi nf_2}-2f_1\frac{\sin\left(2\pi nf_1\right)}{2\pi nf_1}\end{aligned}$$

其中 $n$ 是时间变量，$n \in \{-64, -63, \cdots, +64\}$。

将所有频率（即 $[0, 8kHz]$）分为 $S$ 个（实验中 $S = 70$）不重叠的等长部分，构建 $S$ 个带通滤波器，其中 $f_1^{(i)}=\frac{8kHz}{S}\times(i-1)$，$f_2^{(i)}=\frac{8kHz}{S}\times i$，$i$ 为带通滤波器的索引，$f_1^{(i)}$ 和 $f_2^{(i)}$ 分别为第 $i$ 个带通滤波器的截止频率，$i=1,2,\cdots,S$。

然后，对输入语音和 sinc layer 中的每个带通滤波器 $h(n; f_1^{(i)}, f_2^{(i)})$ 进行 1D 卷积。由于原始语音的时域存在冗余信息，因此将 sinc layer 中所有滤波器的滑动步长设置为 7 来减少运算量。卷积后得到对应的特征向量 $V^{(i)} \in \mathbb{R}^{1 \times T_0}$，其中 $T_0$ 是滤波后音频的时间序列长度。将所有特征向量 $V^{(i)}$ 组合起来，可以得到 spectrogram $G_0 \in \mathbb{R}^{S \times T_0}$。如图所示：
![](image/Pasted%20image%2020240127161841.png)

最后将 $G_0$ 的绝对值输入到 local maximum pooling 层和 batch normalization 层，得到 spectrogram $G_1 \in \mathbb{R}^{1 \times S \times T_1}$。

### 特征提取

特征提取模块进一步提取 high-level feature map。模块基于 ResNet block。每个 ResNet block 包含两个卷积层、一对 batch normalization 层和 local maximum pooling 层。采用 squeeze-and-excitation (SE) block，用两个全连接层将每个特征通道的全局均值映射到通道权重，然后将其分配给相应的通道。如图 1，SE block 嵌入到 ResNet block 中，构成一个 SE-Residual group。通过五个 SE-Residual groups，可以得到新的特征图 $FM \in \mathbb{R}^{C \times S \times T}$，其中 $C$、$S$ 和 $T$ 分别表示 channels 数、height 和 width。
> 实现发现，移除 SE block 或减少 groups 数会导致网络性能下降。

### feature cropping

已有实验表明，与 high-frequency 或 full-frequency (0-8kHz) 相比，使用 low-frequency 部分可以更好地泛化到未知攻击。但是，直接移除 high-frequency part 会不可避免地丢失一些可以有效检测 seen attacks 的特征。基于上述分析，使用 feature cropping 来选择频率。使用 sinc layer 来提取 spectrogram。此外，feature cropping 操作是在特征提取模块得到的特征图 $FM$ 上进行的，而不是在 spectrogram extraction 得到的 spectrogram 上进行的。

feature maps $FM \in \mathbb{R}^{C \times S \times T}$ 沿着 spectral axis 分为两个部分：lower part $LM_2 \in \mathbb{R}^{C \times S/2 \times T}$ 和 upper part $UM_2 \in \mathbb{R}^{C \times S/2 \times T}$。$LM_2$ 主要反映 spectrogram 的 low-frequency 部分，$UM_2$ 主要反映 high-frequency 部分。因此，选择 $LM_2$ 进行后续分析，可以有效防止模型在 seen attacks 上过拟合。此外，SE block 采用 global average pooling 来捕获 feature maps 的全局信息，使得 channel-wise 权重可以动态调整。且多个卷积扩大了神经网络的感受野，使得网络可以捕获 spectrogram 上更广泛的频率特征。因此，$LM_2$ 也会包含一些 high-frequency 相关的特征。

> 实验也发现，$LM_2$ 相比于 $FM$ 和其他频率相关部分，具有更好的检测性能。

### directed statistics pooling

feature extraction 和 feature cropping 后，现有方法通常会在 feature maps 上加入 pooling 层来降低维度。通常采用 global average pooling (avg-pool) 或 global maximum pooling (max-pool)。但是，这些方法主要捕获 feature map 的全局平均值或最大值，导致丢失了大量的时域和频域统计信息。

为了直观地说明 bonafide 和 spoofing 之间的时域和频域特征差异，首先提取 ASVspoof 2019 训练集中每个语音数据的 spectrogram $G_0$，然后计算每个语音数据的 spectrogram 的绝对值，然后分别对 bonafide 和 spoofing 类别的数据进行平均，得到如下两个特征图：
![](image/Pasted%20image%2020240127163141.png)

可以发现，整个特征图中 bonafide 和 spoofing 之间存在显著的统计差异。为了有效地挖掘 spectrogram $G_0$ 上的这些差异，首先对每个样本的 $G_0$ 进行一些处理（如 pooling、BN & activating、feature extraction 和 feature cropping），然后提出 directed statistics pooling (DSP) 层来封装 spectral 或 temporal domains 上更全面的统计特征。

DSP 中的 “directed” 表示在计算不同轴上的统计特征时的有目的的方向。如图所示：
![](image/Pasted%20image%2020240127163326.png)
有两个 pooling 方向，即 “Spectral-to-temporal” 和其反向 “Temporal-to-spectral”：
+ “Spectral-to-temporal” 会产生统计向量来表示 spectral 特征的 temporal 变化
+ “Temporal-to-spectral” 会提取类似的特征向量来表示 temporal 特征的 spectral 变化

以 “Spectral-to-temporal” 为例，首先计算给定 $t$ 时刻的 spectral axis ($\hat{s}$) 上的均值 $\mu^{c,t}$，其中 $t \in \{1,2,\cdots,T\}$，得到特征向量 $\mu^c_{\hat{s}} = (\mu^{c,1}, \mu^{c,2}, \cdots, \mu^{c,T})^\top$。然后沿 temporal axis ($\hat{t}$) 提取标准差 $\sigma^{c}_{\hat{s},\hat{t}}$ 作为所谓的 spectral features 的 temporal 变化的特征。通过聚合所有 channels 的结果，可以得到矩阵 $\mu_{\hat{s}} = (\mu^1_{\hat{s}}, \mu^2_{\hat{s}}, \cdots, \mu^C_{\hat{s}})$ 和向量 $\sigma_{\hat{s},\hat{t}} = (\sigma^1_{\hat{s},\hat{t}}, \sigma^2_{\hat{s},\hat{t}}, \cdots, \sigma^C_{\hat{s},\hat{t}})^\top$。

最后，使用统计向量 $\sigma_{\hat{s},\hat{t}}$ 来表示 spectral features 的 temporal 变化，输入到全连接层。对于另一个 pooling 方向（即 “Temporal-to-spectral”），可以使用与之前相同的方法获得统计向量 $\sigma_{\hat{t},\hat{s}}$，除了方向的顺序不同。与 $\sigma_{\hat{s},\hat{t}}$ 类似，向量 $\sigma_{\hat{t},\hat{s}}$ 可以表示 temporal features 的 spectral 变化。

> 实验表明，"Spectral-to-temporal" 比 "Temporal-to-spectral" 和它们的合并版本表现更好。说明表示 spectral features 的 temporal 变化的统计向量对模型性能的影响更大，

### 损失函数设计

现有方法通常采用二分类损失函数（如 Softmax）来学习 bonafide 和 spoofing 类的 compact feature distributions。但是其实有很多未知攻击没有出现在训练集中。假设 bonafide 类的分布在训练集和测试集中是一致的，[One-class Learning Towards Synthetic Voice Spoofing Detection 笔记](One-class%20Learning%20Towards%20Synthetic%20Voice%20Spoofing%20Detection%20笔记.md) 提出了 One-class Softmax (OC-Softmax) loss 来学习嵌入特征空间，其中 bonafide 类具有 compact distribution，而 spoofing 类则位于 angular margin 之外。然而，OC-Softmax 的性能仍然受到 class imbalance 的影响。

在 ASVspoof 2019 LA 训练集中，spoofing 样本的数量极大地超过了 bonafide 样本。从而 spoofing 样本在模型训练过程中会对梯度更新产生更大的影响。因此，模型可能会偏向于隔离 spoofing 样本，导致在训练集中过拟合已知攻击。于是提出 Threshold One-class Softmax (TOC-Softmax) loss 来减少 spoofing 样本的优化权重。下面分别描述 Softmax、OC-Softmax 和 TOC-Softmax 的损失函数的区别。


Softmax 定义如下：
$$\begin{aligned}
\mathcal{L}_{S}& =-\frac1N\sum_{i=1}^N\log\frac{e^{w_{y_i}^Tx_i}}{e^{w_{y_i}^Tx_i}+e^{w_{1-y_i}^Tx_i}}  \\
&=\frac1N\sum_{i=1}^N\log(1+e^{(w_{(1-y_i)}-w_{y_i})^Tx_i}),
\end{aligned}$$
其中 $x_i$ 和 $y_i \in \{0,1\}$ 分别为 $i$-th 样本的 embedding feature 和 class label；$0$ 和 $1$ 分别表示 bonafide 类和 spoofing 类；$w_0$ 和 $w_1$ 分别为 bonafide 和 spoofing 类的权重向量；$N$ 为 mini-batch 中样本的数量。本文会计算所有语音样本（包括 bonafide 和 spoofing 样本）的 scores，以确定给定样本是否被分类为 bonafide 类。当使用 Softmax 时，第 $i$ 个样本的 score 定义为 $\frac{e^{w_0^Tx_i}}{e^{w_0^Tx_i}+e^{w_1^Tx_i}}\in[0,1]$。

OC-Softmax loss 定义如下：
$$\begin{aligned}\mathcal{L}_{OCS}&=\frac{1}{N}\sum_{i=1}^{N}\log\left(1+e^{\alpha\left(m_{y_i}-\hat{\boldsymbol{w}}_0^T\hat{\boldsymbol{x}}_i\right)(-1)^{y_i}}\right)\end{aligned}$$
其中 $\alpha$ 是 scale factor；$\hat{w}_0 = \frac{w_0}{\|w_0\|}$ 是 bonafide 类的归一化权重向量；$\hat{x}_i = \frac{x_i}{\|x_i\|}$ 是归一化的 embedding feature；令 $\theta_i$ 为 $w_0$ 和 $x_i$ 之间的夹角，则内积 $w_0^Tx_i \in [-1,1]$ 被定义为第 $i$ 个样本的 classification score，等价于 cosine similarity $\cos \theta_i$。引入两个 margin $m_0$ 和 $m_1$ 来限制角度 $\theta_i$：当 $y_i = 0$ 时，$m_0$ 用于强制 $\theta_i$ 小于 $\arccos m_0$，当 $y_i = 1$ 时，$m_1$ 用于强制 $\theta_i$ 大于 $\arccos m_1$。

如图所示：
![](image/Pasted%20image%2020240127165434.png)
Softmax loss 优化两个权重向量 $w_0$ 和 $w_1$ 来学习 embedding feature space。通过更新权重向量，bonafide 和 spoofing 样本的特征趋向于两个相反的方向，即 $w_0 - w_1$ 和 $w_1 - w_0$。OC-Softmax loss 仅优化一个权重向量 $w_0$ 来学习 embedding feature space。margin $m_0$ 用于压缩 bonafide 样本的特征分布，而 margin $m_1$ 用于将 spoofing 样本与权重向量 $w_0$ 分离。

在 OC-Softmax 的情况下，如果 spoofing 样本的数量过多，那么 bonafide 类的特征分布的可能会被 spoofing 样本破坏。为了解决这个问题，引入 thresholding 过程来移除远离向量 $w_0$ 的 spoofing 样本，提出 TOC-Softmax loss：
$$\begin{aligned}\mathcal{L}_{TOCS}&=\frac{1}{N}\sum_{i=1}^{N}\gamma_{i}\log\left(1+e^{\alpha\left(m_{y_{i}}-\hat{\boldsymbol{w}}_{0}^{T}\hat{\boldsymbol{x}}_{i}\right)(-1)^{y_{i}}}\right)\end{aligned}$$

其中 $\gamma_i = 1 - \mathbb{I}({y_i = 1, \hat{w}_0^T\hat{x}_i < m_1})$ 是样本权重，$\mathbb{I}(\cdot)$ 是 indicator function。如果第 $i$ 个样本的 classification score $w_0^Tx_i$ 小于 $m_1$ 且该样本属于 spoofing 类（$y_i = 1$），则样本权重 $\gamma_i$ 将变为 $0$，这意味着第 $i$ 个样本在训练过程中不参与梯度更新。否则，如果 classification score $w_0^Tx_i$ 大于或等于 $m_1$，或者该样本不属于 spoofing 类，则其权重 $\gamma_i$ 将设置为 $1$。如上图中的 c。对于那些远离 $w_0$ 超过阈值 $m_1$ 的 spoofing 样本（没有箭头），它们不再对网络更新起作用。从而可以在训练阶段优化 bonafide 样本。

> 当 $m_1 = 1$ 时，模型在训练过程中忽略所有 spoofing 样本。这会导致模型无法有效地学习 spoofing 样本的特征，从而难以区分两类样本。

总之，TOC-Softmax 中的 thresholding 过程将网络的重点从将 spoofing 样本从 $w_0$ 推开 转移 到尽可能将 bonafide 类推向 $w_0$。这最终导致 bonafide 类的特征分布更准确。

## 实验结果（略）
