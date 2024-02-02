> APSIPA ASC 2023，北邮

1. 大部分用 one-class 做虚假语音检测的模型很难泛化到未知攻击
2. 提出 IOC-Softmax，通过将 OC-Softmax 的 scale factor 分解为两部分，将真实样本的所贡献的 loss 的权重降低
3. 对于不同的任务可以选择不同的 scale factor，防止训练时真实样本的数量过多
4. 在 ASVspoof 2019 数据集，效果比 OC-Softmax 好

## Introduction

1. 大部分的 anti-spoofing 都是二分类问题，假设训练集和测试集的数据分布相似，但实际不是
2. one-class learning 可以提高泛化能力，[One-class Learning Towards Synthetic Voice Spoofing Detection 笔记](One-class%20Learning%20Towards%20Synthetic%20Voice%20Spoofing%20Detection%20笔记.md) 和 [A Deep One-Class Learning Method for Replay Attack Detection 笔记](A%20Deep%20One-Class%20Learning%20Method%20for%20Replay%20Attack%20Detection%20笔记.md) 分别提出了 OC-Softmax 和 Deep One-class Learning (DOL) 来做虚假语音检测
3. OC-Softmax 仍然有两个问题：
    1. 泛化能力不够
    2. 对不同任务的灵活性不够
4. 发现 OC-Softmax 中的 scale factor 是问题的根源，导致真实样本在训练时占据了很大的权重，而且真实样本的 loss 很难进一步降低，因此模型只能通过让虚假样本的 loss 增加来降低总的 loss，但是这样会导致模型泛化能力差
5. 提出 IOC-Softmax，将 OC-Softmax 的 scale factor 分解为两部分，一部分用于真实样本，一部分用于虚假样本，通过调整真实样本和虚假样本的 loss 来提高模型的泛化能力和灵活性
6. 贡献如下：
    1. 分析 scale factor 的影响
    2. 提出 IOC-Softmax，通过将 scale factor 分解为两部分
    3. 在 ASVspoof 2019 数据集上做了实验，验证了 IOC-Softmax 的优势

## 方法

### One-class learning 损失函数

原始的 one-class learning 损失函数定义为：
$$\begin{aligned}L_{OC}&=\frac{1}{M}\sum_{i=1}^{M}\log(1+e^{s(m_{y_i}-\hat{w}_0\cdot\hat{x}_i)(-1)^{y_i}}),\end{aligned}$$
其中 $M$ 是 batch size，$s$ 是 scale factor，$\hat{w}_0$ 是归一化的权重向量，表示真实样本的优化方向，$\hat{x}_i \in \mathbb{R}^D$ 是归一化的特征向量，$D$ 是特征向量的维度，本文取 $D=256$，$y_i$ 是标签，$m_0$ 和 $m_1$ 是两个 margin，用于限制 $w_0$ 和 $x_i$ 之间的夹角 $\alpha_i$。$m_0$ 用于让 $\alpha_i$ 小于 $\arccos m_0$，而 $m_1$ 用于让 $\alpha_i$ 大于 $\arccos m_1$。当输入是真实语音时（$y_i = 0$），$m_0$ 用于让 $\alpha_i$ 小于 $\arccos m_0$，而较小的 $\arccos m_0$ 可以使真实样本更加紧凑地聚集在 $w_0$ 周围。当 $y_i = 1$ 时，$m_1$ 用于强制 $\alpha_i$ 大于 $\arccos m_1$，而相对较大的 $\arccos m_1$ 可以使虚假样本远离 $w_0$。

### Improved one-class learning

下图表中给出了四种情况：
![](image/Pasted%20image%2020240131205846.png)
其中，$\hat{w}_0\hat{x}_i$ 表示 center vector $\hat{w}_0$ 和 speech samples $\hat{x}_i$ 的相似度。图中 a 是 OC-Softmax，B1、B2、B3 和 B4 是图 b 和 c 中讨论的四种情况。图 b 描述了 OC-Softmax（$m_0 = 0.9$，$m_1 = 0.2$）在四种情况下随 scale factor $s$ 变化的情况。这四种情况的 loss 函数分别是 $\mathcal{L}_{OC-B1}$、$\mathcal{L}_{OC-B2}$、$\mathcal{L}_{OC-B3}$ 和 $\mathcal{L}_{OC-B4}$。发现：
+ 与 $\mathcal{L}_{OC-B2}$ 和 $\mathcal{L}_{OC-B3}$ 相比，$\mathcal{L}_{OC-B1}$ 和 $\mathcal{L}_{OC-B4}$ 的 loss 值要小得多，因此真实样本会收敛到 B1，而虚假样本会收敛到 B4
+ 对于 B1 和 B4，当真实样本和虚假样本共享相同的 scale factor 时，这两种样本类型贡献的 loss 值要么几乎为零，要么差异很大。前者使得真实语音和虚假语音的嵌入空间更加紧凑，导致模型的泛化能力差。对于后者，以 OC-Softmax 中的 $s = 20$ 为例，真实样本贡献的 loss 值为 0.127，而虚假样本贡献的 loss 值为 $3.775\times 10^{-11}$，差异很大，导致真实样本在训练时占据了很大的权重
+ 图 c 为当真实样本和虚假样本被正确分类时，OC-Softmax 的 loss 范围，其中 $\gamma = s(m_{y_i} - \hat{w}_0\hat{x}_i)(-1)^{y_i}$。对于真实样本，$\gamma$ 的范围是 $[-2, 0]$，对应的 loss 值的范围是 $[0.127, 0.639]$。对于虚假样本，$\gamma$ 的范围是 $[-24, 0]$，对应的 loss 值的范围是 $[3.775\times 10^{-11}, 0.639]$。当真实语音收敛到 B1 时，其 loss 值达到最小值，即 0.127。为了进一步降低总的 loss 值，模型在训练过程中只能将虚假语音压缩到 B4。这导致了紧凑的虚假嵌入空间，从而导致反欺骗模型的泛化能力差。

而且真实语音和虚假语音用相同的 scale factor 会限制类别的平衡。当合成样本是由不同的 TTS 和 VC 技术产生的，会导致测试数据和训练数据的分布不同。

于是提出 IOC-Softmax，定义如下：
$$\begin{aligned}L_{IOC}&=\frac{1}{M}\sum_{i=1}^{M}\log(1+e^{s_{y_i}(m_{y_i}-\hat{w}_0\cdot\hat{x}_i)(-1)^{y_i}}),\end{aligned}$$

其中 $s_{y_i}$ 是 scale factor，包括两部分，即 $s_0$ 用于真实样本，$s_1$ 用于虚假样本。用来控制语音样本贡献的 loss 值。

## 实验（略）

> 所以重点在于 scale factor 的选择？