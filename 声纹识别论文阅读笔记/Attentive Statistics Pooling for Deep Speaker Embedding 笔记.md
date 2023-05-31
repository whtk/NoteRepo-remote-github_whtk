> interspeech 2018

1. 提出 Attentive Statistics Pooling，用于文本无关的 ASV
2. 利用注意力机制为不同的帧赋予不同的权重，从而生成加权的均值和标准差

## Introduction

1. 在可变长度的语音中，引入 average pooling 来聚合 frame level 的特征从而得到固定维度的 utterance level 的特征
2. 有作者提出使用 statistics pooling 计算均值和标准差，但是并没有说明 标准差 对性能的提升的影响
3. 本文提出 attentive statistics pooling，通过注意力机制计算权值，来对 frame level 聚合生成 均值和标准差 的过程进行加权，通过实验验证了有效性

## 基于 DNN 的说话人 embedding

![](./image/Pasted%20image%2020230213170938.png)
如上图，第一部分是 frame-level 特征提取，这一部分的输入是声学特征序列（如 MFCC），输出为 frame level 的特征，内部的模型可以用 TDNN、CNN、LSTM 等等。

第二部分是pooling 层，将长度可变的 frame level 特征转换成固定维度的向量。一般通过求平均实现。

第三部分为 utterance level 的特征提取，堆叠多个 FC 层，最后一层是 softmax，节点数为 说话人数量。

使用 Cross Entropy loss 进行训练（其他如 contrastive loss、triplet loss 也有）。

## 带有注意力机制的高阶 pooling
> pooling 有两个拓展，使用 高阶统计量、使用注意力机制

### 统计池化

statistics pooling 层基于 frame level 的特征 $\boldsymbol{h}_t(t=1, \cdots, T)$ 计算均值和标准差：$$\begin{gathered}
\boldsymbol{\mu}=\frac{1}{T} \sum_t^T \boldsymbol{h}_t \\
\boldsymbol{\sigma}=\sqrt{\frac{1}{T} \sum_t^T \boldsymbol{h}_t \odot \boldsymbol{h}_t-\boldsymbol{\mu} \odot \boldsymbol{\mu}}
\end{gathered}$$
其中，$\odot$ 为 element-wise 乘法，均值可以看成 utterance-level 特征的主体，但是这个 标准差 也有作用，因为就长时上下文来说，其包含了 说话人的其他特征（有点类似于 LSTM，但是 标准差 对于任何上下文距离都有效，而 LSTM 会导致梯度消失的问题）。
> 要这么说，也有点类似于 attention 了，因为计算 标准差的时候看到了元素整个向量之间的关系

### 注意力机制

attention 模型为每个 frame level 的特征计算一个标量的 score $e_t$：$$e_t=\boldsymbol{v}^T f\left(\boldsymbol{W} \boldsymbol{h}_t+\boldsymbol{b}\right)+k$$
其中，$f()$ 为非线性激活，score 在所有的 frame 维度（也是就时间维度）通过 softmax 进行归一化，从而加起来为 1：$$\alpha_t=\frac{\exp \left(e_t\right)}{\sum_\tau^T \exp \left(e_\tau\right)}$$
归一化之后的 score 用于计算加权的均值：$$\tilde{\boldsymbol{\mu}}=\sum_t^T \alpha_t \boldsymbol{h}_t$$

### Attentive statistics pooling
![](./image/Pasted%20image%2020230213174704.png)
把上面的两个方法结合一下，则加权的均值和上面一样，加权的标准差计算为：$$\tilde{\boldsymbol{\sigma}}=\sqrt{\sum_t^T \alpha_t \boldsymbol{h}_t \odot \boldsymbol{h}_t-\tilde{\boldsymbol{\mu}} \odot \tilde{\boldsymbol{\mu}}}$$
且这个计算过程是可微的，从而可以进行 BP。

## 实验

在 voxceleb 上的结果：![](./image/Pasted%20image%2020230213174751.png)

在 NIST SRE 中，不同音频长度的结果：![](./image/Pasted%20image%2020230213174918.png)
可以看到，i-vector 在长音频段的效果还是很强的。