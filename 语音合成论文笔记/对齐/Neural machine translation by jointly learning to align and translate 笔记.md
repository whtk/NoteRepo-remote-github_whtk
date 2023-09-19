> Jacobs University、Universite de Montreal，Bengio，ICLR 2015

1. 最近的 NMT 采用 encoder-decoder 结构，先将 source sentence encode 到一个固定的向量，然后 decode 得到翻译结果
2. 作者认为，固定长度的向量是一个 bottleneck，于是提出，允许模型自动搜索 source sentence 中与预测目标词相关的部分，而无需 显式地将这些部分形成  hard segmen

## Introduction

1. 提出方法的特点是，不是将 input sentence encode 到 single fixed-length vector，而是 encode 到一系列的 vectors，然后在 decode 的时候自适应地选择这些 vectors 的一部分子集

## 背景：NMT

给定源序列 $\mathbf{x}$，找到能够最大化条件概率的序列 $\mathbf{y}$，即 $\arg \max _{\mathbf{y}} p(\mathbf{y} \mid \mathbf{x})$。

经典的 RNN encoder-decoder 结构如下：

encoder 读取输入序列 $\mathbf{x}=\left(x_1, \cdots, x_{T_x}\right)$，生成输出序列 $c$，即：
$$\begin{aligned}h_t&=f\left(x_t,h_{t-1}\right)\\\\c&=q\left(\{h_1,\cdots,h_{T_x}\}\right),\end{aligned}$$
其中的  $f,q$ 为非线性函数，如 seq2seq 用的 $f$ 是 LSTM，$q\left(\left\{h_1, \cdots, h_T\right\}\right)=h_T$。

decoder 给定 $c$ 和前面所有的预测值 $\left\{y_1, \cdots, y_{t^{\prime}-1}\right\}$ 来预测下一个单词 $y_{t^{\prime}}$，也就是定义了一个关于所有预测的联合概率：
$$p(\mathbf{y})=\prod_{t=1}^Tp(y_t\mid\left\{y_1,\cdots,y_{t-1}\right\},c),$$
其中，$\mathbf{y}=\left(y_1, \cdots, y_{T_y}\right)$，如果用 RNN 模型，其中的条件概率可以建模为：
$$p(y_t\mid\{y_1,\cdots,y_{t-1}\},c)=g(y_{t-1},s_t,c)$$
其中的 $g$ 是非线性函数，用于输出 $y_t$ 的概率。

## LEARNING TO ALIGN AND TRANSLATE

整个架构如图：
![](image/Pasted%20image%2020230904155717.png)

### decoder

提出的 decoder 定义的条件概率为：
$$p(y_i|y_1,\ldots,y_{i-1},\mathbf{x})=g(y_{i-1},s_i,c_i)$$
其中 $s_i$ 是 RNN 的 hidden state，计算为：
$$s_i=f(s_{i-1},y_{i-1},c_i)$$
注意，这里的概率是基于一个有下标的 context vector $c_i$。

而 $c_i$ 和 annotations 序列 $(h_{1},\cdots,h_{T_{x}})$ 相关，每个 annotation $h_i$ 都包含了 information about the whole input sequence with a strong focus on the parts surrounding the i-th word of the input sequence。

具体来说，$c_i$ 计算为 annotation $h_i$ 的加权和：
$$c_i=\sum_{j=1}^{T_x}\alpha_{ij}h_j$$
而其中的权重系数计算为：
$$\begin{aligned}\alpha_{ij}&=\frac{\exp\left(e_{ij}\right)}{\sum_{k=1}^{T_x}\exp\left(e_{ik}\right)},\\\\e_{ij}&=a(s_{i-1},h_j)\end{aligned}$$
这里的 $a$ 称为 alignment model，用于表示输入中的第 $j$ 个和输出中的第 $i$ 个的匹配程度。本质就是一个 feedforward neural network。

### encoder

采用双向 RNN，将两个方向的 $h_j$ 进行拼接 $h_j=\left[\overrightarrow{h}_j^\top;\overleftarrow{h}_j^\top\right]^\top$。