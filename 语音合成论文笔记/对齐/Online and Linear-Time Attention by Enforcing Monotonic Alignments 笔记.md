> Google，ICML，2017

1. 基于 attention 的 RNN 在很多 seq2seq 问题中都很有效，但是，soft attention 是在整个序列中进行的，从而无法在 online 的条件下使用，且时间复杂度是三次的
2. 由于在很多任务中，输入序列和输出序列是 monotonic 的，于是提出一个端到端的、可微分的方法来学习 monotonic alignments，从而在测试时可以 online 地计算 attention，且复杂度是线性的。
3. 在多个任务 sentence summarization, machine translation, and online speech recognition problems 中实验，实现了 competitive 的性能

## Introduction

1. soft attention 的一个问题是，模型在生成输出序列的每个元素时，必须知道整个输入序列
2. 本文的目的是，提出一种 attention 机制，时间复杂度为线性，且可以 online 实现：
	1. 首先注意到，很多问题的输入输出 的 alignment 都是 monotonic 
	2. 于是采用 hard monotonic alignments，实验表明，相比于 softmax-based attention，只有一点点的性能下降

## Online and Linear-Time Attention

### soft attention

具体见[Neural Machine Translation by Jointly Learning to Align and Translate 笔记](Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate%20笔记.md)，模型包含两个 RNN，encoder RNN 输入为 $\mathbf{x}=\left\{x_1, \ldots, x_T\right\}$，产生一系列的 hidden state $\mathbf{h}=\left\{h_1, \ldots, h_T\right\}$，这里的 $\mathbf{h}$ 称为 memory，decoder RNN 基于 memory 产生输出序列 $\mathbf{y}=\left\{y_1, \ldots, y_U\right\}$。

计算 $y_i$ 时，soft attention-based decoder 采用 learnable nonlinear function $a$ 产生标量 $e_{i,j}$，通常 $a$ 时一个单层神经网络+tanh激活。标量通过 softmax 产生概率分布，然后对 $\mathbf{h}$ 进行加权产生 context vector $c_i$，整个计算过程如下：
$$\begin{aligned}
e_{i,j}& =a(s_{i-1},h_j)  \\
\alpha_{i,j}& =\exp(e_{i,j})\left/\sum_{k=1}^T\exp(e_{i,k})\right.  \\
{c_i}& =\sum_{j=1}^T\alpha_{i,j}h_j  \\
{s_i}& =f(s_{i-1},y_{i-1},c_i)  \\
{y_i}& =g(s_i,c_i) 
\end{aligned}$$
观察上面的第二和第三项，其实是一个简单的随机过程的期望的计算，可以看成：
+ 首先，对于每个 memory 中的 $h_j$，概率 $\alpha_{i,j}$ 是独立计算的
+ 然后通过从分布 $k \sim \text { Categorical }\left(\alpha_i\right)$ 中进行采样得到 index $k$，然后令 $c_i$ 为 $h_k$
+ 在这种视角下，第三项可以看成是，用 soft attention 替代采样过程，此时计算的就是 $c_i$ 的期望了

如图：
![](image/Pasted%20image%2020230904165829.png)

### Hard Monotonic Attention 

可以发现，attention 概率的计算其实和 time step 的顺序无关。

对于给定的 time step $i$，从 memory index $t_{i-1}$ 开始处理，$t_{i}$ 为输出的 time step $i$ 选择的 memory 项（令 $t_0=1$），对于给定的 $j=t_{i-1}, t_{i-1}+1, t_{i-1}+2$，可以顺序地计算：
$$\begin{aligned}
&e_{i,j} =a(s_{i-1},h_j)  \\
&p_{i,j} =\sigma(e_{i,j})  \\
&z_{i,j} \sim\operatorname{Bernoulli}(p_{i,j}) 
\end{aligned}$$
其中，$a$ 为 energy function，$\sigma$ 为 sigmoid 函数，对于某个 $j$，如果采样到 $z_{i,j}=1$，则停止循环然后另 $c_i=h_j,t_i=j$，即认为 $h_j$ 为 contex context vector，这里的 $z_{i,j}$ 就可以看成是一种概率选择，要么从 memory 中添加一个新的项（为 0），要么结束得到输出（为 1）。对于接下来的所有的输出 time step，重复此过程，而且每次都从 $t_{i-1}$ 开始。
> 如果一直到 time step $T$，$z_{i,j}=0$，则直接认为 $c_i$ 为全零向量

> 简单来说，就是从输出的第一个 time step 开始，计算他和 memory 中的第一个 $h$ ，得到一个概率 $p$，以此概率作为二项分布的参数进行采样
> 如果采样到 1，则停止，然后把当前这个时刻对应的 $h$ 作为 context vector，同时记录 此时刻，以作为输出的下一个 time step 对 memory 中的项计算概率的初始时刻。
> 如果采样到 0，则继续看 memory 中的下一个 $h$，同样计算得到一个新的 $p$，同样进行采样，重复此过程直到采样到一个 1

如图：
![](image/Pasted%20image%2020230904171102.png)

$i$ 是输出的 index，$j$ 是 memory 的 index，对于某个输出的 index $i$，只需要和 $k\in\{1,\ldots,j\}$ 的 $h_k$ 计算得到 $p_{i,j}$ ，$j$ 后面的则没有计算，从而可以以一种 online 的方式计算。

并且，由于下一个 time step 的起点是上一个 time step 的终点，从而计算 $p_{i,j}$ 的总数是 $\max(T,U)$，也就是线性的时间复杂度。

并且还可以发现，输入输出序列之间的 alignment 是严格 monotonic 的！

###  Training in Expectation

由于上面的过程涉及到采样，所以无法进行反向传播。于是提出  training with respect to the expected value of $c_i$，计算如下：

首先根据前面的描述直接计算 $e_{i,j},p_{i,j}$，此时 attention 的分布为：$$\begin{aligned}
\alpha_{i,j}& =p_{i,j}\sum_{k=1}^{j}\left(\alpha_{i-1,k}\prod_{l=k}^{j-1}(1-p_{i,l})\right)  \\
&=p_{i,j}\left((1-p_{i,j-1})\frac{\alpha_{i,j-1}}{p_{i,j-1}}+\alpha_{i-1,j}\right)
\end{aligned}$$
定义 $q_{i,j}=\alpha_{i,j}/p_{i,j}$，从而：
$$\begin{aligned}
&e_{i,j} =a(s_{i-1},h_j)  \\
&p_{i,j} =\sigma(e_{i,j})  \\
&q_{i,j} =(1-p_{i,j-1})q_{i,j-1}+\alpha_{i-1,j}  \\
&\alpha_{i,j} =p_{i,j}q_{i,j} 
\end{aligned}$$
为了使模型在训练和测试时表现相似，需要有 $p_{i,j}\approx0$ 或者 $p_{i,j}\approx1$。 

### 改进的 energy function（略）

### Encouraging Discreteness

为了实现前面 $p_{i,j}$ 接近 0 和 1，一个直接的方法是在 sigmoid 函数中添加噪声（有点类似于 Gumbel-Softmax 技巧），但是在测试的时候不会加噪声。
