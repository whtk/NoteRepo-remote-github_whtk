> Google，2018 ICML

1. 提出了一种单层的RNN结构：WaveRNN，使用 dual softmax layer，达到了当时语音生成的sota
2. 能够生成24k，16bit的音频，4× faster than real time
3. 发现，在参数相同时，大的稀疏网络效果比小的密集网络好，提出的 Sparse WaveRNN 可以在手机 CPU 上实时运行
5. 提出基于 subscaling 的高并行度语音生成算法，把长序列的生成折叠成若干个短序列，每个短序列并行生成。

### Introduction
1. 目标是提高采样效率，效率计算为：$$
T(\mathbf{u})=|\mathbf{u}| \sum_{i=1}^{N}\left(c\left(o p_{i}\right)+d\left(o p_{i}\right)\right)
$$其中，$|\mathbf{u}|$ 代表 sample 数量。$N$ 代表网络层数。c(op) 代表每一层的计算时间，d(op)代表硬件执行程序的 overhead 时间。
2. $|\mathbf{u}|$ 或者 $N$ 很大都会导致时间非常大
3. 提出减少上面的每个因子来减少时间
4. 输入是 linguistic feature vectors，输出是24k-16bits语音波形。
5. 采用 RNN 架构来减少 $N$，只需要 $N=5$ 次的 matrix-vector 就可以实现很高的 MOS，而 WaveNet 有两层 30 residual  block，需要 $N=30\times 2=60$ 次的 matrix-vector 乘法。
6. 通过自己实现的 GPU 算法来减少 overhead $d\left(o p_{i}\right)$ 的时间
7. 通过减少神经网络的参数来减少 $c(op_i)$ ，通过将 WaveRNN 中的权重进行稀疏化来实现
8. 最后针对 $|\mathbf{u}|$，提出基于 subscale 的生成过程，把长为 $L$ 的 tensor 分成 $B$ 个长为 $L/B$ 的 tensor，这 $B$ 个 sub-tensor 依次产生，每个都基于前一个 sub-tensor。

## WaveRNN 

###  GRU原理：

主要是为了和 WaveRNN 做比较。
$$
\begin{aligned}
r_{t} &=\sigma\left(x_{t} W_{x r}+h_{t-1} W_{h r}+b_{r}\right) \\
z_{t} &=\sigma\left(x_{t} W_{x z}+h_{t-1} W_{h z}+b_{z}\right) \\
\tilde{h} &=\tanh \left(x_{t} W_{xh}+\left(r_{t} \odot h_{t-1}\right) W_{hh}+b_{hh}\right) \\
h_{t} &=(1-z_{t}) \odot h_{t-1}+z_t \odot \tilde{h}
\end{aligned}
$$
其中 $r$ 代表重置门，$z$ 代表更新门，$\odot$ 代表element-wise product，输入为 $x_t$。


###  WaveRNN 原理

![](image/Pasted%20image%2020230915110504.png)

计算如下：
$$
\begin{aligned}
&\mathbf{x}_{t}=\left[\mathbf{c}_{t-1}, \mathbf{f}_{t-1}, \mathbf{c}_{t}\right] \\
&\mathbf{u}_{t}=\sigma\left(\mathbf{R}_{u} \mathbf{h}_{t-1}+\mathbf{I}_{u}^{\star} \mathbf{x}_{t}\right) \\
&\mathbf{r}_{t}=\sigma\left(\mathbf{R}_{r} \mathbf{h}_{t-1}+\mathbf{I}_{r}^{\star} \mathbf{x}_{t}\right) \\
&\mathbf{e}_{t}=\tau\left(\mathbf{r}_{t} \circ\left(\mathbf{R}_{e} \mathbf{h}_{t-1}\right)+\mathbf{I}_{e}^{\star} \mathbf{x}_{t}\right) \\
&\mathbf{h}_{t}=\mathbf{u}_{t} \circ \mathbf{h}_{t-1}+\left(1-\mathbf{u}_{t}\right) \circ \mathbf{e}_{t} \\
&\mathbf{y}_{c}, \mathbf{y}_{f}=\operatorname{split}\left(\mathbf{h}_{t}\right) \\
&P\left(\mathbf{c}_{t}\right)=\operatorname{softmax}\left(\mathbf{O}_{2} \operatorname{relu}\left(\mathbf{O}_{1} \mathbf{y}_{c}\right)\right) \\
&P\left(\mathbf{f}_{t}\right)=\operatorname{softmax}\left(\mathbf{O}_{4} \operatorname{relu}\left(\mathbf{O}_{3} \mathbf{y}_{f}\right)\right)
\end{aligned}
$$
式中，$\mathbf{u}_{t}$ 相当于GRU中的 $z_t$，$\mathbf{e}_t$ 相当于 $\tilde{{h}}$，带 $\star$ 的表示掩膜矩阵，$\operatorname{split}$ 就是简单的分离，通过分离可以利用两个softmax（dual softmax）来实现输出bit构建，将输出维度从 $2^{16}$ 减少到 $2 \times 2^8$，$\mathbf{R}、\mathbf{I}、\mathbf{O}$ 都代表可学习的权值矩阵。

具体的生成过程可能是，首先初始化参数，在第 $t$ 个 time step，根据 $\mathbf{f}_{t-1}、\mathbf{c}_{t-1}$，进行一次GRU运算生成 $\mathbf{c}_{t}$，然后根据 $\mathbf{f}_{t-1}、\mathbf{c}_{t-1}、\mathbf{c}_{t}$ 生成 $\mathbf{f}_{t}$ 。

一个更详细的图：
$\mathbf{c}_{t}$ 的生成：
![](image/Pasted%20image%2020230915152844.png)
$\mathbf{f}_{t}$ 的生成：
![](image/Pasted%20image%2020230915152920.png)

## 稀疏 WaveRNN

### 权值稀疏化
1. 基于权值的裁剪，随着训练的进行，稀疏性逐渐增加。
2. 具体实现：每500个steps，对权值进行排序，将最小的k个权值置0，k 是总权值数的一个比例，随着训练的进行，权值数增加，k也随之增加，稀疏性也增加。
3. $k$ 随着 step 的变换如下：$$z=Z\left(1-\left(1-\frac{t-t_0}S\right)^3\right)$$
4. 是对GRU中的三个 gate 中的 权值矩阵进行稀疏化

###  结构稀疏化
1. 目的：减少内存 overhead
2. 首先将权值矩阵分成多个块，结构稀疏化就是将每个分块的权值要么保留要么全部清零
3. 发现选 $m=16(16\times 1)$ 的权值块几乎没有性能损失，并且可以减少内存需求到 $\frac1m$
4. $4\times 4$ 的权值块效果也还可以

## Subscale WaveRNN

一次产生 batch 为 $B$ 的样本（一次产生 $B$ 个 batch），总时间变为：
$$
T(\mathbf{u})=\frac{|\mathbf{u}|}{B} \sum_{i=1}^{N}\left(c\left(o p_{i}^{B}\right)+d\left(o p_{i}^{B}\right)\right)
$$
可以减少 $c\left(o p_{i}\right)$ 的时间，因为权值可以重用。

之前的工作中，在每个 step 同时生成多个样本，但是打破了样本之间的 local dependency：也就是两个邻近的样本本来是有 很强的相关性，但是是独立生成的。

### Subscale Dependency 的序列生成：

首先生成第一个 sub-tensor，然后第二个 sub-tensor 基于第一个，然后第三个 sub-tensor 基于前两个。

> 考虑一个长为128的序列，首先按照第一点，生成 $B=8$ 个sub scale 的序列，每个子序列长为 $16$，同时定义前瞻变量 $F=3$，则（括号内表示同步生成）首先逐次生成序号为 $1, 9, 17, 25$ 的 序列，然后再逐次生成 $33(2), 41(10), 49(18), 57(26)$ 的序列，再逐次生成 $65(34, 3), 73(42, 11), 81(50, 19), 89(58, 27)$ 的序列，再逐次生成 $97(66, 35, 4), 105(74, 43, 12), 113(82, 51, 20), 121(90, 59, 28)$ 的序列 $\cdots$，依次进行，在序列足够长时（$16$足够大），每次运行相当于并行生成了 $B$ 个序列。

> 上面的解释好像不对。。。没看懂