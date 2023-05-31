<!--
 * @Author: error: git config user.name && git config user.email & please set dead value or install git
 * @Date: 2022-06-23 15:38:49
 * @LastEditors: error: git config user.name && git config user.email & please set dead value or install git
 * @LastEditTime: 2022-06-24 17:33:52
 * @FilePath: \PA\wavernn.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
## Efficient Neural Audio Synthesis 笔记（2018年2月提出）
1. 提出了一种单层的RNN结构：WaveRNN，使用双softmax层，达到了当时语音生成的sota。
2. 能够生成24k，16bit的pcm音频
3. 采用权值裁剪技术，实现大型稀疏性网络，稀疏性达到了96%
4. 提出高并行度语音生成算法，把长序列的生成折叠成若干个短序列，每个短序列并行生成。
5. 第一个可在各类计算平台上进行实时语音合成的神经网络模型

   
### Introduction
1. 语音生成效率：
$$
T(\mathbf{u})=|\mathbf{u}| \sum_{i=1}^{N}\left(c\left(o p_{i}\right)+d\left(o p_{i}\right)\right)
$$
其中，$|\mathbf{u}|$ 代表 sample 数量。$N$ 代表网络层数。c(op) 代表每一层的计算时间，d(op)代表硬件执行程序的overhead 时间。
2. 输入是linguistic feature vectors，输出是24k-16bits语音波形。

### 原理

#### 数学公式
1. GRU原理：
$$
\begin{aligned}
r_{t} &=\sigma\left(x_{t} W_{x r}+h_{t-1} W_{h r}+b_{r}\right) \\
z_{t} &=\sigma\left(x_{t} W_{x z}+h_{t-1} W_{h z}+b_{z}\right) \\
\tilde{h} &=\tanh \left(x_{t} W_{xh}+\left(r_{t} \odot h_{t-1}\right) W_{hh}+b_{hh}\right) \\
h_{t} &=(1-z_{t}) \odot h_{t-1}+z_t \odot \tilde{h}
\end{aligned}
$$
其中 $r$ 代表重置门，$z$ 代表更新门，$\odot$ 代表element-wise product。

2. WaveRNN 数学原理：
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
式中，$\mathbf{u}_{t}$ 相当于GRU中的 $z_t$，$\mathbf{e}_t$ 相当于 $\tilde{{h}}$，带 $\star$ 的表示掩膜矩阵，$\operatorname{split}$ 就是简单的分离，通过分离可以利用两个softmax（dual softmax）来实现输出bit构建，将输出维度从 $2^{16}$ 减少到 $2 \times 2^8$，$\mathbf{R}、\mathbf{I}、\mathbf{O}$ 都代表待学习的权值矩阵。

具体的生成过程可能是，首先初始化参数，在第 $t$ 个 time step，根据 $\mathbf{f}_{t-1}、\mathbf{c}_{t-1}$，进行一次GRU运算生成 $\mathbf{c}_{t}$，然后根据 $\mathbf{f}_{t-1}、\mathbf{c}_{t-1}、\mathbf{c}_{t}$ 生成 $\mathbf{f}_{t}$ 。

### 稀疏WaveRNN

#### 权值稀疏化
1. 基于权值的裁剪，随着训练的进行，稀疏性逐渐增加。
2. 具体实现：每500个steps，对权值进行排序，将最小的k个权值置0，k 是总权值数的一个比例，随着训练的进行，权值数增加，k也随之增加，稀疏性也增加。
3. 对GRU中的三个门分别进行稀疏化。

#### 结构稀疏化
1. 目的：减少内存overhead
2. 结构稀疏化就是将整个分块的权值要么保留要么全部清零
3. m=16(16*1) 的权值块几乎没有性能损失，并且可以减少内存需求到 $\frac1m$
4. 4*4 的权值块效果也还可以

### Subscale WaveRNN
1. 一次产生batch为 $B$ 的样本，总时间变为：
$$
T(\mathbf{u})=\frac{|\mathbf{u}|}{B} \sum_{i=1}^{N}\left(c\left(o p_{i}^{B}\right)+d\left(o p_{i}^{B}\right)\right)
$$
可以减少 $c\left(o p_{i}\right)$ 的时间，因为权值可以重用。
2. Subscale Dependency 的序列生成：
    考虑一个长为128的序列，首先按照第一点，生成 $B=8$ 个sub scale 的序列，每个子序列长为 $16$，同时定义前瞻变量 $F=3$，则（括号内表示同步生成）首先逐次生成序号为 $1, 9, 17, 25$ 的 序列，然后再逐次生成 $33(2), 41(10), 49(18), 57(26)$ 的序列，再逐次生成 $65(34, 3), 73(42, 11), 81(50, 19), 89(58, 27)$ 的序列，再逐次生成 $97(66, 35, 4), 105(74, 43, 12), 113(82, 51, 20), 121(90, 59, 28)$ 的序列 $\cdots$，依次进行，在序列足够长时（$16$足够大），每次运行相当于并行生成了 $B$ 个序列。
