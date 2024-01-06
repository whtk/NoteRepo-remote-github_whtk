> 从频域理解图卷积网络

## 在图上进行计算的挑战

1. 图缺乏一致的结构
2. Node-Order Equivariance：不应该依赖于图中节点的顺序
	> 这个应该这么理解：就是将图中的节点进行重新排列，最后模型（函数）输出的结果不变（或者说，最终的结果也是对应的重排后的序列）
1. Scalability：使用稀疏的方式处理图

## 图网络的用处、数学符号设置

1. 节点分类
2. 图分类
3. 节点聚类
4. 链接预测：预测缺失的连接
5. 识别最具影响力的节点

在图中，每个节点都会映射到固定大小的实值向量中（’embedding‘ 或 ’representation‘），不同的 GNN 的计算方式不同，但都是迭代进行的。

采用 $h_v^{(k)}$ 表示第 $k$ 次迭代后节点 $v$ 的嵌入向量，而每次的迭代都可以看成是网络中的一层。

在图 $G$ 中，节点集合为 $V$，边集合为 $E$，同时节点还有各自的特征（features）作为输入。定义 $x_v$ 为节点 $v$ 的特征。

> 注意：这里的 特征 $x_v$ 是不会随着网络迭代而改变的，例如，对于一张图片来说，特征 可以是某个节点（像素）的RGB像素值。
> 也就是说，特征和嵌入向量不一样

> 下面假设 $G$ 都是无向图

图的性质（properties）可以写成矩阵 $M$，其中每一行 $M_v$ 对应于节点 $v$ 的性质。

## 将卷积网络拓展到图中

CNN 在图像中应用很成功。在图像中，每个像素可以看成图的节点，像素值可以看作节点的特征（features），因此可以将CNN推广到图中。

但是普通的卷积不是 Node-Order Equivariance 的。

## 图中的多项式滤波器

### 图的拉普拉斯矩阵

给定图 $G$，其邻接矩阵定义为 $A$，则度矩阵为：
$$D_v=\sum_u A_{v u} .$$
其中，$A_{vu}$ 代表第 $v$ 行第 $u$ 列，则拉普拉斯矩阵为：
$$L=D-A$$
> 示例：![](./image/Pasted%20image%2020221107202750.png)

### 拉普拉斯多项式

拉普拉斯多项式定义为：
$$p_w(L)=w_0 I_n+w_1 L+w_2 L^2+\ldots+w_d L^d=\sum_{i=0}^d w_i L^i$$
注意，$P_w(L)$ 也是一个 $n\times n$ 的矩阵。而权值 $w_i$ 可以被看成是CNN中的滤波器的权值。

假设节点的特征 $x_v$ 是一维的（即是一个实数），所有的节点的特征组成一个向量 $x \in \mathbb{R}^n$，如图：![](./image/Pasted%20image%2020221107202829.png)
则基于拉普拉斯多项式实现的卷积如下：
$$x^{\prime}=p_w(L) x$$
> 考虑一个特殊情况，$w_1=1$ 其他都为0，则节点 $v$ 的特征定义为$$\begin{aligned}
x_v^{\prime}=(L x)_v &=L_v x \\
&=\sum_{u \in G} L_{v u} x_u \\
&=\sum_{u \in G}\left(D_{v u}-A_{v u}\right) x_u \\
&=D_v x_v-\sum_{u \in \mathcal{N}(v)} x_u
\end{aligned}$$
其实就是 $v$ 的邻近节点的数量乘以自身的特征减去所以邻近节点的特征，光这一项就可以说明，卷积聚合了自身特征和邻节点的特征，这也就是 GNN 中所谓的 信息传递机制。
> 然后通过迭代进行，GNN 的感受野逐步增大。

同时，有以下公式成立：
$$\operatorname{dist}_G(v, u)>i \quad \Longrightarrow \quad L_{v u}^i=0$$
> 就是说，两个节点在图中的距离如果大于 $i$，则拉普拉斯矩阵的 $i$ 次幂在对应的位置的值为 0。

所以有：
$$\begin{aligned}
x_v^{\prime}=\left(p_w(L) x\right)_v &=\left(p_w(L)\right)_v x \\
&=\sum_{i=0}^d w_i L_v^i x \\
&=\sum_{i=0}^d w_i \sum_{u \in G} L_{v u}^i x_u \\
&=\sum_{i=0}^d w_i \sum_{\substack{u \in G \\
\operatorname{dist}_G(v, u) \leq i}} L_{v u}^i x_u .
\end{aligned}$$

> 就是说，节点 $v$ 卷积之后的值只与 关于 $v$ 节点小于距离 $d$ 的邻近有关。

## ChebNet

ChebNet 使用的多项式定义为：
$$p_w(L)=\sum_{i=1}^d w_i T_i(\tilde{L})$$
其中，$T_i$ 是 度为 $i$ 的一阶切比雪夫多项式，$\tilde{L}$ 是归一化拉普拉斯矩阵：
$$\tilde{L}=\frac{2 L}{\lambda_{\max }(L)}-I_n$$
> 1. $L$ 是半正定矩阵，即 $L$ 的所有特征值都大于等于0，如果 $L$ 的最大特征值大于1，则 $L$ 的次幂的值将一直增加，而归一化后的 $\tilde{L}$ 确保其特征值在 $[-1,1]$ 范围内。
> 2. 切比雪夫多项式在进行插值时具有数值稳定性。
。

## 证明多项式滤波器是  Node-Order Equivariant

首先，0-1 正交置换矩阵 $P$ 有如下性质：
$$P P^T=P^T P=I_n$$
如果函数满足，对所有的置换矩阵 $P$ 都有：
$$f(P x)=P f(x)$$
则称函数 $f$ 为 node-order equivariant。

当对节点应用置换矩阵后，有：
$$\begin{aligned}
x & \rightarrow P x \\
L & \rightarrow P L P^T \\
L^i & \rightarrow P L^i P^T
\end{aligned}$$
而：
$$\begin{aligned}
f(P x) &=\sum_{i=0}^d w_i\left(P L^i P^T\right)(P x) \\
&=P \sum_{i=0}^d w_i L^i x \\
&=P f(x) .
\end{aligned}$$
证毕。

## Embedding 计算

通过层叠多个 ChebNet 网络即可进行 Embedding 的迭代计算，首先假设有 $K$ 层网络（$K$ 个不同的多项式滤波器层），第 $k$ 层对应的参数为 $w^{(k)}$ ，则迭代过程如下：
![](./image/Pasted%20image%2020221107210538.png)
图中，$h^{(0)}=x$ 代表初始的节点特征，然后计算多项式滤波器 $p^{(k)}$，最后把滤波器用于下一层 Embedding 的计算（$\sigma()$ 为非线性激活函数）。

## 不同的 GNN 网络（其实就是不同的信息传递机制）

### GCN
![](./image/Pasted%20image%2020221107211243.png)
最终模型的预测输出为：$\hat{y}_v=\operatorname{PREDICT}\left(h_v^{(K)}\right)$ ，其中 $\operatorname{PREDICT}$  通常是另一个和GNN一起训练的神经网络。训练过程中，参数 $W^{(k)} , B^{(k)}$ 在所有节点中共享。这里定义的归一化为：
$$f\left(W \cdot \sum_{u \in \mathcal{N}(v)} \frac{h_u}{|\mathcal{N}(v)|}+B \cdot h_v\right)$$
原论文中，归一化为：
$$f\left(W \cdot \sum_{u \in \mathcal{N}(v)} \frac{h_u}{\sqrt{|\mathcal{N}(u)||\mathcal{N}(v)|}}+B \cdot h_v\right)$$
> 其实，原论文中，用的公式为：
> $$H^{(l+1)}=\sigma\left(\tilde{D}{ }^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)}\right)$$但是这个没有本文用的公式好理解。

### GAT
![](./image/Pasted%20image%2020221107213505.png)
最终模型的预测输出为：$\hat{y}_v=\operatorname{PREDICT}\left(h_v^{(K)}\right)$ ，其中 $\operatorname{PREDICT}$  通常是另一个和GNN一起训练的神经网络。训练过程中，参数 $W^{(k)} , A^{(k)}$ 在所有节点中共享。这里为了简单，只写出了 single-head 的 attention。

> Attention 是输入一排 vectors，输出同样长度的 一排 vectors。


### GraphSAGE

![](./image/Pasted%20image%2020221107213938.png)
最终模型的预测输出为：$\hat{y}_v=\operatorname{PREDICT}\left(h_v^{(K)}\right)$ ，其中 $\operatorname{PREDICT}$  通常是另一个和GNN一起训练的神经网络。训练过程中， $f^{(k)}, AGG, W^{(k)}$ 在所有节点中共享。原始的论文通过以下几种方法来计算 $\underset{u \in \mathcal{N}(v)}{\operatorname{AG} G}\left(\left\{h_u^{(k-1)}\right\}\right)$ （反正最终都是返回一个聚合之后的向量（或矩阵））：
1. 均值（和 GCN 有点类似）：$$W_{\text {pool }}^{(k)} \cdot \frac{h_v^{(k-1)}+\sum_{u \in \mathcal{N}(v)} h_u^{(k-1)}}{1+|\mathcal{N}(v)|}$$
2. Dimension-wise Maximum：$$\max _{u \in \mathcal{N}(v)}\left\{\sigma\left(W_{\text {pool }}^{(k)} h_u^{(k-1)}+b\right)\right\}$$
3. LSTM：把邻节点依次作为LSTM的输入
	> 问题：这样最终的结果不就和选取的邻节点的顺序有关系了吗？（因为 LSTM 的输出和序列的顺序有关啊）

同时在原论文中采用了 “neighbourhood sampling” 邻域采样法，也就是说，无论节点周围的邻节点有几个，都固定数量进行随机采样，这会增加整个 Embedding 的方差，但是就可以用在大型网络上了。


### GIN

![](./image/Pasted%20image%2020221107215304.png)
最终模型的预测输出为：$\hat{y}_v=\operatorname{PREDICT}\left(h_v^{(K)}\right)$ ，其中 $\operatorname{PREDICT}$  通常是另一个和GNN一起训练的神经网络。训练过程中， $f^{(k)}, \epsilon^{(k)}$ 在所有节点中共享。