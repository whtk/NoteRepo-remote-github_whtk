
> 为什么需要位置编码：attention 模块捕获输入序列的位置信息，即无法区分不同位置的 token

### 绝对位置编码

绝对位置编码会加入到输入序列中：在输入的第 $k$ 个向量 $x_k$ 中加入位置向量 $p_k$ 得到 $x_k+p_k$，且 $p_k$ 只依赖于位置的编号 $k$。

### 训练式

直接将位置编码当作可训练参数。
> 比如最大长度为512，编码维度为768，那么就初始化一个 512×768 的矩阵作为位置向量，让它随着训练过程更新。

优缺点：没有外推性。

### 三角式

即所谓的 Sinusoidal 位置编码，来自 transformer：
$$\begin{cases}p_{k,2i}=\sin\left(k/10000^{2i/d}\right)\\p_{k,2i+1}=\cos\left(k/10000^{2i/d}\right)&\end{cases}$$
其中 $p_{k,2i}, p_{k,2i+1}$ 分别是位置 $k$ 的第 $2i, 2i+1$ 维，$d$ 是向量的维度。

优缺点：有外推性。

### 递归式

由于 RNN 模型不需要位置编码，因为其本身的结构就自带了位置信息。因此在输入后接一层 RNN 再接 transformer，理论上就可以不用再加位置编码了。

优缺点：有一定的外推性，但是牺牲了并行性。

### 相乘式

前面说的位置编码都是相加的，似乎可以考虑相乘的？即 $x_k\otimes p_k$。
> 甚至也可以考虑拼接的做法。

## 相对位置编码（略）

## 旋转式位置编码（RoPE）

> 出发点：通过绝对位置编码的方式实现相对位置编码

假设通过下述操作给 $q,k$ 添加绝对位置信息：
$$\tilde{q}_m=f(q,m),\quad\tilde{k}_n=f(k,n)$$
即分别为 $q,k$ 设计函数 $f(\cdot,m),f(\cdot,n)$，其中 $m,n$ 为绝对位置。经过此操作之后，$\tilde{q}_m,\tilde{k}_n$ 就带有了位置 $m,n$ 的绝对位置信息。Attention 核心计算为内积，所以希望内积的结果带有相对位置信息，于是假设存在恒等关系：
$$\langle f(q,m),f(k,n)\rangle=g(q,k,m-n)$$
目标是，求出此恒等式的一个尽可能简单的解。求解时，假定下面的初值条件 $f(q,0)=q$，即位置 0 的绝对位置信息就是原始的 $q$，同理 $f(k,0)=k$。

对于一个复数，有 $\langle{q},{k}\rangle=\mathrm{Re}[{q}{k}^*]$（注意：这里的 $q,k$ 都必须为实数才成立），从而有：
$$\mathrm{Re}[f(q,m)f^*(k,n)]=g(q,k,m-n)$$
假设存在复数 $g(q,k,m-n)$ 使得 $[f(q,m)f^*(k,n)]=g(q,k,m-n)$，然后设：
$$
\begin{aligned}
{f(q,m)}& =R_f(q,m)e^{\mathrm{i}\Theta_f(q,m)}  \\
{f(k,n)}& =R_f(k,n)e^{\mathrm{i}\Theta_f(k,n)}  \\
{g(q,k,m-n)}& =R_g(q,k,m-n)e^{\mathrm{i}\Theta_g(\boldsymbol{q},\boldsymbol{k},m-n)} 
\end{aligned}
$$
带入得到：
$$\begin{aligned}{R_f(q,m)R_f(k,n)}&={R_g(q,k,m-n)}\\{\Theta_f(q,m)-\Theta_f(k,n)}&={\Theta_g(q,k,m-n)}\end{aligned}$$
对于第一个方程，带入 $m=n$ 得到：
$$R_f(q,m)R_f(k,m)=R_g(q,k,0)=R_f(q,0)R_f(k,0)=\|q\|\|k\|$$
假设 $R_f(q,m)=\|q\|,R_f(k,m)=\|k\|$，即他们不依赖于 $m$。

同样对于第二个方程，带入 $m=n$ 有：
$$\Theta_f(q,m)-\Theta_f(k,m)=\Theta_g(q,k,0)=\Theta_f(q,0)-\Theta_f(k,0)=\Theta(q)-\Theta(k)$$
从而 $\Theta_f(q,m)-\Theta(q)=\Theta_f(k,m)-\Theta(k)$，即 $\Theta_f(q,m)-\Theta(q)$ 是一个只和 $m$ 相关而和 $q$ 无关的函数，记为 ${\varphi(m)}$，从而 $\Theta_f(q,m)=\Theta(q)+\varphi(m)$，进一步带入 $n=m-1$，有：
$$\varphi(m)-\varphi(m-1)=\Theta_g(q,k,1)+\Theta(k)-\Theta(q)$$
这说明 $\varphi(m)$ 为等差数列，设右端为 $\theta$，得到 $\varphi(m)=m\theta$。

从而有：
$f(q,m)=R_f(q,m)e^{\mathrm{i}\Theta_f({q},m)}=\|q\|e^{\mathrm{i}(\Theta({q})+m\theta)}=\boldsymbol{q}e^{\mathrm{i}m\theta}$
对于二维形式，改变换其实对应向量的旋转，称之为旋转式位置编码，写成矩阵形式为：
$$\left.f(q,m)=\left(\begin{array}{rr}\cos m\theta&-\sin m\theta\\\sin m\theta&\cos m\theta\end{array}\right.\right)\left(\begin{array}{r}q_0\\q_1\end{array}\right)$$
由于内积满足线性叠加性，对于任意偶数维度的 RoPE，都可以表示为二维情况的拼接：
![](image/Pasted%20image%2020240325195624.png)
即，将位置为 $m$ 的向量 $q$ 乘上矩阵 $\mathcal{R}_{m}$、位置为 $n$ 的向量 $k$ 乘上矩阵 $\mathcal{R}_{n}$，用变换后的 $Q,K$ 进行 attention，此时的 attention 就会包含相对位置信息，即下式成立：
$$(\mathcal{R}_mq)^\top(\mathcal{R}_nk)=q^\top\mathcal{R}_m^\top\mathcal{R}_nk=q^\top\mathcal{R}_{n-m}k$$
> 注意，这里的 $\mathcal{R}_{m}$ 是一个单位正交矩阵，因此不会改变向量的模长。

