
**重参数** 主要用于处理这样一种目标函数：$$L_\theta=\mathbb{E}_{z \sim p_\theta(z)}[f(z)]$$
其本质就是求，$z$ 在满足分布 $p_\theta(z)$ 下的 $f(z)$ 期望值。对于连续和离散情况，可以写为：$$\int p_\theta(z) f(z) d z(\text { 连续情形 }) \quad \quad \quad \sum_z p_\theta(z) f(z) \text { (离散情形) }$$
## 连续

考虑连续情况下：$$L_\theta=\int p_\theta(z) f(z) d z$$
其中的 $p_\theta(z)$ 一般是显示的分布（也就是已知分布），如正太分布。

但是计算的时候通常是不可能做积分的，所以一般通过采样来近似，并且在采样的时候，由于参数 $\theta$ 是需要优化的，我们**要保留 $\theta$ 的梯度**。

那要在带有参数 $\theta$ 的分布中采样，可以这样做：
1. 从不带有参数 $\theta$ 的分布 $q(\epsilon)$ 中采样一个随机变量 $\epsilon$
2. 通过带参数的变换函数 $z=g_\theta(\epsilon)$ 生成 $z$
那么上式变为：$$L_\theta=\mathbb{E}_{\varepsilon \sim q(\varepsilon)}\left[f\left(g_\theta(\varepsilon)\right)\right]$$
这时候被采样的分布就没有任何参数了，因此可以直接采样一些点，然后带入计算 loss。

## 离散

在离散情况下，为了方便，把随机变量 $z$ 替换成 $y$，则离散时的目标函数为：$$L_\theta=\mathbb{E}_{y \sim p_\theta(y)}[f(y)]=\sum_y p_\theta(y) f(y)$$
由于已经是离散的，这里的 $y$ 一般是有限且可以枚举的，或者说，在模型训练中，这里的 $y$ 就是一个分类值，此时的分布就变成不同类别的概率：$$p_\theta(y)=\operatorname{softmax}\left(o_1, o_2, \ldots, o_k\right)=\frac{1}{\sum_{i=1}^k e^{o i}}\left(e^{o 1}, e^{o 2}, \ldots, e^{o k}\right)$$
式中的 $k$ 表明可以分为 $k$ 类，$o_i$ 是 $\theta$ 的函数，其实就是输出的 logit，$\theta$ 其实就是模型的权重参数。

但是其实还是要采样，因为类别可能非常大，全部计算很困难。

如果通过采用若干个点就可以估计 $L_\theta$ ，同时还能保留梯度的信息那就最好。

假定已知每个类别的概率为 $p_1,p_2,\cdots,p_k$，**Gumbel Max** 是一种依概率采样的方法：$$\underset{i}{\arg \max }\left(\log p_i-\log \left(-\log \varepsilon_i\right)\right)_{i=1}^k, \quad \varepsilon_i \sim U[0,1]$$
就是先从 0-1 均匀分布中采样出 $k$ 个噪声值 $\epsilon_i$，然后代入上式计算得到 $k$ 个数，然后选择最大的那个数对应的索引即可实现依不同类的概率进行采样。这个过程精确等于依照概率 $p_1,p_2,\cdots,p_k$ 采样每个类。
> 按：这里的均匀分布其实就和连续情况中的正太分布 $q(\epsilon)$ 相对应了。

但是但是！argmax 的操作会丢失梯度信息（**因为在连续情况下，采样之后是通过函数 $f$ 来计算的，这里面包含了梯度，但是离散情况下使用 Gumbel Max 丢失了梯度信息**），于是我们可以把 argmax 转换为 softmax：$$\operatorname{softmax}\left(\left(\log p_i-\log \left(-\log \varepsilon_i\right)\right) / \tau\right)_{i=1}^k, \quad \varepsilon_i \sim U[0,1]$$
> 按：这也正是 soft 这个单词的来历，相当于从 hard 转换为 soft，很简单的数学原理，当参数 $\tau > 0$ 越小的时候，其输出就越接近 one-hot 的形式，也就和 argmax 的表现越接近。 
> 注：这里得到的其实是一个向量，而 argmax 得到的是一个索引，但是其实没关系。

## 附录 - Gumbel Max 的证明

为什么通过上面的 log 计算最终得到的就是对应 $p_i$ 的分布概率采样呢。首先，由于有 argmax 操作，那么：$$\log p_i-\log \left(-\log \varepsilon_i\right)>\log p_j-\log \left(-\log \varepsilon_j\right),\quad j\neq i$$
化简一下，有：$$\varepsilon_j<\varepsilon_i^{p_j / p_i} \leq 1,\quad j\neq i$$
由于 $\varepsilon_j$ 是均匀分布，那么 $\varepsilon_j<\varepsilon_i^{p_j / p_i}$ 的概率就是 $\varepsilon_i^{p_j / p_i}$，那么，假设 $i=1$，上市对于所有的 $j$ 同时成立的概率为：$$\varepsilon_1^{p_2 / p_1} \varepsilon_1^{p_3 / p_1} \ldots \varepsilon_1^{p_k / p_1}=\varepsilon_1^{\left(p_2+p_3+\cdots+p_k\right) / p_1}=\varepsilon_1^{\left(1 / p_1\right)-1}$$
然后对所有的 $\varepsilon_1$ 取平均，则：$$\int_0^1 \varepsilon_1^{\left(1 / p_1\right)-1} d \varepsilon_1=p_1$$
同理其他的下标也成立。