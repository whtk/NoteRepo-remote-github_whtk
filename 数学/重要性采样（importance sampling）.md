
# 来自知乎
> https://zhuanlan.zhihu.com/p/41217212

## 蒙特卡洛积分

对于函数 $f(x)$ 如果要求其在区间 $[a,b]$ 之间的积分，则用解析形式可以表述为 $\int_a^bf(x)dx$，但是这个过程通常难以解析，无法直接求积分。

于是可以采用估计的方式，在区间 $[a,b]$ 之间进行采样，得到一系列的采样点 $\{x_1,x_2\ldots,x_n\}$ 和对应的取值 $\{f(x_1),f(x_2),\ldots,f(x_n)\}$。

此时，积分可以估计为样本点之间的距离（宽度）乘以取值，即 $\begin{aligned}\int_a^bf(x)dx=\frac{b-a}N\sum_{i=1}^Nf(x_i)\end{aligned}$，这里的 $\frac{b-a}{N}$ 就是宽度（假设两个采样点之间的距离是固定的）。

随着采样点数的增加，估计越来越准确。

## 重要性采样

一个问题就是，对于函数值 $f(x_i)$ 越高的区域，这个误差会越大。

所以可以在采样的时候，以更大的概率选择数值较大的区域，从而提高准确度，假设以分布 $p(x)$ 在原函数进行采样，则对于蒙特卡洛积分，$p(x)$ 为 $[a,b]$ 上的均匀分布。

因此，如果 $p(x)$ 可以在 $f(x)$ 数值大的地方有更高的概率，则可以提高估计的准确度。

但是这也带来一个问题，此时就不能对 $\{f(x_1),f(x_2),\ldots,f(x_n)\}$ 做简单的求和了，因为每个取值的权重不一样，所以要根据某个权重对这些值进行加权，而这个权重则称为重要性权重。

再进一步拓广，假设原函数定义在分布 $\pi(x)$ 下，此时要求的不是积分而是函数在概率分布下的期望 $E[f(x)]=\int_{x}\pi(x)f(x)dx$，但是这个期望也不好求（因为正常的求法需要对 $\pi(x)$ 进行采样），所以借助一个更加简单的分布 $p(x)$ 来进行采样，此时期望估计为 $E[f(x)]=\int_xp(x)f(x)dx\approx\frac1N\sum_{i=1}^Nf(x_i)$，然后用一个小技巧：$\pi(x)=p(x)\frac{\pi(x)}{p(x)}$，此时期望写为：
$$E[f(x)]=\int_xp(x)\frac{\pi(x)}{p(x)}f(x)dx$$
然后在 $p(x)$ 上采样得到一系列的点 $\{x_1,x_2\ldots,x_n\}$，估计期望为：
$$E[f(x)]=\frac{1}{N}\sum_{i=1}^{N}\frac{\pi(x_i)}{p(x_i)}f(x_i)$$
其中的 $\frac{\pi(x_i)}{p(x_i)}$ 就是重要性权重。

# 来自 Data Science
> https://towardsdatascience.com/importance-sampling-introduction-e76b2c32e744

考虑求解函数 $f(x)$ 期望，其中 $x\sim p(x)$，近似估计如下：
$$E[f(x)]=\int f(x)p(x)dx\approx\frac{1}{n}\sum_if(x_i)$$
上面的就是简单的 蒙特卡洛采样法，也就是从分布 $p(x)$ 中采样得到一系列的点然后求平均。

但是，如果 $p(x)$ 很难采样呢？

于是有以下的公式：
$$E[f(x)]=\int f(x)p(x)dx=\int f(x)\frac{p(x)}{q(x)}q(x)dx\approx\frac{1}{n}\sum_{i}f(x_{i})\frac{p(x_{i})}{q(x_{i})}$$
这里的 $x$ 是从 $q(x)$ 中采样的，且 $q(x)$ 不为 0 。此时的期望估计变为，从分布 $q(x)$ 中采样，计算采样比（或采样权重） $\frac{p(x)}{q(x)}$，带入上式进行数值求解。

## 来自 Stanford 教材

很多情况下， 我们需要计算 $\mu=\mathbb{E}(f(\boldsymbol{X}))$，其中 $f(\boldsymbol{x})$ 在区域 $A$ 之外的值接近 0，且 $\mathbb{P}(\boldsymbol{X}\in A)$ 很小，于是需要用到重要性采样。

考虑计算 $\mu=\mathbb{E}(f(\boldsymbol{X}))=\int_{\mathcal{D}}f(\boldsymbol{x})p(\boldsymbol{x})\mathrm{d}\boldsymbol{x}$，其中 $p$ 为区域 $\mathcal{D}\subseteq\mathbb{R}^{d}$ 上的 PDF，$f$ 是被积函数，当 $\boldsymbol{x}\notin{D}$ 时，$p(\boldsymbol{x})=0$，此时如果 $q$ 是在 $\mathbb{R}^{d}$ 恒正的 PDF，则：
$$\mu=\int_{\mathcal{D}}f(\boldsymbol{x})p(\boldsymbol{x})\operatorname{d\boldsymbol{x}}=\int_{\mathcal{D}}\frac{f(\boldsymbol{x})p(\boldsymbol{x})}{q(\boldsymbol{x})}q(\boldsymbol{x})\operatorname{d\boldsymbol{x}}=\mathbb{E}_q{\left(\frac{f(\boldsymbol{X})p(\boldsymbol{X})}{q(\boldsymbol{X})}\right)}$$
其中 的因子 $p(\boldsymbol{x})/q(\boldsymbol{x})$ 称为 似然比，$q$ 为重要性分布。


后面大部分都是分析估计量的均值和方差。