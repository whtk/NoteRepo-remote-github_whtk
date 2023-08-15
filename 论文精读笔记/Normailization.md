> 总结了不同的 Normalization 方法

## Batch Norm
> 实际上，Batch Normalization 和对输入样本进行 Normalization 差不多，只不过样本经过激活函数得到一层一层的中间值之后不再是输入了，但是本质上还是对这些数据进行归一化，或者更准确地说，是在 卷积或者线性层之后，激活层之前对数据进行归一化，从而使得激活后的数据分布更为均匀。

对图像来说，设输入一个batch的图像（或者说中间特征） $x \in \mathbb{R}^{N \times C \times H \times W}$（$N$ 代表 batch size），则 batch normalization 在 mini-batch 上对所有的图像数据（除了通道）求均值和方差并进行归一化：
$$
\begin{gathered}
\mu_{c}(x)=\frac{1}{N}\frac{1}{H}\frac{1}{W} \sum_{n=1}^{N}\sum_{h=1}^{H}\sum_{w=1}^{W} x_{n c h w} \\
\sigma_{c}(x)=\sqrt{\frac{1}{N}\frac{1}{H}\frac{1}{W} \sum_{n=1}^{N}\sum_{h=1}^{H}\sum_{w=1}^{W}\left(x_{n c h w}-\mu_{c}(x)\right)^{2}+\varepsilon}
\end{gathered}
$$
完成均值和方差计算后，计算归一化后的特征：
$$
x_{\text{norm}} = \frac{x-\mu_{c}}{\sqrt{\sigma_{c}(x)^2+\epsilon}}
$$
同时进行放射变换：
$$
\text{BN}(x) = \gamma \cdot x_{\text{norm}} + \beta
$$
其中，$\gamma, \beta$ 为训练过程中可学习的参数。

> 当 batch size 为 1 的时候，如果仍然使用 batch norm，此时反而可能导致性能变差：
> 1. batch norm 本质是对 mini batch 的数据进行归一化，单个样本上“平均”的 batch norm 在样本之间差异很大，从而导致更高的方差
> 2. 已经有很多实验证明，当 batch size < 8 时使用 batch norm 会导致性能大幅下降 https://stackoverflow.com/questions/59648509/batch-normalization-when-batch-size-1
> 3. 此时推荐使用 Group Norm

> Batch Normalization 还有轻微的 正则化作用，因为在计算均值和标准差的时候用的是 mini batch，这个过程中存在一点误差（或者说并不是无偏估计量），有点类似与 Dropout。
> 
> 还有一个就是，在测试的时候可能是一个一个样本进行测试的，此时 batch size 相等于为1，测试时，使用（训练时候的均值和方差的）指数加权平均 来估计测试时候的均值和方差

## Layer Norm
LN 可以克服 BN 的缺点，它在 $(C,H,W)$ 的维度上计算均值和方差：
并进行归一化：
$$
\begin{gathered}
\mu_{n}(x)=\frac{1}{C}\frac{1}{H}\frac{1}{W} \sum_{c=1}^{C}\sum_{h=1}^{H}\sum_{w=1}^{W} x_{n c h w} \\
\sigma_{n}(x)=\sqrt{\frac{1}{C}\frac{1}{H}\frac{1}{W} \sum_{c=1}^{C}\sum_{h=1}^{H}\sum_{w=1}^{W}\left(x_{n c h w}-\mu_{n}(x)\right)^{2}+\varepsilon}
\end{gathered}
$$


## Group Normalization
由于在 batch norm 中过于依赖于 batch size，会存在以下几个问题：
+ batch size 过小的时候方差很大
+ 训练和测试的时候存在不匹配

GN 处于 LN 和 IN 之间，首先将 channel 分为多个组，对每个组做 LN 的归一化。具体来说，给定 batch 样本 $x \in \mathbb{R}^{N \times C \times H \times W}$，GN 先将样本 reshape 为 $\tilde{x} \in \mathbb{R}^{N \times G \times {(C//G)} \times H \times W}$，然后在 ${(C//G)} \times H \times W$ 上计算得到 $\mu_{ng}$ 和 $\sigma_{ng}$，最后进行归一化。

显然，当 $G=1$ 时，退化为 LN，当 $G=C$ 时，变成 IN。


## Instance Normalization
对图像来说，设输入一个batch的图像 $x \in \mathbb{R}^{N \times C \times H \times W}$（$N$ 代表 batch size），则 $IN$ 将在张图片的每个通道上做归一化：
$$
I N(x)=\gamma\left(\frac{x-\mu(x)}{\sigma(x)}\right)+\beta
$$ 
其中，$\gamma, \beta$ 为仿射参数，且有：
$$
\begin{gathered}
\mu_{n c}(x)=\frac{1}{H W} \sum_{h=1}^{H} \sum_{w=1}^{W} x_{n c h w} \\
\sigma_{n c}(x)=\sqrt{\frac{1}{H W} \sum_{h=1}^{H} \sum_{w=1}^{W}\left(x_{n c h w}-\mu_{n c}(x)\right)^{2}+\varepsilon}
\end{gathered}
$$
$\mu_{nc}$ 的下标 $nc$ 代表第 $n$ 张图片的第 $c$ 个通道。

## Conditional Instance Normalization
$IN$ 学习一个单一的仿射参数集（ $\gamma$ 和 $\beta$ ），在所有的图片上都通用，而 $CIN$ 为每图片都学习一个仿射参数集（$\gamma^s, \beta^s$）：
$$
C I N(x ; s)=\gamma^{s}\left(\frac{x-\mu(x)}{\sigma(x)}\right)+\beta^{s}
$$
其中，$\mu(x),\sigma(x)$ 和 $IN$ 一样。

## Adaptive Instance Normalization（AdaIN）
与上述方法不同，$AdaIN$ 中没有可学习的仿射参数，而是需要两张图片，输入图 $x$ 和风格图 $y$，其归一化过程如下：
$$
\operatorname{AdaIN}(x, y)=\sigma(y)\left(\frac{x-\mu(x)}{\sigma(x)}\right)+\mu(y)
$$
其中，$\mu(x),\sigma(x),\mu(y),\sigma(y)$ 的计算和 $IN$ 一样。