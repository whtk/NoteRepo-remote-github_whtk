> 来自论文 - Res2Net: A New Multi-Scale Backbone Architecture

1. 本文提出 Res2Net，通过在一个 residual block 中构建  hierarchical residual-like connections
2. 提出的模块可以插入 CNN backbone 模型来提高性能

## Introduction

1. 讲了一下多尺度特征的重要性
2. 大多数的网络使用分层的多尺度表征，但是本文则提出了一个更为细粒度的多尺度表征 Res2Net：将一个 $n$ 通道的 $3\times 3$ 的滤波器替换成更小的滤波器组，每个小的滤波器有 $w$ 个通道，如图：![](./image/Pasted%20image%2020230212204843.png)然后这些小的滤波器以 hierarchical residual-like 的风格进行连接，从而增加输出特征包含的尺度。
3. 具体来说，就是将输入 feature map 分成几组，使用一些 filters 提取其中的一组特征，然后前一组的输出特征和当前组的连接起来，使用另一些 filters 进行特征提取，重复直到处理完所有的组，从而可以形成很多个尺度
4. Res2Net 导致一个新的维度，称为 scale 的数量，也就是前面说的 组 的数量，作者发现增加这个的效果要比增加 深度的效果更明显

## 相关工作（略）

## RES2NET

### Res2Net 模块
> Res2 的名字来源：**res**idual-like connections within a single **res**idual block，两个 res

更详细地讲，在 $1\times 1$ 卷积之后，将 feature maps 分成 $s$ 个 feature map 子集，记为 $\mathbf{x}_{i},i\in \{1,2,\dots,s\}$  。每个 $\mathbf{x}_{i}$ 和原来的 feature map 的 spatial size 相同，只是 channel 变成了 $\frac{1}{s}$，然后除了 $\mathbf{x}_{1}$，其他所有的 $\mathbf{x}_{i}$ 都有一个对应的 $3\times 3$ 卷积核，记为 $\mathbf{K}_{i}()$，同时记 $\mathbf{y}_{i}$ 为卷积后的输出，然后$\boldsymbol{x}_{i}$ 和前一组的输出 $\mathbf{y}_{i-1}$ 相加，送入到 $\boldsymbol{K}_{i}()$。 $\mathbf{x}_{1}$ 不进行 $3\times 3$ 卷积的一个好处就是减少了参数。
最后，输出 $\mathbf{y}_{i}$ 可以写为：$$\mathbf{y}_i= \begin{cases}\mathbf{x}_i & i=1 \\ \mathbf{K}_i\left(\mathbf{x}_i\right) & i=2 \\ \mathbf{K}_i\left(\mathbf{x}_i+\mathbf{y}_{i-1}\right) & 2<i \leqslant s\end{cases}$$
每个 $3\times 3$ 的卷积算子可以潜在地从前面所有的 split $\left\{\mathbf{x}_j, j \leq i\right\}$ 中获取信息，然后每经过一个 $3\times 3$ 的卷积算子都可以增加一次感受野，从而可以在一层里面包含很多种尺度的特征。
>  $1\times 1$ 卷积 的作用就是只改变 channel 的维度，其他维度保持不变

完成上面的操作之后，为了更好地融合这些多尺度的特征，将这些 splits 进行拼接然后通过一个 $1\times 1$ 的卷积，从而能够更高效的处理特征。

$s$ 越大，能够学习具有更丰富感受野大小的特征，而级联引入的计算/内存开销可以忽略不计。

### 和其他模块集成

通过在 Res2Net模块的 residual connections 之前添加SE 模块，能够使得 Res2Net 从中受益：![](./image/Pasted%20image%2020230212211757.png)
emmm确实受益了，但是不多。。