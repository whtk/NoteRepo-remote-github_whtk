> 平安科技，ICASSP 2021

1. 提出 location-variable convolution，来建模波形序列的相关性
2. LVC 使用具有不同系数的卷积核对不同的波形区间进行卷积，其中的系数是根据条件声学特征来预测的
3. 基于 LVC 设计 用于波形生成的LVCNet，用在 Parallel WaveGAN  中实现更高效的 vocoder
4. 实验表明，速度可以提高四倍，但是质量没有下降

## Introduction

主要贡献：
+ 提出新的卷积方法，LVC，用来建模时间相关的特征
+ 基于 LVC 设计 LVCNet，用在 Parallel WaveGAN 实现更高效的 vocoder
+ 对比实验

## 方法

### LVC

传统的线性预测 vocoder 中，采用全极点线性滤波器以自回归的方式生成波形，线性预测系数是根据声学特征计算的。这个过程有点类似于自回归的 wavenet vocoder，除了线性预测中的系数在每帧中都不一样，而 wavenet 在所有帧中使用了相同系数的卷积核。

设卷积的输入序列 $\boldsymbol{x}=\left\{x_1, x_2, \ldots, x_n\right\}$，local 条件序列记为 $\boldsymbol{h}=\left\{h_1, h_2, \ldots, h_m\right\}$，且 条件序列中的元素 与 输入序列中的连续间隔相关联。为了有效地利用局部相关性对输入序列的特征进行建模，LVC 对输入序列中的不同区间使用不同的卷积核来实现卷积运算。
> 按：local 条件序列一般可以是 mel spectrogram 或者 MFCC。

具体来说，有一个 kernel predictor 根据 local 条件序列来预测 卷积核，local 条件序列中的每个元素都对应一系列的卷积核，用于在相关的序列中进行卷积运算。
> 换句话说，卷积核在每个序列片段（区间）都不一一样，不同区间使用其对应的卷积核，而这个区间的卷积核与当前的 local 条件向量有关。

![](image/Pasted%20image%2020230527152259.png)

和 wavenet 相似，也采用了 gated activation unit，local 条件卷积表示为：$$\begin{aligned}
& \left\{\boldsymbol{x}_{(i)}\right\}_m=\operatorname{split}(\boldsymbol{x}) \\
\left\{\boldsymbol{W}_{(i)}^f\right. & \left., \boldsymbol{W}_{(i)}^g\right\}_m=\text { Kernel Predictor }(\boldsymbol{h}) \\
\boldsymbol{z}_{(i)} & =\tanh \left(\boldsymbol{W}_{(i)}^f * \boldsymbol{x}_{(i)}\right) \odot \sigma\left(\boldsymbol{W}_{(i)}^g * \boldsymbol{x}_{(i)}\right) \\
\boldsymbol{z} & =\operatorname{concat}\left(\boldsymbol{z}_{(i)}\right)
\end{aligned}$$
其中，$\boldsymbol{x}_i$ 表示和 $h_i$ 对应的输入序列片段（区间），$\boldsymbol{W}_{(i)}^f, \boldsymbol{W}_{(i)}^g$ 表示对应于 $\boldsymbol{x}_i$ 的 filter 和 gate 核。
> 其实非常简单，如果没有 LVC，那么 $\boldsymbol{W}_{(i)}^f, \boldsymbol{W}_{(i)}^g$ 就没有下标 $i$，也就是在每个 local 条件序列中的卷积是固定的，本文就是提出 一个 kernel predictor 来预测不同的 $i$ 对应不同的 kernel。

这样做的好处是，可以为不同的条件序列生成不同的核，比传统的卷积网络具有更强大的建模长期依赖性的能力。

### LVCNet 和 Parallel WaveGAN

![](image/Pasted%20image%2020230527152946.png)

堆叠具有不同 dilations 的 LVC 得到 LVCNet，输入输出添加 linear 层 实现信道转换，同时还有 skip connection。

用在 Parallel WaveGAN 中，其实就是直接替换其中的 WaveNet 模块。

## 实验（略）