> Google，2016


1. 引入 WaveNet，基于 DNN 的波形生成模型，模型是 ully probabilistic and autoregressive
2. 可以在大量数据中进行高效训练
3. 比最好的  parametric and concatenative systems 效果好
4. 一个 WaveNet 模型就可以捕获多个不同说话人的特征，且可以把 speaker identify 作为条件进行切换
5. 当训练用于音乐生成时，可以生成新的且真实度很高的音乐段
6. 甚至还可以作为判别模型，在音素识别上效果都不错

## Introduction

1. 受 PixelRNN 启发，将像素或单词的联合概率建模为条件分布乘积可以实现 SOTA 的生成
2. 提出 WaveNet，基于 PixelRNN 的音频生成模型，贡献如下：
	1. 可以生成自然度很高的原始波形信号
	2. 基于 dilated causal convolution 发明了一种新的架构来建模波形信号的长时依赖
	3. 当以 speaker identity 为条件时，可以生成不同的声音
	4. 可以用于生成音乐

## WaveNet

给定波形信号 $\mathbf{x}=\left\{x_1, \ldots, x_T\right\}$，其联合分布为：
$$p(\mathbf{x})=\prod_{t=1}^T p\left(x_t \mid x_1, \ldots, x_{t-1}\right)$$
每个样本 $x_t$ 都基于所有前面时刻的样本。

通过堆叠的卷积层来建模条件概率分布，模型中没有 pooling 层，模型的输出和输入的时间维度是相同的。

模型采用 softmax 层输出下一个时刻的样本值 $x_t$ 的 categorical distribution，通过最大化其对数似然来优化参数。

### DILATED CAUSAL CONVOLUTIONS

WaveNet 的主要组成是因果卷积，即在时刻 $t$ 模型的预测输出 $p\left(x_{t+1} \mid x_1, \ldots, x_t\right)$ 不会依赖于未来的时间步 $x_{t+1}, x_{t+2}, \ldots, x_T$，如图：
![](image/Pasted%20image%2020230824221100.png)

训练的时候，所有 timestep 的条件预测可以并行执行，因为所有的 timestep 的 GT 是已知的，生成的时候是顺序的：
+ 每个样本完成预测后，送回到网络来预测下一个样本

因果卷积的问题是需要很多层，或者需要更大的 filters 来增加感受野，如上图的感受野为 5（#layers + filter length - 1 = 4+2-1），本文使用 dilated convolutions  来增加感受野。
> dilation 为 1 时就是标准的卷积。

![](image/Pasted%20image%2020230824222525.png)
Stacked dilated convolutions 使得网络有更大的感受野同时层数更少。本文使用的 dilation 每层加倍，到达最大后重复，即：
$$
1,2,4, \ldots, 512,1,2,4, \ldots, 512,1,2,4, \ldots, 512
$$
背后的原理是：
+ 每个 $1,2,4, \ldots, 512$ 都有 1024 的感受野，可以看作是一个 $1\times 1024$ 的卷积
+ 通过堆叠可以进一步增加感受野

### SOFTMAX DISTRIBUTIONS

categorical distribution 灵活且可以建模任意分布。

由于音频信号通常存储为 16 比特，从而 softmax 需要 65536 个概率。这显然太大了，于是首先对信号进行 $\mu$ 率压缩，然后量化到 256 个值：
$$f\left(x_t\right)=\operatorname{sign}\left(x_t\right) \frac{\ln \left(1+\mu\left|x_t\right|\right)}{\ln (1+\mu)}$$
其中，$-1<x_t<1 , \mu=255$。

### GATED ACTIVATION UNITS

使用和 PixelRNN 相同的 GAU 如下：
$$\mathbf{z}=\tanh \left(W_{f, k} * \mathbf{x}\right) \odot \sigma\left(W_{g, k} * \mathbf{x}\right)$$
其中，$*$ 为卷积，$\odot$ 为 element-wise 乘积，$\sigma(\cdot)$ 为 sigmoid function，$k$ 为 layer index，$f,g$ 分别表示 filter 和 gate，实现发现，这种非线性激活比 ReLU 效果好。

### RESIDUAL AND SKIP CONNECTIONS

使用了 residual  和 arameterised skip connection 来加速收敛，从而可以训练更深的模型，如图：
![](image/Pasted%20image%2020230824223603.png)
> 输入和输出的维度是一样的，当输入的音频长度远大于 感受野大小时，pad 部分就可以忽略不计。


### CONDITIONAL WAVENETS

给定条件输入 $\mathbf{h}$，WaveNet 建模条件分布 $p(\mathbf{x} \mid \mathbf{h})$ 如下：
$$p(\mathbf{x} \mid \mathbf{h})=\prod_{t=1}^T p\left(x_t \mid x_1, \ldots, x_{t-1}, \mathbf{h}\right)$$
通过把外部输入变量作为条件，可以引导 WaveNet 生成特定特征的音频。例如，在多说话人情况下可以选择 speaker identity，对于 TTS 可以说 text。

用两种方式来引入条件：

global condition，是单个的 latent representation $\mathbf{h}$，其可以影响所有 time step 的输出，此时的激活函数变成：
$$\mathbf{z}=\tanh \left(W_{f, k} * \mathbf{x}+V_{f, k}^T \mathbf{h}\right) \odot \sigma\left(W_{g, k} * \mathbf{x}+V_{g, k}^T \mathbf{h}\right)$$

 
对于 local condition，有时间序列 $h_t$ （如 TTS 中的语言特征），首先通过 transposed convolutional 上采样到和音频序列相同的长度 $L$，记为，$\mathbf{y}=f(\mathbf{h})$，此时激活函数为：
$$\mathbf{z}=\tanh \left(W_{f, k} * \mathbf{x}+V_{f, k} * \mathbf{y}\right) \odot \sigma\left(W_{g, k} * \mathbf{x}+V_{g, k} * \mathbf{y}\right)$$

### CONTEXT STACKS（上下文堆栈）

一种补充方法是使用一个单独的、较小的上下文堆栈 处理音频信号的较长部分，并在 local condition 下使用一个较大的 WaveNet，该 WaveNet 只处理音频信号的较小部分（在末端裁剪）。
> 没懂

## 实验

在三个不同的任务上作评估：
+ 多说话人语音生成
+ TTS
+ 音乐生成