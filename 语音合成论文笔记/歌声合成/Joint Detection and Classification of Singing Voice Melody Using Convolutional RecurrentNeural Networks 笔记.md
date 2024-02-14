> MDPI，2019，KAIST

1. 歌声韵律提取包含两个任务：
	1. 检测 activity
	2. 估计 voiced segments 的 pitch
2. 提出 joint detection and classification (JDC) 网络，同时进行声音检测和音高估计，包含：
    1. 主网络，预测歌声旋律的音高轮廓，基于卷积循环神经网络和残差连接
    2. 辅助网络，利用主网络得到的 multi-level features 辅助检测歌声
3. 超过了 SOTA 的方法

## Introduction

1. 音高提取是估计旋律源的基频或音高
2. 歌声通常比背景音乐更响，而且有独特的特征，如颤音和共振峰
3. 歌声旋律提取包含声音检测，因为旋律源不总是活跃的
4. 提出 joint detection and classification (JDC) 网络，可以看作多任务学习
5. 贡献：
    1. 提出了一个 CRNN 架构，用于高分辨率的音高分类
    2. 提出了 JDC 网络，可以独立执行两个任务
    3. 提出了一个 joint melody loss，用于结合两个任务
    4. 通过公共测试集的结果表明，提出的方法优于 SOTA

## 方法

如图：
![](image/Pasted%20image%2020240214165704.png)

### 主网络

主网络如图 a，用于从音频中提取歌声旋律，包含：
+ 1 ConvBlock
+ 3 ResBlocks
+ 1 PoolBlock
+ 1 双向 LSTM 层

ResBlock 是 ConvBlock 的变种，包含额外的 BN/LReLU，一个 max-pooling 层，一个 skip connection。Max-pooling 只在频率维度上进行，时间维度不变。Skip connection 有一个 1 × 1 卷积层来匹配两个特征图之间的维度。PoolBlock 包含 BN、LReLU 和 MaxPool。最后，Bi-LSTM 层从卷积块中取 31 帧特征（2 × 256）并通过 softmax 函数预测音高标签。

音高标签范围从 D2 (73.416 Hz) 到 B5 (987.77 Hz)，分辨率为 1/16 个半音（6.25 cents）。此外，添加了“non-voice”（或“zero-pitch”）标签。当歌声不在时为此标签。标签总数为 722。

主网络输入为 spectrogram，先将音频下采样到 8 kHz，使用 1024 点的 Hann 窗和 80 个样本（10 ms）的 hop size 来计算 spectrogram，并在对数尺度上压缩幅度。最后，使用 513 个频率 bin（0 Hz–4000 Hz）和 31 个连续帧作为主网络的输入。

### 损失函数

将音高范围的连续尺度量化为离散值。用 one-hot 向量表示。损失函数的一个问题是，除非预测的音高与量化尺寸（我们的情况下为 6.25 cents）内的 ground-truth 音高足够接近，否则它被视为“错误类”。为了减轻 ground truth 附近音高的过多损失，提出了高斯模糊版本的 one-hot 向量。因此主网络的损失函数 $L_{pitch}$ 定义如下：
$$\begin{gathered}L_{pitch}=\mathcal{C}E(\mathbf{y}_{g^{\prime}},\mathbf{\hat{y}})\\y_g(i)=\begin{cases}exp(-\frac{(c_i-c_{true})^2}{2\sigma_g^2})&\mathrm{if~}c_{true}\neq0\mathrm{~and~}|c_i-c_{true}|\leq M,\\0&\mathrm{otherwise},&\end{cases}\end{gathered}$$
其中 $\mathcal{C}E(y_g,\hat{y})$ 是音高预测的交叉熵损失。$c_{true}$ 是真实音高的常数索引，$c_i$ 是变量索引。$M$ 确定非零元素的数量。实验中将 $M$ 设置为 3，$\sigma_g$ 设置为 1。

### JDC 网络

主网络的输出层由音高标签和特殊的 non-voice 标签组成。在音高估计中，网络在每个帧上预测音高的连续变化。但是 voice detection 和 pitch estimation 任务对特征的需求处于不同的 level： 
+ pitch estimation 只需要邻近的 contextual information
+ voice detection 需要 wider context，如 vibrato 或 formant modulation

为了解决两个任务之间的差异，提出了 joint detection and classification (JDC) 网络。两个任务共享模块，然后在主网络上添加一个额外的分支用于歌声检测。JDC 网络共享 ConvBlock、ResBlock 和 PoolBlock，但每个任务有一个独立的 Bi-LSTM 模块。voice detection 任务使用了来自主网络的多级特征。ResBlock 的输出经过 max-pooling 后进行拼接，Bi-LSTM 层通过 softmax 函数预测是否有歌声。卷积块的特征是通过主网络和辅助网络共同学习的，损失函数也是由主网络的损失函数和辅助网络的损失函数组合而成。

JDC 通过最小化联合旋律损失进行优化，包含了主网络和辅助网络的两个损失函数。用两个网络来检测歌声。将主网络的 721 个音高预测求和，然后转换为一个“voice”预测 $o_{mv}$。将这个结果与辅助网络的输出 $o_{v}$ 相加，得到最终的 voice detection 结果：
$$o_{sv}=o_{mv}+o_v$$
然后，歌声检测的损失函数定义为输出和 ground truth $\mathbf{v}_{gt}$ 之间的交叉熵：
$$L_{voice}=CE(softmax(o_{sv}),\mathbf{v}_{gt})$$
最终，联合旋律损失函数由歌声检测的损失函数和音高估计的损失函数组合而成：
$$L_{joint}=L_{pitch}+\alpha L_{voice}$$
其中 $\alpha$ 是一个平衡权重，实验中使用 $\alpha=0.5$。

## 实验（略）
