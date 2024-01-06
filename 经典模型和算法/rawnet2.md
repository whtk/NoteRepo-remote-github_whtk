> Improved RawNet with Feature Map Scaling for Text-independent Speaker Verification using Raw Waveforms 笔记


1. RawNet 直接从原始波形中提取 speaker embedding，本文通过缩放特征图 feature map 来改进 RawNet
2. 采用 sigmoid 非线性函数的 scale vector，其维度等于滤波器的数量，采用加法、乘法或者两者兼有的方式进行缩放特征图
3. 同时用SincNet的sinc卷积层替换第一卷积层
4. 相比于原始的 RawNet，性能最好的系统将 EER 降低了一半

## Introduction

提出 特征图缩放（FMS），使用维度和滤波器数量相同的 scale vector，值在 0-1 之间，采用 sigmoid 而非 softmax 的原因是，sigmoid 可以同时激活多个滤波器，而 softmax 操作抑制这种激活。

同时研究了使用 self-attentive pooling 和 self-multi-head-attentive pooling 进行帧级信息聚合，来替代RawNet 的 GRU 层。

具体而言，将 FMS 用于特征图，按顺序进行乘法、加法或者两者兼有的操作：
+ 通过乘法运算缩放特征图，希望独立强调特征图的每个滤波器
+ 通过加法运算缩放特征图，希望提供小的扰动来增强判别能力
+ 或者依次使用乘法和加法
同时研究了用sinc卷积层代替RawNets第一卷积层，能够比传统卷积层更好地捕获聚合频率响应。

进行 FMS 的四种方法：![](./image/Pasted%20image%2020221129104428.png)


## Baseline：RawNet

  
RawNet是一种神经说话人 embedding  提取器，它直接输入原始波形而无需预处理技术，并输出为说话人验证而设计的说话人 embedding。

RawNet采用 CNN-GRU 架构，其中第一个CNN层的步长与滤波器长度相同。前面的CNN层包括残差块接最大池化层，以提取帧级表示。然后，GRU层将帧级特征聚合为话语级表示，即 GRU输出的最后一个 time step。GRU 层然后接到全连接层，得到说话人 embedding 输出。

具体架构为（来自 RawNet 原始论文）：![](./image/Pasted%20image%2020221129110807.png)

## Filter-wise FMS

FMS 独立缩放特征图的每个 filter，采用一个 scale vector，其维度和 filters 的数量相等，这个过程没有额外的参数。

令 $c=\left[c_1, c_2, \cdots, c_F\right]$ 为 feature map，其中 $c_f \in \mathbb{R}^T$ ，$T ,F$ 分别为序列长度和滤波器的数量。

scale vector 的计算过程为：
+ 在时间轴上进行 global average pooling 
+ 通过一个全连接层
+ 最后是 sigmoid 激活函数

最后得到的向量 $s=\left[s_1, s_2, \cdots, s_F\right]$，其中 $s_f \in \mathbb{R}^1$ 为标量。

那么 scaling 的操作可能为：
+ 加：$c_f^{\prime}=c_f+s_f$
+ 乘：$c_f^{\prime}=c_f \cdot s_f$
+ 组合：
	+ $c_f^{\prime}=\left(c_f+s_f\right) \cdot s_f$
	+ $c_f^{\prime}=c_f \cdot s_f+s_f$
上面的计算过程中，$s_f$ 都要进行 广播，以实现 element-wise 的操作，如第一张图所示。

乘法 FMS 和注意力机制有一定的相似性，注意力机制仅仅使用 softmax激活函数 来突出特征图，这可以解释为在滤波器域中使用多头部注意机制，head 的数量等于滤波器的数量。

加法 FMS 按滤波器缩放的方法将0到1之间的值添加特征图，目的是将数据驱动的扰动应用于具有相对较小值的特征图，从而增加了特征图的辨别力。


## 实验和结果分析（略）