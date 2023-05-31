> interspeech 2020

1. 基于人脸验证和CV相关领域进展，提出对 TDNN 网络的增强方法：
	1. 可以将初始帧层重构为具有 skip connection 的 1d-Res2Net 模块
	2. 引入 SE 模块来模拟通道之间的相关性
	3. 多层特征融合
	4. 使用 channel-dependent frame attention 改进 statistics pooling

## Introduction

1. 最近有很多关于 x-vector 的改进方法来提高性能，如
	1. 引入 ResNet，增强  embedding，加快收敛速率，避免梯度消失等问题
	2. 将 temporal self-attention 系统引入 pooling 层，只关注重要的帧（也可以视为一种 VAD来检测不相干的非语音帧）
2. 本文对 TDNN 的架构和 statistic pooling 进行改进，引入额外的 skip connection 在整个系统中传播和聚合信道信息，同时将全局上下文的Channel attention 引入 frame layer 和 statistics pooling layer 来进一步提高性能

## DNN 说话人识别系统

用两个系统做 baseline：
+ 基于 x-vector
+ 基于 ResNet

### Extended-TDNN x-vector

初始的 frame layer 由交织有 dense layer 的 1-dimensional dilated convolutional layer 组成。每个 filter 都可以看到前一层的所有特征。

dilated convolutional layer 的任务是逐步建立时间上下文。在所有 frame layer 中引入 skip connection。frame layer 之后是一个 attentive statistics pooling layer，用于计算最终 frame level 的特征的平均值和标准差。

注意力机制允许模型选择其认为相关的帧。在 statistics pooling 之后，引入了两个全连接层，第一个层充当 bottleneck layer ，以生成低维说话人特征embedding。

### ResNet-based r-vector

r-vector 基于 ResNet 架构，具体见 [[BUT system description to VoxCeleb speaker recognition challenge 2019 笔记]]。在 pooling 前将特征看成是 2d 的信号。

## ECAPA-TDNN 结构
![](./image/Pasted%20image%2020230131161343.png)

### Channel- and context-dependent statistics pooling

说话人特征可以在特定帧上提取，作者认为可以将这种时间注意力机制进一步拓展到 channel 维度，从而可以关注在同一时间段不同的说话人特征（如元音和辅音特征）。

采用 [[Attentive Statistics Pooling for Deep Speaker Embedding 笔记]] 中的注意力机制，但是把它用在 channel 维度：$$e_{t, c}=\boldsymbol{v}_c^T f\left(\boldsymbol{W} \boldsymbol{h}_t+\boldsymbol{b}\right)+k_c$$
其中，$\boldsymbol{h}_t$ 为上一层 frame layer 在时间 $t$ 激活后的输出，待训练的参数 $W \in \mathbb{R}^{R \times C} \text { and } b \in \mathbb{R}^{R \times 1}$ 将信息投影到低维的 $R$ 维表征中，且这个参数在所有的 $C$ 个 channel 中共享来避免过拟合。然后通过非线性函数 $f()$ 将信息变成得分，然后对 $\boldsymbol{v}_c \in \mathbb{R}^{R \times 1}$ 加权，最后加上偏置 $k_c$。
> 注意 $\boldsymbol{v}_c$ 和 $k_c$ 不共享，每个 channel 都单独一个。

然后在时间维度逐信道应用 softmax 函数得到注意力得分：$$\alpha_{t, c}=\frac{\exp \left(e_{t, c}\right)}{\sum_\tau^T \exp \left(e_{\tau, c}\right)}$$
然后会有一个的均值和标准差：$$\tilde{\mu}_c=\sum_t^T \alpha_{t, c} h_{t, c} .$$
$$\tilde{\sigma}_c=\sqrt{\sum_t^T \alpha_{t, c} h_{t, c}^2-\tilde{\mu}_c^2} .$$
最终 pooling 层的输出是将这两个向量拼接起来。
> 其实本质就是，之前的 attentive 是用标量 alpha 乘以向量（相当于这个标量在 channel 维度是共享一致的，而这里的 alpha 是一个矩阵，但是强调的是每个元素都不一样，即使是在同一个帧中，每个 channel 的贡献也不一样）

同时还将局部的 $\boldsymbol{h}_t$ 和全局的 $\boldsymbol{h}_t$ 拼接起来作为输入。

### 1-Dimensional Squeeze-Excitation Res2Blocks

原始的 x-vector 中 frame layer 的时间上下文为 15 帧，但是能够考虑全局信息最好，于是引入  1-dimensional Squeeze-Excitation 模块，首先 squeeze 操作计算 frame level 特征的时间均值：$$\boldsymbol{z}=\frac{1}{T} \sum_t^T \boldsymbol{h}_t$$
然后基于 $\boldsymbol{z}$ 来计算 excitation 操作：$$\boldsymbol{s}=\sigma\left(\boldsymbol{W}_2 f\left(\boldsymbol{W}_1 \boldsymbol{z}+\boldsymbol{b}_1\right)+\boldsymbol{b}_2\right)$$
得到权重向量，然后进行 channel wise 的乘法：$$\tilde{\boldsymbol{h}}_c=s_c \boldsymbol{h}_c$$
采用的是下图所示的 SE-Res2Block：![](./image/Pasted%20image%2020230131173417.png)
两个 dense layer，context 都是1帧，第一个 dense 层用于减少特征维度，第二个用于恢复维度（？？？），然后接 SE 模块，最后进行 skip connection。

### Multi-layer feature aggregation and summation

原始的 x-vector 仅使用最后一层 frame layer 的特征来计算统计值，但是 TDNN 其实是分层的，深层的特征固然和说话人密切相关，但是浅层的特征也有助于实现更鲁棒的说话人 embedding。

于是把所有的 SE-Res2Blocks 的输出进行拼接，形成所谓的 Multi-layer Feature Aggregation，然后用一个 dense layer 来处理这些信息，最后进行 attentive statistics pooling。

另一种利用多层信息的方式是使用某一层之前的所有的 SE-Res2Blocks 的输出（也包括输入层）作为该层的输入，具体来说，就是把 residual 那块的变成前面所有的 SE-Res2Blocks 的输出 的 和（求和而非拼接的原因是限制模型参数）。

## 实验

为每个样本生成6个增强样本：
+ MUSSAN 数据集，添加  babble, noise （2个）
+ RIR 数据集，添加 reverb）（1个）
+ 采用 open-source SoX，进行 tempo up, tempo down
+ 采用 FFmpeg 改变编码方式（交替使用 opus和 aac，两种有损编码格式）

特征：80维 MFCC，25 ms window with a 10 ms frame shift，进行了归一化，没用 VAD，同时使用 SpecAugment 进行数据增强

使用 cyclical learning rate scheduler + Adam 优化器，使用 AAM-softmax 损失进行训练，batch 为 128。

speaker embedding 是从最后的 FC 中提取的，然后使用余弦距离产生得分，然后使用自适应s范数对所有得分进行归一化。

评估指标：EER 和 MinDCF


## 结果

性能比较：
![](./image/Pasted%20image%2020230131213949.png)

效果还是很好的！

消融实验：![](./image/Pasted%20image%2020230131214446.png)
表明：
1. Channel- and context-dependent statistics pooling 能够改善 EER 
2. SE 模块也可以改善 EER，表明 frame level 的 有限的 时间上下文是不够的，需要全局信息作为补充
3. 多尺度Res2Net可以减少参数量同时提高模型性能

