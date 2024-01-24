> ICASSP 2019

1. 提出基于 CapNets 的 SER，考虑语音特征在频谱图中的空间关系
2. 在 capnet 中引入循环结构，提高时间敏感性，优于 CNN-LSTM baseline

## Introduction

1. 话语级的情感识别需要全局特征，既包含详细的局部信息，也包含和情感相关的全局特征
2. 最近使用 DNN 来提取 high-level 的表征，通常用的是 CNN，但是 CNN 对空间关系或方向信息不敏感
3. 有些论文使用了 ELM、RNN、自注意力等来构建话语级的特征
4. 先前使用的 CapNets 没有考虑时间信息，但是情感识别很需要这个
5. 本文贡献：提出一种顺序胶囊结构，首先将输入切分成一个一个的 window，然后迭代使用 CapNet，将其输出进行聚合，再使用一个 CapNet 得到最终的话语级特征，同时引入循环来建模时间信息

## Capsule Network

### 基本结构（略）

### Sequential Capsules (SeqCaps)

SER 的输入为可变长度的特征，但是把整个序列作为输入是不现实的。于是提出  SeqCaps，如图：![](image/Pasted%20image%2020230127210656.png)
首先将输入分成重叠的窗口（多个 time step），然后应用 CapNet（共享参数，也就是一个网络）得到 window capsule，然后对这些 capsule 进行 routing 得到 window emo-capsule，最后转换成 window emo-vector $\boldsymbol{o}$，其包含一个窗口中所有 $N$ 个 emo-capsule 的方向和长度：$$\boldsymbol{o}=\left[\boldsymbol{v}_1^T, \ldots, \boldsymbol{v}_N^T,\left\|\boldsymbol{v}_1\right\|, \ldots,\left\|\boldsymbol{v}_N\right\|\right]$$
所有的 emo-vector 再进行 routing 来激活 utterance capsule。

### Recurrent Capsules (RecCaps)
语音的时间信息包含了很多情感识别的线索，在routing算法中引入循环连接来提高时间建模能力，定义在 window $t-1$ 的第 $l+1$ 层的第 $j$ 个capsule 为 $\boldsymbol{v}_{t-1, j}$，在 window $t$ 的第 $l$ 层的第 $i$ 个capsule 为 $\boldsymbol{u}_{t, i}$，则 window $t$ 的预测向量 $\hat{\boldsymbol{u}}_{t, j \mid i}$ 计算为：$$\hat{\boldsymbol{u}}_{t, j \mid i}=\boldsymbol{W}_{i j}^u \boldsymbol{u}_{t, i}+\boldsymbol{W}_{i j}^o \boldsymbol{o}_{t-1}+\boldsymbol{b}_{i j}$$
其中，$$\boldsymbol{o}_{t-1}=\left[\boldsymbol{v}_{t-1,1}^T, \ldots, \boldsymbol{v}_{t-1, N}^T,\left\|\boldsymbol{v}_{t-1,1}\right\|, \ldots,\left\|\boldsymbol{v}_{t-1, N}\right\|\right]$$
从而前一个 window 中的空间信息可以用来帮助确定耦合系数和进行激活。


## 系统结构

使用频谱图作为输入，使用两个CNN进行特征提取，如图：
![](image/Pasted%20image%2020230127210457.png)


前面的都是用 CNN 和 Max pooling 进行特征提取，右边下面是使用 CNN+GRU 的对照，上面是 SeqCap，其输出包含四个16维的向量，然后输入到 dense layer 进行分类，如果两个一起用就是 CNN-GRU-SeqCap

## 结论
CapsNets在中性、愤怒和悲伤情绪方面的表现更好，而在快乐类别中的表现更差（快乐的情绪更难识别）。

![](image/Pasted%20image%2020230127213652.png)

数值越大越好。

CapsNets擅长捕捉详细的空间信息以区分情绪，RNN具有很强的捕获长上下文特征的能力，CNN GRU SeqCap 结合CapsNets和RNN的优势实现了最佳性能。