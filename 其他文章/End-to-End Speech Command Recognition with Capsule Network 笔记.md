> interspeech 2018

1. 使用 capsule network 在频率轴和时间轴上捕获谱图特征的空间关系和姿态关系
2. 提出的SR模型在 1s 的语言命令识别中都获得了更好的结果

## Introduction

1. LSTM+CTC 结合能够在普通的 SR 中实现最好的性能
2. 对于短时间依赖的任务，如关键字识别，还是有很多基于 CNN 的 SR 
3. 但是 NN 的问题在于无法捕获底层特征之间的空间关系
4. 作者认为，语言领域中的特征之间的空间关系也很重要，采用 capsule network 来考虑特征之间的空间关系和姿态信息

## capsule network（略）

## 模型

### baseline CNN

三个卷积层+两个 FC 层，第一个卷积之后添加 max pooling，最后一个FC之后是一个大小为 30类 的FC，前面的层用了 dropout，最后是 softmax。

### capsule network
![](./image/Pasted%20image%2020230131111406.png)

使用的结构如上图：一层卷积层（带ReLU激活）、一层输入 capsule ，一层输出 capsule，输入是谱图，输出是 $256\times T\times F$，然后以输入 capsule 的向量长度（也就是 $8$）和 stride=2 进行卷积，卷积次数为 capsule channel（也就是 $64$） 的大小（但是实际是卷积一次，然后卷积的 channel 为capsule channel 的大小 ），最终得到的tensor的维度是 $64\times 8\times \frac{T}{2} \times \frac{F}{2}$。

然后进行 routing 的计算，输出的 capsule 层的 capsule 数量为分类的数量（也就是 $30$）。

## 实验

数据集：speech commands dataset，还提供了背景噪声文件

特征：40维的，delta 和delta-deltas 的 mel 滤波器组系数，归一化到零均值和单位方差

损失：margin loss 
adam 优化器，learning rate 0.001

**实际实验中发现 decoder 网络好像没啥用，于是删掉了。**

![](./image/Pasted%20image%2020230131113138.png)

把 capsule 的 channel 变成 16（前面的都是 64，主要目的是为了减少参数），不同配置的结果：
![](./image/Pasted%20image%2020230131113309.png)

反正不管怎样都比 CNN 好。