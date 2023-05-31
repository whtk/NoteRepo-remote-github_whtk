1. 本文提出了一种基于注意力的单个卷积神经网络来学习鉴别特征以进行欺骗检测；其关键思想是减少信道间的信息冗余，并将重点放在语音表示的信息最丰富的子带上。
2. ASVspoof 2019 评估集 EER 为 1.87% 


## Introduction
1. 研究表明，没有一个前端能够可靠地检测到不同欺骗攻击所产生的全部伪影
2. 研究表明，并非所有频带都有助于检测欺骗，只有使用在相同频带中具有高频谱分辨率的前端才能可靠地检测到这些欺骗
3. 沿频率轴的T-F谱图中存在非局部相关性，但简单叠加几个2D卷积层无法捕获这种全局相关性。
4. 本文提出了一种基于注意力的欺骗检测系统。基于注意力的方法已经被用于重放攻击检测[[End-To-End Audio Replay Attack Detection Using Deep Convolutional Networks with Attention 笔记]]、[[Attentive Filtering Networks for Audio Replay Attack Detection 笔记]]
5. 提出的注意力模块包含两个块：频率注意力块（FAB）和通道注意力块（CAB）。频率关注块的目的是学习T-F谱图中沿频率轴的非局部相关性，并关注基本子带，而信道关注块的目标是学习信道间关系，以减少信道间的信息冗余。
6. 在ASVspoof 2019 LA 中优于单个系统

## 检测方法

### 总体架构
![[Pasted image 20221027111541.png]]
可以看到，作者将注意力模块添加到ResNet结构的每个 Residual bottleneck的顶部。

给定输入特征图 $F_i$，注意力模块首先通过矩阵相乘生成频率间的关系矩阵和加权特征，然后通过逐元素求和生成频率细化特征 $F_f$，类似方法可以生成信道细化特征 $F_c$ 。同时使用 attentive temporal pooling 对输入的一些片段分配更高的重要性。


### FAB
由于并非所有频带都有助于欺骗任务，可能导致信息冗余甚至过拟合。

将FAB插入每个残差块的顶部，以使模型具有全频率感受野并聚焦于信息最丰富的频带。FAB 结构如下：
![[Pasted image 20221027113502.png]]
输入特征 $F_i \in R^{C \times F \times T}$，首先在时间和channel维度进行压缩，得到特征向量 $F_f^{a v g}$  和 $F_f^{\max }$，concate 之后得到 $F_f^{c a t}$，然后通过 2D 1x1 卷积，得到 $F_f^{p o o l}=\operatorname{Conv}\left(F_f^{a v g}(\mathrm{c}) F_f^{\max }\right)$，其维度为 $F_f^{\text {pool }} \in R^{1 \times F \times 1}$，$c$ 代表 concate。然后计算向量 $F_f^{p o o l}$ 的自相关矩阵 $A_f=\operatorname{Softmax}\left(F_f^{\text {pool }} \otimes\left(F_f^{p o o l}\right)^{\mathrm{T}}\right)$，最后将自相关矩阵和 输入特征 $F_i$ 相乘然后执行残差连接：
$$F_f=F_i \oplus\left(\alpha \times\left(A_f \otimes F_i\right)\right)$$
得到的 $F_f \in R^{C \times F \times T}$ ，其中 $\alpha$ 是一个可学习参数，初始化为0，以减少前几个训练周期中收敛过程的难度。

### CAB

标准卷积网络通道数随着网络加深而增加，从而导致信息冗余。CAB 用于学习特征图通道间的关系，如下图：
![[Pasted image 20221027222124.png]]
和 FAB 中相同，首先沿频率和channel 轴分别使用 MaxPool 和 AvgPool ，讲得到的结果相加，然后进行卷积运算，通过 softmax 之后得到信道自相关矩阵：
$$A_c=\operatorname{Softmax}\left(\operatorname{Conv}\left(F_c^{a v g} \oplus F_c^{\max }\right)\right)$$
最后也和 CAB 一样，有：
$$F_f=F_i \oplus\left(\beta \times\left(A_c \otimes F_i\right)\right)$$

### 注意力模块集成设计
两个块的组合方式有：
1. Sequential，顺序组合
2. Seq-inversed，反序组合
3. Parallel，并行组合

如图：![[Pasted image 20221027222636.png]]


## 实验

### 参数

输入：对数功率谱

fft：512
帧长：15 ms，帧移：10 ms
固定输入为 750 帧

最终输入维度：750*257，（257是 frequency bins），且输入信号不进行 VAD 或 去混响等预处理操作

采用 [[Generalized end-to-end detection of spoofing attacks to automatic speaker recognizers 笔记]] 中的网络结构，基于 ResNet18，其中全局avg pool 层被attentive temporal pool 层取代，基于 softmax 损失进行训练。

Adam 优化器，$\beta_1=0.9, \beta_2=0.999$，batch size 为 32，初始 $lr=0.0003$，每 10 个 epoch 减半。训练 100 个 epoch。

### 结果

在开发和评估集上的结果如图：
![[Pasted image 20221028110514.png]]
总结：无论是与FAB还是CAB聚合，两者的评估结果都取得了比基线更好的性能；同时聚合两个注意力时，采用顺序聚合的效果优于其他两种方法。

和其他系统对比的结果：
![[Pasted image 20221028111433.png]]

总结：单系统性能最好，融合效果也不赖。
