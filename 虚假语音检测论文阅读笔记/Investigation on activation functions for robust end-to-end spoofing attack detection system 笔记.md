> 就这？？？？？？？？？？？？？？？？？

1. 激活函数的选择会影响端到端 欺骗检测 聚焦于特征图的相关区域的能力，本文研究了不同激活函数在反欺骗任务中的影响
2. 结果发现，不同的激活函数使系统能够学习用于欺骗检测的补充信息，同时建议在端到端系统将不同激活函数的输出汇集在一起以进行互补

## Introduction

1. [[One-class Learning Towards Synthetic Voice Spoofing Detection 笔记]] 提出了具有ResNet18架构的 one class softmax loss ，性能优于经典的基于 GMM 的方法
2. 有论文提出了 arelu，基于注意力的激活，以有效地关注特征图的相关区域
3. 本文 研究了各种激活函数对端到端欺骗检测系统性能的影响，并建议将激活集成技术应用于端到端系统
4. 实验结果表明，端到端欺骗检测可以大大受益于多个激活函数和激活集合的使用

## 端到端欺诈检测系统

提出的模型包括两个网络，embedding network + classification network。

embedding network 使用 SE-ResNet，同时为了聚合 ResNet的帧级输出，采用了 attention pooling layer 将加权的 一阶和二阶矩在时间维度进行拼接得到最终的表征。最后送到分类器中，包含一个全连接层和一个二维的 softmax 层。

结构如图：
![[Pasted image 20221222093840.png]]

目标函数为 one-class softmax ，计算为：$$L_{O C S}=-\frac{1}{N} \sum_{i=1}^N \log \left(1+e^{k\left(m_{y_i}-\hat{W}_0 \hat{\omega}_i\right)(-1)^{y_i}}\right)$$
其中，$k$ 为缩放因子，$\omega_i \in R^D$ 为 embedding，$y_i \in\{0,1\}$ 为 label，$i$ 表示样本索引，$N$ 为 batch size，$W_0$ 为 target class embedding 的 权重向量，上标 
 $\hat{}$  为归一化。

## 传统激活函数

### 不可训练的

RELU：$$f_{R e L U}\left(x_i\right)= \begin{cases}x_i & \text { if } x_i>0 \\ 0 & \text { otherwise }\end{cases}$$
RELU 可以克服梯度消失问题，加速训练。但是缺点是负轴没了。

用于改进 RELU 的有：
LeakyReLU：$$f_{\text {LeakyReLU }}\left(x_i\right)= \begin{cases}x_i & \text { if } x_i>0 \\ \gamma x_i & \text { otherwise }\end{cases}$$

ELU：$$f_{E L U}\left(x_i\right)= \begin{cases}x_i & \text { if } x_i>0 \\ r\left(e^{x_i}-1\right) & \text { otherwise }\end{cases}$$

RReLU：$$f_{R R e L U}\left(x_i\right)= \begin{cases}x_i & \text { if } x_i>0 \\ a_i x_i & \text { otherwise }\end{cases}$$
### 可训练的 

PReLU：$$f_{P R e L U}\left(x_i\right)= \begin{cases}x_i & \text { if } x_i>0 \\ \xi_i x_i & \text { otherwise }\end{cases}$$
其中，$\xi_i$ 是可训练的参数。

AReLU：通过采用可训练的注意力机制来提高相关输入特征的贡献同时抑制不相关的输入特征：$$\begin{aligned}
f_{A R e L U}\left(x_i\right) & =f_{R e L U}\left(x_i\right)+g_{a t t}\left(x_i, \alpha, \beta\right) \\
& = \begin{cases}C(\alpha) x_i & \text { if } x_i<0 \\
(1+\sigma(\beta)) x_i & \text { otherwise }\end{cases}
\end{aligned}$$
其中，$\alpha,\beta$ 为可学习的缩放因子，$C$ 为钳位操作，将数值限制在 $[0.01,0.99]$ 之间。

## 激活函数集成框架

使用多个激活函数，求和来汇聚输出：$$f_{e n s}\left(x_i\right)=\sum_{j=1}^J f_j\left(x_i\right)$$
$j$ 为激活函数的标号，一共有 $J$ 个。如图：![[Pasted image 20221222100914.png]]

## 实验

不同激活函数参数为：![[Pasted image 20221222101038.png]]


结果：![[Pasted image 20221222101103.png]]
1. 可学习的激活函数比不可学习的好
2. AReLU 的效果最好

融合结果：![[Pasted image 20221222101550.png]]
1. RELU 和 可学习的激活函数进行集成可以提高性能
2. ReLU、AReLU、PReLU、LeakyReLU和ELU 组合是最好的

