> Squeeze-and-Excitation Networks 论文笔记

1. 提出一种新的架构，Squeeze-and-Excitation 模块，通过显式地建模信道之间的相互依赖，自适应地重新校准信道特征响应
2. SE 模块以最小的额外计算带来了显著的性能提升

## Introduction

本文从信道关系的视角，引入新的结构——SE 模块，通过显示的构建卷积特征信道之间的相互依赖来提高网络的表征能力
	1. 提出一种机制使得网络可以进行 feature recalibration，通过学习全局信息来选择性地突出信息量大的特征，抑制无用的特征

SE 模块如图：![](./image/Pasted%20image%2020221128145805.png)
对于给定的变换（如卷积变换） $\mathbf{F}_{t r}: \mathbf{X} \rightarrow \mathbf{U}$，其中 $\mathbf{X} \in \mathbb{R}^{H^{\prime} \times W^{\prime} \times C^{\prime}}, \mathbf{U} \in \mathbb{R}^{H \times W \times C}$ ，通过以下操作来构造一个 SE 模型进行 feature recalibration：
1. feature $\mathbf{U}$ 通过一个 squeeze 操作，在空域 $H\times W$ 维度聚合 feature map 来产生 channel descriptor，这个 descriptor 集成了 channel-wise的全局特征响应，使得网络中的 全局 感受野的信息可以被低层的网络利用
2. 然后通过 excitation 操作，其中通过基于通道依赖性的自选通机制为每个通道学习的样本特定激活控制每个通道的激发（in which sample-specific activations, learned for each channel by a self-gating mechanism based on channel dependence, govern the excitation of each channel）
> 就是说，每个 channel 都有一个 excitation value，且这个 value 的计算和样本、channel 之间的关系都有关
3. 最后 feature map $\mathbf{U}$ 进行 reweighted 得到最后的输出


## 相关工作（略）

## SE 模块

设 $\mathbf{F}_{t r}$ 代表卷积操作，$\mathbf{V}=\left[\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_C\right]$ 为一系列的 filter kernels，那么输出 $\mathbf{U}=\left[\mathbf{u}_1, \mathbf{u}_2, \ldots, \mathbf{u}_C\right]$ 中的元素有：$$\mathbf{u}_c=\mathbf{v}_c * \mathbf{X}=\sum_{s=1}^{C^{\prime}} \mathbf{v}_c^s * \mathbf{x}^s$$
其中，$*$ 代表卷积，$\mathbf{v}_c=\left[\mathbf{v}_c^1, \mathbf{v}_c^2, \ldots, \mathbf{v}_c^{C^{\prime}}\right],\mathbf{X}=\left[\mathbf{x}^1, \mathbf{x}^2, \ldots, \mathbf{x}^{C^{\prime}}\right]$ ，$\mathbf{V}_c^s$ 为二维 空间 kernel（其维度也就是 长乘宽）。由于输出是通过所有通道的求和产生的，因此通道依赖性隐式嵌入在 $\mathbf{v}_C$ 中，但是这些 dependencies 和 滤波器捕获的空间相关性耦合在一起了。  

目标是确保网络能够提高其对信息特征的敏感性，以便后续变换能够利用这些特征，并抑制不太有用的特征，于是提出，通过显式建模信道相关性来实现这一点，以便在将滤波器响应输入下一个转换之前，分两个步骤（squeeze and excitation）重新校准滤波器响应。

> channel 之间是有相关性的（dependencies），这也是作者打算建模和提取的东西，但是由于一般的卷积操作直接对所有 channel 的结果进行求和，从而导致 channel dependencies 被淹没。于是通过显示的建模来实现，也就是所谓的 SE 模块。

### Squeeze：全局信息嵌入（Global Information Embedding）

在卷积操作中，信号和滤波器之间的计算都限定在局部感受野中进行计算（也就是对应滤波器大小的范围）而无法利用这之外的信息，这种情况在低层网络中尤为突出，因为通常低层的感受野（kernel size）较小。

squeeze 操作用于缓解这个问题，把 global spatial 信息都聚合在一个 所谓的 channel descriptor 中，也就是最终得到一个 $\mathbf{z} \in \mathbb{R}^C$ 向量，其中每个通道 $z_c$ 都聚合了 spatial 的信息：$$z_c=\mathbf{F}_{s q}\left(\mathbf{u}_c\right)=\frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W u_c(i, j)$$
> 其实就是直接求平均。。。

### Excitation：自适应校准（Adaptive Recalibration）

为了利用 squeeze 中得到的信息，excitation 操作用于捕获 channel-wise dependencies，且满足：
+ 能够学习通道之间的非线性相互作用
+ 能够学习 non-mutually-exclusive 关系，也就说，可以同时 emphasised 多个通道而不是像 one-hot activation 一样只能 emphasised 一个 channel

使用基于 Sigmoid 激活函数的选通机制来实现：$$\mathbf{s}=\mathbf{F}_{e x}(\mathbf{z}, \mathbf{W})=\sigma(g(\mathbf{z}, \mathbf{W}))=\sigma\left(\mathbf{W}_2 \delta\left(\mathbf{W}_1 \mathbf{z}\right)\right)$$
其中，$\delta$ 为 ReLU，$\mathbf{W}_1 \in \mathbb{R}^{\frac{C}{r} \times C},\mathbf{W}_2 \in \mathbb{R}^{C \times \frac{C}{r}}$ ，为了减少参数，使用两个 FC 层实现选通机制，且参数矩阵进行了 $r$ 倍的降维，最后使用 $\mathbf{s}$ 来激活输出得到最终的输出：$$\widetilde{\mathbf{x}}_c=\mathbf{F}_{\text {scale }}\left(\mathbf{u}_c, s_c\right)=s_c \cdot \mathbf{u}_c$$注意到，$\mathbf{u}_c \in\mathbb{R}^{H \times W}$ ，$s_c$ 为标量。

### 示例：将 SE 模块用在 Inception 和 ResNet 中

如图：![](./image/Pasted%20image%2020221128160711.png)
注意这里不要和 residual 搞混了。
> SE 模块与其说是一个 module ，不如说是一种 operation，或者门控机制，他不会改变输入输出的维度，只是对数据进行 scaling。


## 计算复杂度（略）

## 实现（略）

## 实验（略）