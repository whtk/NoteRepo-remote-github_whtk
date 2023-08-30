> NVIDIA，ICASSP 2019

1. 提出 WaveGlow，基于 flow 的生成模型，可以从 mel 谱 合成高质量的波形
2. 将 glow 和 wavenet 组合起来，实现快速、高效和高质量的语音合成且不需要自回归
3. 只用一个网络和一个损失函数（最大化训练数据的对数似然）就可以实现训练
4. 在 NVIDIA V100 GPU 上产生速率为 500 kHz ；MOS 好于 WaveNet

## Introduction 

1. 当前的 TTS 分为两步：
	1. 第一步，将文本转换为 time-aligned features，如 mel 谱、F0 或其他语言特征
	2. 第二步，将这些 time-aligned feature 转换为音频样本，通常称为 vocoder，在计算上很有挑战且非常影响音频质量
2. 大多数模型是自回归的，实现和训练起来简单，但是由于是连续的，不能充分利用 GPU 的并行处理特性
3. 当时的三个非自回归模型，但是很难训练，而且有额外的损失函数，还存在各种问题：
	1. Parallel WaveNet
	2. Clarinet
	3. MCNN
4. 提出 WaveGlow，将 glow 和 wavenet 组合起来，实现和训练都很简单

## WaveGlow

WaveGlow 通过从从分布中采样来生成音频。首先从一个简单分布中采样，本文用的是标准正太分布，然后将样本通过一系列的网络层，实现从简单分布到需要的分布的转换：
$$\begin{gathered}
\boldsymbol{z} \sim \mathcal{N}(\boldsymbol{z} ; 0, \boldsymbol{I}) \\
\boldsymbol{x}=\boldsymbol{f}_0 \circ \boldsymbol{f}_1 \circ \ldots \boldsymbol{f}_k(\boldsymbol{z})
\end{gathered}$$

同时基于 mel 谱 建模音频样本分布。通过直接最大化 NLL 来训练模型。

通过限制每一层的网络都是 双射 的，flow based 模型的似然可以使用变量替换定理计算为：
$$\begin{gathered}
\log p_\theta(\boldsymbol{x})=\log p_\theta(\boldsymbol{z})+\sum_{i=1}^k \log \left|\operatorname{det}\left(\boldsymbol{J}\left(\boldsymbol{f}_i^{-1}(\boldsymbol{x})\right)\right)\right| \\
\boldsymbol{z}=\boldsymbol{f}_k^{-1} \circ \boldsymbol{f}_{k-1}^{-1} \circ \ldots \boldsymbol{f}_0^{-1}(\boldsymbol{x})
\end{gathered}$$
在本文中，第一项为 标准正太分布 的对数似然，第二项来自于变量替换，$\boldsymbol{J}$ 为 Jacobian 矩阵。Jacobian 矩阵的对数行列式会使得网络层在前向过程中增加 space volume，同时还能防止网络为了优化损失而将 $\boldsymbol{x}$ 项乘以零。

模型和 Glow 非常相似，如图：
![](image/Pasted%20image%2020230826162500.png)
在 forward 过程中，把一组 8 个音频样本作为向量，称为 ”squeeze” 操作，然后将这些向量通过 steps of flow，即 invertible 1 × 1 convolution + affine coupling layer。

### Affine Coupling Layer

可逆神经网络通常使用 coupling layer 来构造，采用的是 RealNVP 中的 affine coupling layer，使用一半的 channel 对应的数据作为输入，通过模型产生 scale  和 shift 项来转换另一半的 channel 对应的数据：
$$\begin{gathered}
\boldsymbol{x}_a, \boldsymbol{x}_b=\operatorname{split}(\boldsymbol{x}) \\
(\log \boldsymbol{s}, \boldsymbol{t})=W N\left(\boldsymbol{x}_a, \text { mel-spectrogram }\right) \\
\boldsymbol{x}_b=\boldsymbol{s} \odot \boldsymbol{x}_b+\boldsymbol{t} \\
\boldsymbol{f}_{\text {coupling }}^{-1}(\boldsymbol{x})=\operatorname{concat}\left(\boldsymbol{x}_a, \boldsymbol{x}_{b^{\prime}}\right)
\end{gathered}$$
其中，$WN()$ 可以是任意变换（也就是说可以是神经网络），即使 $WN()$ 不可逆，整个算法也是可逆的。本文 $WN()$ 使用 layers of dilated convolutions with gated-tanh nonlinearities，再加 residual connections and skip connections。此架构和 WaveNet 很相似，但是没使用因果卷积。affine coupling layer 同时可以把 mel 谱 作为条件，且该层只有其中的 $\boldsymbol{s}$ 对损失有贡献（或者说影响对数似然）：
$$\log \left|\operatorname{det}\left(\boldsymbol{J}\left(\boldsymbol{f}_{\text {coupling }}^{-1}(\boldsymbol{x})\right)\right)\right|=\log |\boldsymbol{s}|$$
### 1x1 Invertible Convolution

在 Affine Coupling Layer 中，有一半的通道永远不会被修改，这是一个很大的限制，于是在 ACL 层之前使用 Glow 中的 1x1 convolution layer，权重 $W$ 通过标准正交基来初始化以确保可逆，此时 Jacobian 的对数行列式为：
$$\begin{array}{c}
\boldsymbol{f}_{\text {conv }}^{-1}=\boldsymbol{W} \boldsymbol{x} \\
\log \left|\operatorname{det}\left(\boldsymbol{J}\left(\boldsymbol{f}_{\text {conv }}^{-1}(\boldsymbol{x})\right)\right)\right|=\log |\operatorname{det} \boldsymbol{W}|
\end{array}$$
最终，总的损失为：
$$\begin{aligned}
\log p_\theta(\boldsymbol{x})=& -\frac{\boldsymbol{z}(\boldsymbol{x})^T\boldsymbol{z}(\boldsymbol{x})}{2\sigma^2}  \\
&+\sum_{j=0}^{\#coupling}\log\boldsymbol{s}_j(\boldsymbol{x},mel\text{-}spectrogram) \\
&+\sum_{k=0}^{\boldsymbol{\#}conv}\log\det|\boldsymbol{W}_k|
\end{aligned}$$
其中第一项来自于球型高斯的对数似然，$\sigma^2$ 为分布的方差。

### Early outputs

作者发现，与其让所有通道穿过所有层，不如在每 4 个 coupling layer 后输出 2 个通道到损失函数。在经过网络的所有层后，最终向量与之前输出的所有通道串联起来，形成最终的 $\boldsymbol{z}$。提前输出一些维度可以让网络更容易在多个时间尺度上添加信息，并帮助梯度传播到更早的层（类似于 skip connection）。

### 推理

完成训练后，从高斯分布中采样 $\boldsymbol{z}$ 输入网络即可得到输出。而且从一个标准偏差低于训练时假设的标准偏差的高斯中进行采样，会使音频质量略有提高。训练时用的是 $\sigma=\sqrt{0.5}$，推理时用的是 $\sigma=0.6$。

## 实验（略）