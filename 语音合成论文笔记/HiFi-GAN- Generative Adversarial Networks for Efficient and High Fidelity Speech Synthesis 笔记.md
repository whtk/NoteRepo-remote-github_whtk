> Kakao，NIPS 2020

1. 现有的 GAN 波形生成还是无法达到自回归或者 flow-based 模型的效果
1. 提出 HiFi-GAN，兼顾高效和高保真度
2. 表明，对音频的周期进行建模对于提高输出样本质量至关重要
3. 输出 22.05 kHz 音频条件下，速度超过 real time 167.9倍（single V100 GPU）

> HiFi-GAN 相比于 MelGAN 的改进为：
> 1. 改进了 generator，提出 MRF，提高波形的生成能力
> 2. 改进了 discriminator，多用了一个 MDP 判别器
> 3. 

## Introduction

1. 大多数的语音合成模块有两步：
   1. 预测一个中间表征，如 mel 谱或者 linguistic features
   2. 从中间表征合成音频，即本文所关注的
2. 现有的模型：
	1. WaveNet，是自回归的，很慢
	2. Flow-based 生成模型，如Parallel WaveNet、WaveGlow，参数多
	3. 基于GAN的，如MelGAN、GAN-TTS，效果不如前两者
3. 提出的 HiFi-GAN 兼顾高效和高保真度（比 AR 或者 flow-based 模型还要好）；提出 discriminator 包含子 discriminator，每个子 discriminator 都提取波形的特定周期部分
4. MOS 比 WaveNet 和 WaveGlow 高

## HiFi-GAN 结构

### 总览

包含一个生成器，两个判别器（multi-scale 和 multi-period），加入两个额外的损失提高训练稳定性和性能。

### 生成器

生成器为全卷积网络。输入为 Mel 谱，转置卷积进行上采样，直到输出序列的长度与原始波形的时间分辨率相匹配。生成器由 $|k_u|$ 个层叠的转置卷积和多感受野融合（MRF，multi- receptive field fusion）模块组成：
![](image/Pasted%20image%2020230928112151.png)

#### MRF 模块

MRF 返回多个残差块的输出和，每个残差块有不同的 kernel size 和 dilation rate，每个残差块又由多个 LeakyRelu 和 Conv1d 层组成。

### 判别器

为了识别语音中不同的周期模式，提出了多周期判别器（MPD），包含几个子判别器，每个子判别器处理输入音频周期的一部分。而为了捕获连续、长期依赖，采用了 MelGAN 提出的多尺度判别器（MSD），两者是输出都是用来判断这段信号是合成的还是真的。

#### MPD

MPD包含多个子判别器，每个子判别器输入为等距样本，设周期为 $p$（在论文中，作者取 $p$ 分别为 $2,3,5,7,11$ 来避免overlap），对于第二个MPD $p=3$，如图所示（内部是Relu+层叠的卷积）：

将长为 $T$ 的数据折叠成 $p \times \frac Tp$ 的2D数据，然后做2D卷积，且kernel size为 $k\times 1$（也就是可以保持行数不变，改变列数），最后进行权值归一化。

#### MSD

MPD 的每个子判别器只接受不相交的样本，因此采用 MSD 来评估连续语音序列。MSD 混合三个子判别器，对应于不同尺度：

![](image/Pasted%20image%2020230928112243.png)


每个 sub-discriminator 为 stack of strided convolutional layers with leaky rectified linear unit (ReLU)，同时也使用了 weight normalization。

MPD 的输入是不相邻的样本，于是用 MSD 来处理连续样本。
第二个子判别器的结构如图（来自 MelGAN）：
  
![](image/Pasted%20image%2020230928112214.png)

输入为：
+ raw audio
+ ×2 average-pooled audio
+ ×4 average-pooled audio
采用 spectral normalization 稳定训练。

## 训练损失

**对抗损失**，采用的是平方损失计算而不是交叉熵：

$$
\begin{aligned}
&\mathcal{L}_{A d v}(D ; G)=\mathbb{E}_{(x, s)}\left[(D(x)-1)^{2}+(D(G(s)))^{2}\right] \\
&\mathcal{L}_{A d v}(G ; D)=\mathbb{E}_{s}\left[(D(G(s))-1)^{2}\right]
\end{aligned}
$$

其中，$x$ 为GT音频，$s$ 为音频对应的Mel谱。

**Mel谱损失**，类似于重构损失，为生成器合成的波形的mel谱与GT波形的mel谱之间的L1距离：

$$
\mathcal{L}_{M e l}(G)=\mathbb{E}_{(x, s)}\left[\|\phi(x)-\phi(G(s))\|_{1}\right]
$$

其中，$\phi(\cdot)$ 表示求波形的mel谱，这一损失可以稳定早期的训练，提高合成音频的真实度。

**特征匹配损失**，通过GT波形和生成波形之间的判别器特征差异测量相似性，其定义为：

$$
\mathcal{L}_{F M}(G ; D)=\mathbb{E}_{(x, s)}\left[\sum_{i=1}^{T} \frac{1}{N_{i}}\left\|D^{i}(x)-D^{i}(G(s))\right\|_{1}\right]
$$

其中，其中 $T$ 表示判别器的层数；$D^i(\cdot)$ 和 $N_i$ 分别表示求判别器特征（判别器第 $i$ 层的输出）和第 $i$ 层中的特征数。

则总损失为：

$$
\begin{aligned}
&\mathcal{L}_{G}=\mathcal{L}_{A d v}(G ; D)+\lambda_{f m} \mathcal{L}_{F M}(G ; D)+\lambda_{\text {mel }} \mathcal{L}_{M e l}(G) \\
&\mathcal{L}_{D}=\mathcal{L}_{A d v}(D ; G)
\end{aligned}
$$

然后因为判别器包含多个子判别器，上式更准确的可以写成：

$$
\begin{aligned}
\mathcal{L}_{G} &=\sum_{k=1}^{K}\left[\mathcal{L}_{A d v}\left(G ; D_{k}\right)+\lambda_{f m} \mathcal{L}_{F M}\left(G ; D_{k}\right)\right]+\lambda_{m e l} \mathcal{L}_{M e l}(G) \\
\mathcal{L}_{D} &=\sum_{k=1}^{K} \mathcal{L}_{A d v}\left(D_{k} ; G\right)
\end{aligned}
$$

其中，$D_k$ 表示第 $k$ 个子MPD和MSD判别器。
