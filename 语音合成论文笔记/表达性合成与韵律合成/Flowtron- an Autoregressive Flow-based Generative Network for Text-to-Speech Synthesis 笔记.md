> NVIDIA，ICLR，2021

1. 提出 Flowtron，是一种自回归的，flow-based TTS 模型，可以控制语音中的变量，进行风格迁移
2. Flowtron 可以学习从 数据到 latent space 的可逆变换，latent space 可以控制语音合成的多个方面（pitch、tone、speech rate、cadence、accent）
3. MOS 可以匹配 SOTA 的 TTS 

## Introduction

1. 像 [Hierarchical Generative Modeling for Controllable Speech Synthesis 笔记](Hierarchical%20Generative%20Modeling%20for%20Controllable%20Speech%20Synthesis%20笔记.md) 和 [Style Tokens- Unsupervised Style Modeling, Control and Transfer inEnd-to-End Speech Synthesis 笔记](Style%20Tokens-%20Unsupervised%20Style%20Modeling,%20Control%20and%20Transfer%20inEnd-to-End%20Speech%20Synthesis%20笔记.md) 都通过学习一组 embedding 来控制非文本的信息，但是这些方法需要对 embedding 的维度事先假定，且不能保证包含所有的非文本信息，并且关注的都是固定长度的 embedding，而且不能控制  degree of variability
2. 提出 Flowtron，学习从 mel 谱 到 latent space $z$ 之间的可逆变换；然后就可以从 0 均值的 spherical Gaussian 先验中采样，通过调整方差来控制 variation 的数量
3. 通过最大化似然，Flowtron 可以生成 sharp mel-spectrogram 而不需要额外的 pre-net 或 post-net，也不需要额外的损失函数。

## 相关工作（略）

## Flowtron

Flowtron 是一个自回归模型，通过基于前面时刻的序列来产生当前序列，$\begin{aligned}p(x)=\prod p(x_t|x_{1:t-1})\end{aligned}$，考虑两个和 mel 谱维度相同的采样分布 $p(z)$：
+ 零均值的 spherical Gaussian
+ 混合 spherical Gaussian，其参数是固定或者可学习的

分别对应下式：
$$\begin{gathered}z\sim\mathcal{N}(z;0,\boldsymbol{I})\\z\sim\sum_k\hat{\phi}_k\mathcal{N}(z;\hat{\mu}_k,\hat{\Sigma}_k)\end{gathered}$$
采样值通过一系列可逆变换 $f$，将 $p(z)$ 转换为 $p(x)$：
$$\boldsymbol{x}=\boldsymbol{f}_0\circ\boldsymbol{f}_1\circ\ldots\boldsymbol{f}_k(\boldsymbol{z})$$
而对于自回归的 normalizing flow，第 $t$ 时刻的变量 $z_t^\prime$ 基于前面的 time step $z_{1:t_{t-1}}$：
$$z_t^{\prime}=f_k(z_{1:t-1})$$
而且由于是自回归的，变换 $f$ 的雅各比矩阵是下三角的，方便计算。然后通过最大化数据的对数似然训练 flowtron：
$$\begin{gathered}\log p_\theta(\boldsymbol{x})=\log p_\theta(\boldsymbol{z})+\sum_{i=1}^k\log|\det(\boldsymbol{J}(\boldsymbol{f}_i^{-1}(\boldsymbol{x})))|\\\boldsymbol{z}=\boldsymbol{f}_k^{-1}\circ\boldsymbol{f}_{k-1}^{-1}\circ\ldots\boldsymbol{f}_0^{-1}(\boldsymbol{x})\end{gathered}$$
在 forward pass 时，把 mel-spectrogram 视作 vector，然后通过以 text 和 speaker id 为条件的几步的 flow 过程。

其中的 flow 包含 affine coupling layer。

### Affine Coupling Layer

采用 RealNVP 中的 affine coupling layer，描述为：
$$\begin{gathered}(\log\boldsymbol{s}_t,\boldsymbol{b}_t)=NN(\boldsymbol{x}_{1:t-1},\boldsymbol{text},\boldsymbol{speaker})\\\boldsymbol{x}_t^{\prime}=\boldsymbol{s}_t\odot\boldsymbol{x}_t+\boldsymbol{b}_t\end{gathered}$$
其中，$NN()$ 表示自回归的因果神经网络。而逆函数的雅各比矩阵为：
$$\log|\det(\boldsymbol{J}(\boldsymbol{f}_{coupling}^{-1}(\boldsymbol{x})))|=\log|\boldsymbol{s}|$$
在 flow 的每个 time step 反转顺序。

### 模型架构

text encoder 来自 Tacotron，把 batch norm 替换为 instance norm。

decoder 和 NN 架构如图：
![](image/Pasted%20image%2020230908194033.png)

移除了 Tacotron 中的 pre-net 和 post-net。

采用单个 speaker embedding 然后 channel-wise 拼接到 encoder 的输出。

### 推理

