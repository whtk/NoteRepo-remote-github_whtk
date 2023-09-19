> Kakao Enterprise，首尔大学，NIPS 2020

1. FastSpeech 和 ParaNet 可以并行生成 mel 谱
2. 但是没有自回归模型 TTS 模型作为外部 aligner 来引导时，并行 TTS 模型无法训练
3. 提出 Glow-TTS，基于 flow 的并行 TTS 模型，不需要任何外部对齐器，通过结合 flow 和 动态编程，模型可以搜索 文本 和 latent representation 之间最可能的  monotonic alignment，可以实现快速、多样性、可控的生成
4. 合成质量和 Tacotron 2 差不多

## Introduction

1. Tacotron 2 和 Transformer TTS 等自回归模型已经有 SOTA 的性能，但是速度慢；且大部分的自回归模型都缺乏鲁棒性
2. 最近提出的并行模型都使用 external aligners 或 预训练的自回归 TTS 模型来提取 attention map，模型性能很大程度上依赖这些 aligners
3. 提出 Glow-TTS，不需要外部的 aligners，可以在内部学习对齐；且发现强制的 hard monotonic alignment 可以得到更鲁棒的 TTS，可以泛化到长语音中
4. Glow-TTS 可以比 Tacotron 2 快 15.7 倍，性能也超过它，尤其是当输入语音很长的时候；同时还可以拓展到多说话人情况

## 相关工作（略）

## Glow-TTS

受人类按顺序阅读不会跳字的启发，Glow-TTS  以文本和语音表之间的 monotonic and non-skipping alignment 为条件。其训练和推理过程如图：
![](image/Pasted%20image%2020230825194246.png)

### 训练和推理过程

Glow-TTS 建模（mel 谱）条件分布 $P_X(x \mid c)$，此过程是用 flow-based decoder $f_{dec}:z\rightarrow x$ 将条件先验分布 $P_Z(z \mid c)$ 转换得到的。其中 $x,c$ 分别表示输入的 mel 谱 和文本序列，用概率论的知识，其对数似然为：
$$\log P_X(x \mid c)=\log P_Z(z \mid c)+\log \left|\operatorname{det} \frac{\partial f_{d e c}^{-1}(x)}{\partial x}\right|$$
用 $\theta$ 表示神经网络的参数，用 $A$ 表示对齐函数。$P_Z$ 为各相同性多元高斯分布，其均值和方差通过 text encoder $f_{enc}$ 获得。text encoder 是将 text $c=c_{1:T_{text}}$ 映射为统计参数 $\mu=\mu_{1:T{text}}$ 和 $\sigma=\sigma_{1:T{text}}$，其中 $T_{text}$ 为文本长度。对齐函数 $A$ 表示从语音表征的索引到 text encoder 的统计参数的索引的映射：$A(j)=i, z_j\sim{N}(z_j;\mu_i,\sigma_i)$，且映射函数 $A$ 是单调且满射的
> 满射的意思是，对于每个 $z_j$ 都有一个或多个 $\mu_i,\sigma_i$ 与之对应。
> 单调性确保没有单词被重复，满射性确保没有单词被跳过。

此时先验分布可以表示为：
$$\log P_Z(z \mid c ; \theta, A)=\sum_{j=1}^{T_{m e l}} \log \mathcal{N}\left(z_j ; \mu_{A(j)}, \sigma_{A(j)}\right)$$
目标是找到参数 $\theta$ 和对齐函数  $A$ 最大化上述似然。但是其计算困难，于是分解成两个子问题：
+ 问题1：给定 $\theta$ 下找到最可能的对齐 $A^*$
+ 问题2：再通过最大化似然 $\log p_X\left(x \mid c ; \theta, A^*\right)$ 来更新参数 $\theta$

修改后的目标函数不能确保全局最优解，但是是全局解的一个下界：
$$\begin{gathered}
\max _{\theta, A} L(\theta, A)=\max _{\theta, A} \log P_X(x \mid c ; A, \theta) \\
A^*=\underset{A}{\arg \max } \log P_X(x \mid c ; A, \theta)=\underset{A}{\arg \max } \sum_{j=1}^{T_{m e l}} \log \mathcal{N}\left(z_j ; \mu_{A(j)}, \sigma_{A(j)}\right)
\end{gathered}$$
为了解决问题1，引入  alignment  搜索算法，monotonic alignment search (MAS)。

推理时为了估计 $A^*$，训练一个 duration predictor $f_{dur}$ 来匹配从 $A^*$ 中计算得到的 duration label。用的是 FastSpeech 的架构，将  duration predictor 放在 text encoder 的上面，在对数域采用 MSE 进行训练。还采用了 stop gradient 算子 $sg[\cdot]$，移除了backward pass 过程中输入的梯度。duration predictor 的损失为：
$$\begin{gathered}
d_i=\sum_{j=1}^{T_{m e l}} 1_{A^*(j)=i}, i=1, \ldots, T_{\text {text }} \\
L_{d u r}=M S E\left(f_{d u r}\left(s g\left[f_{\text {enc }}(c)\right]\right), d\right)
\end{gathered}$$
> stop gradient 算子的目的是只这部分的损失只更新 duration predictor，而不更新 text encoder。

推理时，通过 text encoder 得到 $\mu,\sigma$，通过 duration predictor 得到 alignment $A$，然后从先验分布中采样得到 latent variable，通过将此 latent variable 送入到 flow-based decoder 中生成 mel 谱。
> 训练的时候其实就是把 MAS 得到的结果作为 duration label 来训练 duration predictor

### Monotonic Alignment Search

MAS 搜索在 latent variable（来自语音） 和 先验分布的统计参数（来自文本）之间最可能的 alignment，下图给出了一个示例：
![](image/Pasted%20image%2020230825213215.png)

令 $Q_{i,j}$ 为最大对数似然，$i,j$ 分别代表先验分布的统计参数 和 latent variable 的第 $i$ 和 第 $j$ 个元素，那么 $Q_{i,j}$ 可以 recursively 写为 $Q_{i-1,j-1}$
和 $Q_{i,j-1}$，因为如果 $z_j$ 和 $\{\mu_i,\sigma_i\}$ 对齐，那么 $z_{j-1}$ 要么对应到 $\{\mu_i,\sigma_i\}$ 要么对应到 $\{\mu_{i-1},\sigma_{i-1}\}$：
$$Q_{i, j}=\max _A \sum_{k=1}^j \log \mathcal{N}\left(z_k ; \mu_{A(k)}, \sigma_{A(k)}\right)=\max \left(Q_{i-1, j-1}, Q_{i, j-1}\right)+\log \mathcal{N}\left(z_j ; \mu_i, \sigma_i\right)$$
通过迭代计算所有的值直到 $Q_{T_{t e x t}, T_{m e l}}$，从而 $A^*$ 对应的是使得  $Q$ 最大的那个对齐。算法复杂度为 $O\left(T_{\text {text }} \times T_{m e l}\right)$，且不需要在 GPU 上算，直接在 CPU 算就行，推理的时候也不用算。

算法描述如下：
![](image/Pasted%20image%2020230825214258.png)


### 模型架构

encoder 和 duration predictor 架构如下：
![](image/Pasted%20image%2020230825214400.png)
用的是 Transformer TTS 中的 encoder，两个修改：
+ 移除了 positional encoding ，加上了 relative position representations
+ 在 encoder pre-net 中加了 residual connection
然后在 encoder 顶部加上  linear projection layer 来估计均值和方差。

duration predictor 由两个卷积层和 ReLU 激活、layer normalization,、dropout 以及最后的 projection layer 组成，结构和 FastSpeech 一样。

decoder 架构如下：
![](image/Pasted%20image%2020230825214415.png)
Glow-TTS 的核心为 flow-based decoder，训练的时候需要将 mel 谱 转换为 latent representation 以用于最大似然估计和对齐算法。推理时则需要将先验分布转换为 mel 谱 来实现高效的并行解码。因此，decoder 由一系列可以并行执行 forward 和 inverse transformation 的 flow 组成。具体来说，是 stack of multiple blocks，包含：
+ activation normalization layer
+ invertible 1x1 convolution layer
+ affine coupling layer（和 WaveGlow 差不多，但是没有用  local conditioning）

## 实验（略）