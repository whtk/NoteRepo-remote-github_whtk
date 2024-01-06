> Neosapience，ICASSP，2019

1. 提出用于 emotional 和 expressive speech synthesis 网络的 prosody embeddings
2. 在 embedding 网络中引入 temporal structure，从而可以合成语音的 speaking style
3. 通过调整学习到的 prosody features，可以在 frame level 和 phoneme level 调整语音的 pitch 和 amplitude
4. 也引入了 prosody embeddings temporal normalization，在韵律迁移中鲁棒性更好

> 本质就是提出了两种方法，将可变长度的 prosody embedding 引入到 Tacotron 中。

## Introduction

1. 之前的 prosody 模型有两个 limitation：
	1. 控制某个特定时段的 prosody 是不清晰的，固定长度的 embedding 很难实现 fine-grained prosody control
	2. 当 source speaker  和 target speaker 的 pith range 相差很大时，跨说话人的韵律迁移并不鲁棒
2. 引入两种可变长度的 prosody embedding，分别和 reference speech 或 input text 的长度相同，同时表明 normalizing prosody embedding  可以提高韵律迁移的鲁棒性

## 相关工作（略）

## baseline

用 [Tacotron 2- Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions 笔记](../Tacotron%202-%20Natural%20TTS%20Synthesis%20by%20Conditioning%20WaveNet%20on%20Mel%20Spectrogram%20Predictions%20笔记.md) 的简化版作为 encoder 和 decoder，但是用的 原始的 Tacotron 中的 post-processing net 和  Griffin-Lim 算法。

encoder 输入 $x$ 为 phoneme 序列，通过 embedding lookup layer 将 one-hot speaker identity 转换为 speaker embedding：
$$\begin{aligned}
e_{1: l_e} & =\operatorname{Encoder}\left(x_{1: l_e}\right) \\
\alpha_i & =\operatorname{Attention}\left(e_{1: l_e}, d_{i-1}\right) \\
e_i^{\prime} & =\Sigma_j \alpha_{i j} e_j \\
d_i & =\operatorname{Decoder}\left(e_i^{\prime}, s\right)
\end{aligned}$$
其中，$e,p,d$ 分别表示 text encoder state，可变长度的 prosody embedding 和 decoder state。

Reference speech 通过 reference encoder 得到 prosody embedding。具体来说，将 reference speech 的 mel 谱 通过 2D 卷积，最后一层卷积的输出通过单向的 GRU，GRU 的最后的输出 $r_N$ 即为固定长度的 embedding $p$，但是如果用 GRU 每个 time step 的输出 $r_{1:N}$ ，就可以可变长度的 prosody embedding $p_{1:N}$。

## 方法

Fine-grained prosody control 可以调整可变长度的 prosody embedding，提出两种控制方法：
+ speech side control：在 encoder 模型中把 prosody embedding 作为 condition
+ text side control：在 decoder 模型中把 prosody embedding 作为 condition

### Reference encoder

采用 CoordConv 作为第一层卷积层，采用 ReLU 作为激活函数，其他的都一样。训练也和 Tacotron 一致。损失是 target spectrogram 和 预测的 spectrogram 之间的 L1 loss。

### speech side prosody  control

如果 reference spectrogram 的长度为 $l_{ref}$，那么可变长度的 prosody embedding 的长度 $l_p$ 就等于 $l_{ref}$，而 decoder 生成的 spectrogram 的长度应该和 reference spectrogram 的长度一样，所以这三个的长度都是一样的，不用特别对齐。

在 decoder 的 time step $i$，$p_i$ 通过 attention 和 $e_{1:l_e}$ 计算权重向量 $\alpha_i$ ，然后对 $e_{1:l_e}$ 加权得到 context vector $e_i^\prime$，decoder 在 $i$ 时刻的输入为 $\left\{e_i^{\prime}, p_i, s\right\}$ 拼接。总的计算如下：
$$\begin{aligned}
e_{1:l_e}& =\operatorname{Encoder}(x_{1:l_e})  \\
\alpha_i& =\text{Attention}(e_{1:l_e},p_i,d_{i-1})  \\
e_i^\prime& =\Sigma_j\alpha_{ij}e_j  \\
d_i& =\text{Decoder}(e_i^{\prime},p_i,s) 
\end{aligned}$$

### text side prosody control

在 encoder 端，$l_e$ 和 $l_p$ 的长度不能确保一致，于是引入一个 reference attention 模块，采用 scaled dot-product attention 来找到 $e_{1:l_e}$ 和 $p_{1:l_{ref}}$ 之间的对齐。attention 的 key 和 value 来自 $p$，query 来自 $e$。

为了获得来自 prosody embedding 的 key 和 value $v$，将输出 $h$ 的维度翻倍，然后分成两个大小为 $l_{ref}\times h$ 的矩阵，计算 attention 然后对 $v_{1:l_{ref}}$ 加权后得到 text side prosody embedding $p^t$，最后和 $e$ 进行拼接：
$$\begin{aligned}
e_{1:l_e}& =\operatorname{Encoder}(x_{1:l_e})  \\
\begin{bmatrix}\kappa_{1:l_{ref}};\upsilon_{1:l_{ref}}\end{bmatrix}& =p_{1:l_{ref}}  \\
\beta_{j}& =\operatorname{Ref-Attention}(e_j,\kappa_{1:l_{ref}})  \\
p_{j}^{t}& =\Sigma_k\beta_{jk}v_k  \\
\alpha_i& =\text{Attention}(\begin{bmatrix}e_{1:l_e};p_{1:l_e}^t\end{bmatrix},d_{i-1})  \\
e_i^\prime& =\Sigma_j\alpha_{ij}\begin{bmatrix}e_j;p_j^t\end{bmatrix}  \\
d_i& =\text{Decoder}(e_i^{\prime},s) 
\end{aligned}$$
解释如下：
![](image/Pasted%20image%2020230908115154.png)

### prosody normalization

prosody embedding 使用每个 speaker 的 prosody 均值进行归一化，训练的时候，沿着时间维度计算均值。

## 实验和结果
