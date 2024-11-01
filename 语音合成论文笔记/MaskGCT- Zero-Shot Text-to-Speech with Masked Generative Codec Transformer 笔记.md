> prerpint 2024, CUHK、趣丸科技

1. 提出 Masked Generative Codec Transformer (MaskGCT)，非自回归 TTS，不需要显示地对齐和 phone-level 的 duration prediction
2. MaskGCT 为两阶段模型：
	1. 阶段一：采用文本预测从 SSL 模型中提取的 semantic token
	2. 阶段二：基于 semantic token 预测 acoustic token
3. 模型遵循 掩码预测 框架，训练时，模型基于 condition 或者 prompt 预测 mask 的 token，推理时，以并行的方式生成特定长度的 token
4. 用 100K h 的数据训练，在质量、相似度上超过了 SOTA，模型和代码开源

## Introduction

1. 提出 MaskGCT，包含两个阶段：
    1. 阶段一，T2S 模型：预测 masked semantic tokens，采用 text token 和 prompt 语音的 semantic token 作为 prefix ，不需要显示的 duration prediction
    2. 阶段二，S2A 模型：利用 semantic tokens 预测 masked acoustic tokens
2. 推理时，MaskGCT 可以在几个 step 的迭代中生成指定长度的 semantic tokens
3. 还训练 VQ-VAE 来量化 SSL 的 embedding，而非使用 k-means 提取 semantic tokens
4. 实验结果表明，MaskGCT 在 speech quality、similarity、prosody 和 intelligibility 上表现优于现有模型


## 相关工作（略）

## 方法

### 背景：非自回归 Masked Generative Transformer

给定离散表征序列 $\mathbf{X}$，定义 $\mathbf{X}_t = \mathbf{X} \odot \mathbf{M}_t$ 为用二进制 mask $\mathbf{M}_t = [m_{t,i}]_{i=1}^N$ 对 $\mathbf{X}$ 进行 mask，其中 $m_{t,i} = 1$ 时，用特殊的 [MASK] token 替换 $x_i$，否则保持不变。每个 $m_{t,i}$ 都是独立同分布的伯努利分布 $\gamma(t)$，$\gamma(t) \in (0, 1]$ 表示 mask schedule 函数（例如，$\gamma(t) = \sin(\frac{\pi t}{2T}), t \in (0, T]$）。定义 $\mathbf{X}_0 = \mathbf{X}$。非自回归 Masked Generative Transformer 用未 mask 的 token 和 condition $C$ 预测 mask 的 token，建模为 $p_{\theta}(\mathbf{X}_0|\mathbf{X}_t, C)$，参数 $\theta$ 通过最小化 mask token 的负对数似然进行优化：
$$\mathcal{L}_{\text{mask}}=\underset{\mathbf{X}\in\mathcal{D},t\in[0,T]}{\operatorname*{\mathbb{E}}}-\sum_{i=1}^Nm_{t,i}\cdot\log(p_\theta(x_i|\mathbf{X}_t,\mathbf{C})).$$

推理阶段，通过迭代解码并行解码 token。从 fully masked 的序列 $\mathbf{X}_T$ 开始，假设总的 step 为 $S$，对于每个 step $i$，先从 $p_{\theta}(\mathbf{X}_0|\mathbf{X}_{T-(i-1)\cdot\frac{T}{S}}, C)$ 中采样 $\hat{\mathbf{X}}_0$，然后基于 confidence score 采样 $\lfloor N\cdot\gamma(T-i\cdot\frac{T}{S})\rfloor$ 个 token 进行 remask，得到 $\mathbf{X}_{T-i\cdot\frac{T}{S}}$，其中 $N$ 是 $\mathbf{X}$ 中 token 的总数。当 $x_{T-(i-1)\cdot\frac{T}{S}, i}$ 是 [MASK] token 时，$\hat{x}_i$ 的 confidence score 赋值为 $p_{\theta}(\mathbf{X}_0|\mathbf{X}_{T-(i-1)\cdot\frac{T}{S}}, C)$；否则，confidence score 赋值为 1，表示已经 unmask 的 token 不会再次 mask。特别地，选择 confidence score 最低的 $\lfloor N\cdot\gamma(T-i\cdot\frac{T}{S})\rfloor$ 个 token 进行 mask。


## 模型框架

框架如图：
![](image/Pasted%20image%2020241101115148.png)

MaskGCT 在两个阶段都使用非自回归 Masked Generative Model，不需要显示的 text-speech 对齐 和 phone-level duration prediction：
+ 对于第一阶段模型，训练模型学习 $p_{\theta_{s1}}(\mathbf{S}|\mathbf{S}_t, (\mathbf{S}^p, \mathbf{P}))$，其中 $\mathbf{S}$ 是 semantic token 序列，$\mathbf{S}^p$ 是 prompt semantic token 序列，$\mathbf{P}$ 是 text token 序列，$\mathbf{S}^p$ 和 $\mathbf{P}$ 是第一阶段模型的 condition
+ 第二阶段模型训练学习 $p_{\theta_{s2}}(\mathbf{A}|\mathbf{A}_t, (\mathbf{A}^p, \mathbf{S}))$，其中 $\mathbf{A}$ 是 acoustic token 序列，$\mathbf{A}^p$ 是 prompt acoustic token 序列

#### Speech Semantic Representation Codec

离散语言表征可以分为 semantic 和 acoustic token。之前的方法用 k-means 对 semantic feature 进行离散化得到 semantic token，但是这种方法可能会导致信息丢失。本文训练 VQ-VAE 模型学习 codebook，从 speech SSL 模型中重构 speech semantic representations。对于 speech semantic representation 序列 $\mathbf{S} \in \mathbb{R}^{T \times d}$，vector quantizer 将 encoder $E(\mathbf{S})$ 的输出量化为 $\mathbf{E}$，decoder 将 $\mathbf{E}$ 重构为 $\hat{\mathbf{S}}$。使用重构损失、codebook 损失和 commitment loss 优化 encoder 和 decoder。总损失为：
$$\mathcal{L}_{{\mathrm{total}}}=\frac1{Td}(\lambda_{{\mathrm{rec}}}\cdot||\mathbf{S}-\hat{\mathbf{S}}||_{1}+\lambda_{{\mathrm{codebook}}}\cdot||\mathrm{sg}(\mathcal{E}(\mathbf{S}))-\mathbf{E}||_{2}+\lambda_{{\mathrm{commit}}}\cdot||\mathrm{sg}(\mathbf{E})-\mathcal{E}(\mathbf{S})||_{2}).$$
其中 $\mathrm{sg}$ 表示 stop-gradient。

具体来说，使用 W2v-BERT 2.0 的第 17 层 hidden states 作为 speech encoder 的 semantic feature。encoder 和 decoder 由多个 ConvNext block 组成。使用 factorized codes 将 encoder 的输出投影到低维潜变量空间。codebook 包含 8,192 个 entry，每个维度为 8。
> 即使用 VQ-VAE 根据重构损失训练量化器，将 reperesentation 量化为 token

#### Text-to-Semantic Model

非自回归 Masked Generative Transformer 训练 T2S 模型。训练时，随机提取 semantic token 序列的 prefix 的一部分作为 prompt $\mathbf{S}^p$，然后将 text token 序列 $\mathbf{P}$ 和 $\mathbf{S}^p$ 拼接作为 condition。将 $(\mathbf{P}, \mathbf{S}^p)$ 作为输入 masked semantic token 序列 $\mathbf{S}_t$ 的 prefix 序列，来利用语言模型的 in-context learning 能力。
> 使用 Llama-style transformer 作为模型的 backbone，包括 GELU 激活、rotation position encoding 等，但是用双向注意力替换 causal attention。使用 adaptive RMSNorm，接受时间步 $t$ 作为 condition。

推理时，基于 text 和 prompt semantic token 序列 生成指定长度的 semantic token 序列。
> 训练了一个 flow matching 模型预测总时长，条件是 text 和 prompt 语音的时长。

#### Semantic-to-Acoustic Model

采用 masked generative codec transformer 模型，以 semantic 为条件训练 S2A 模型。基于 SoundStorm，生成多层 acoustic token 序列。给定 $N$ 层 acoustic token 序列 $\mathbf{A}^{1:N}$，在训练时，选择第 $j$ 层，将第 $j$ 层 acoustic token 序列记为 $\mathbf{A}^j$。mask $\mathbf{A}^j$ 得到 $\mathbf{A}^{j,t}$，模型训练时，基于 prompt $\mathbf{A}^p$、semantic token 序列 $\mathbf{S}$ 和小于 $j$ 层的 acoustic token 预测 $\mathbf{A}^j$，即 $p_{\theta_{s2a}}(\mathbf{A}^j|\mathbf{A}^{j}_t, (\mathbf{A}^p, \mathbf{S}, \mathbf{A}^{1:j-1}))$。使用线性 schedule $p(j) = 1 - \frac{2j}{N(N+1)}$ 采样 $j$。推理时，从 coarse 到 fine 生成每一层的 token，使用迭代并行解码。训练流程如下：
![](image/Pasted%20image%2020241101160827.png)


#### Speech Acoustic Codec

speech acoustic codec 用于将 speech waveform 量化为多层 discrete token。使用 RVQ 将 24K 语音压缩为 12 层的 token。每层 codebook 大小为 1,024，维度为 8。模型架构、判别器和训练损失遵循 [DAC- High-Fidelity Audio Compression with Improved RVQGAN 笔记](DAC-%20High-Fidelity%20Audio%20Compression%20with%20Improved%20RVQGAN%20笔记.md)，使用 [Vocos- Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis 笔记](轻量化/Vocos-%20Closing%20the%20gap%20between%20time-domain%20and%20Fourier-based%20neural%20vocoders%20for%20high-quality%20audio%20synthesis%20笔记.md) 作为 decoder。

semantic 和 acoustic codec 的对比：
![](image/Pasted%20image%2020241101161122.png)

### 其他应用

MaskGCT 可以实现除 zero-shot TTS 外的其他任务，如 duration-controllable speech translation、emotion control、speech content editing 和 voice conversion。

## 实验和结果（略）
