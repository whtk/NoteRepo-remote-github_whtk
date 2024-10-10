> ICLR 2023，Meta AI

1. 提出 AudioGen，根据输入文本生成语音的自回归生成模型，基于离散的语音标准
2. 提出增强方法，混合不同的音频样本，使模型学习多个 source
3. 使用 10 个数据集来缓解额文本-音频数据稀缺问题
4. 使用 multi-stream modeling 来加速推理，实现更短的序列同时保持相似的比特率和感知质量

> 任务是根据文本描述生成音频，不是 TTS。也是 codec + LM 的结构，然后在 LM 的时候把 text embedding 不仅作为自回归的输入，还作为了 cross attention 的 KV。

## Introduction
1. 图像生成和音频生成的一些区别：
    + 音频是一维信号，对于 overlap 的区分更困难
    + 现实世界的音频具有混响，区分起来更加困难
    + 心理声学和心理视觉特性不同，听觉“分辨率”（等响度）在频率上呈 U 形，4kHz 有一个低谷，8kHz 有一个高峰
    + 文本描述的音频数据远远少于文本-图像配对数据
2. 本文提出 AudioGen，包含两个阶段：
    + 使用 codec 模型将音频编码为离散 token 序列
    + 使用 Transformer-decoder LM 模型基于离散音频 token 和文本来生成音频

## 相关工作（略）

## 方法
AudioGen 包含两个步骤：
1. 使用 auto-encoding 学习音频的离散表征
2. 基于 encoder 得到的 codes 训练 Transformer LM，条件为文本特征

结构如图：
![](image/Pasted%20image%2020241010111253.png)

### 音频表征

音频信号可以表示为 $x \in [-1, 1]^{C_a \times T}$，其中 $C_a$ 是通道数，$T = d \cdot f_{\text{sr}}$ 是音频样本数，$f_{\text{sr}} = 16\text{kHz}$。音频表征模型包含三部分：
+ encoder 网络 $E$，输入音频，输出表征 $z$
+ 量化层 $Q$，使用 VQ 得到压缩后的表征 $z_q$
+ decoder 网络 $G$，从压缩表征 $z_q$ 重构时域信号 $\hat{x}$

整个系统端到端训练，最小化重构损失和感知损失。预训练完成后，可以用 encoder 和 quantizer 作为 discrete feature extractor（$Q \circ E$），用 $G$ 解码表征到时域信号。对于 $Q$，使用一个包含 2048 个 code 的 codebook，每个 code 是 128 维向量。

架构如下：
+ encoder $E$ 包含 $C$ 通道的 1D 卷积，后接 $B$ 个卷积块。每个卷积块包含一个残差单元和一个下采样层，下采样层包含一个 stride 为 $S$ 的卷积，kernel size 为 $2S$。残差单元包含两个卷积和一个 skip-connection。每次下采样时通道数翻倍。卷积块后接两层 LSTM 用于序列建模，最后是一个 kernel size 为 7，输出通道数为 $D$ 的 1D 卷积层。使用 $C = 32, B = 4$，stride 为 $(2, 2, 2, 4)$。使用 ELU 作为非线性激活函数和 LayerNorm。
+ decoder 与 encoder 镜像，使用转置卷积代替 stride 卷积，stride 顺序与 encoder 相反，输出最终音频

通过最小化重构损失和对抗损失进行训练。具体来说，对于时域，最小化目标和重构音频之间的 L1 距离 $$\ell_t(\boldsymbol{x},\hat{\boldsymbol{x}})=\|\boldsymbol{x}-\hat{\boldsymbol{x}}\|_1$$
对于频域，使用 mel-spectrogram 上 L1 和 L2 损失的线性组合：
$$\ell_f(\boldsymbol{x},\hat{\boldsymbol{x}})=\frac{1}{|\alpha|\cdot|s|}\sum_{\alpha_i\in\alpha}\sum_{i\in e}\|\mathcal{S}_i(\boldsymbol{x})-\mathcal{S}_i(\hat{\boldsymbol{x}})\|_1+\alpha_i\|\mathcal{S}_i(\boldsymbol{x})-\mathcal{S}_i(\hat{\boldsymbol{x}})\|_2,$$
其中 $\mathcal{S}_i$ 是 64-bin mel-spectrogram，使用窗口大小为 $2^i$ 和 hop 长度为 $2^i/4$ 的归一化 STFT，$e = 5, \cdots, 11$ 是 scale，$\alpha$ 是 L1 和 L2 的系数。

为了进一步提高生成质量，使用 MS-STFT discriminator，基于多尺度复值 STFT，每个子网络包含一个 2D 卷积层（kernel size 为 $3 \times 8$，32 个通道），后接时间维度上 dilation rate 为 1、2 和 4 的 2D 卷积，频率轴上 stride 为 2。最后是一个 kernel size 为 $3 \times 3$，stride 为 $(1, 1)$ 的 2D 卷积。使用 5 个不同的尺度，STFT 窗口长度为 $[2048, 1024, 512, 256, 128]$。生成器的对抗损失为：
$$\ell_{feat}(\boldsymbol{x},\hat{\boldsymbol{x}})=\frac{1}{KL}\sum_{k=1}^K\sum_{l=1}^L\|D_k^l(\boldsymbol{x})-D_k^l(\hat{\boldsymbol{x}})\|_1,$$
其中 $K$ 是 discriminator 网络数，$L$ 是特征数。
discriminator 训练最小化以下损失：
$$\ell_d(\boldsymbol{x},\hat{\boldsymbol{x}})=\frac{1}{K}\sum_{k=1}^K\max(0,1-D_k(\boldsymbol{x}))+\max(0,1+D_k(\hat{\boldsymbol{x}})),$$
其中 $K$ 是 discriminator 数，生成器训练最小化以下损失：
$$\ell_g(\boldsymbol{x},\hat{\boldsymbol{x}})=\lambda_t\ell_t(\boldsymbol{x},\hat{\boldsymbol{x}})+\lambda_f\ell_f(\boldsymbol{x},\hat{\boldsymbol{x}})+\lambda_g\ell_g(\hat{\boldsymbol{x}})+\lambda_{\text{feat}}\ell_{\text{feat}}(\boldsymbol{x},\hat{\boldsymbol{x}}).$$

### Audio Language Model

给定文本输入 $c$，ALM 输出音频 token 序列 $\hat{z}_q$，然后用 $G$ 解码为波形。

text encoder $F$ 首先从原始文本输入得到语义表征 $F(c) = u$。然后，Look-Up-Table（LUT）将音频 token $\hat{z}_q$ 嵌入到连续空间，$LUT(\hat{z}_q) = v$。然后将 $u$ 和 $v$ 连接为 $Z = u_1, \cdots, u_{T_u}, v_1, \cdots, v_{T_v}$，其中 $T_u$ 和 $T_v$ 分别是文本和音频 token 的长度。
使用交叉熵损失训练 Transformer-decoder LM：
$$L_{\mathrm{LM}}=-\sum_{i=1}^{N} \sum_{j=1}^{T_{v}} \log p_{\theta}\left(\boldsymbol{v}_{j}^{i} \mid \boldsymbol{u}_{1}^{1}, \ldots, \boldsymbol{u}_{T_{u}}^{i}, \boldsymbol{v}_{1}^{1}, \ldots, \boldsymbol{v}_{j-1}^{i}\right)$$
Transformer-decoder LM 用类似 GPT2 的架构。在 transformer 的每个 attention block 中添加 audio 和 text 之间的 cross-attention。

在训练时，以 10% 的概率随机忽略 text conditioning。在推理时，从条件和无条件概率的线性组合中采样。具体来说，从下面的分布中采样：
$$\gamma\log p_{\theta}(\boldsymbol{v}_{j}^{i}|\boldsymbol{u}_{1}^{1},\ldots,\boldsymbol{u}_{{T_{u}}}^{i},\boldsymbol{v}_{1}^{1},\ldots,\boldsymbol{v}_{j-1}^{i})+(1-\gamma)\log p_{\theta}(\boldsymbol{v}_{j}^{i}|\boldsymbol{v}_{1}^{1},\ldots,\boldsymbol{v}_{j-1}^{i}),$$
其中 $\gamma$ 是 guidance scale。

为了生成高质量的音频，将原始音频下采样 32 倍，每个音频 token 对应 2ms。导致每秒音频包含 500 个 token。于是提出 Multi-Stream 方法。

考虑长度为 $T_v$ 的序列，可以使用两个比特率大致相同的并行的 steam 学习长度为 $T_v/2$ 的表征。推广到 $k$ 个 steam，每个长度为 $T_v/k$，每个 codebook 大小为 $2048/k$。可以通过将 $Q$ 从单个 code book VQ 推广为 residual VQ 模块来得到。在时间 $t$，网络输入 $k$ 个离散 code，然后使用 $k$ 个 embedding layer。时间 $t$ 的最终 embedding 是这 $k$ 个 embedding 的均值。使用 $k$ 个 LM prediction heads 输出 $k$ 个 code。
> prediction heads 互相独立，尝试在 stream $i-1$ 上条件 stream $i$，但没有观察到性能提升。

## 实验（略）
