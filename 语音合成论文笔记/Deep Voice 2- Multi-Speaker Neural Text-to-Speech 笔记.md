> Baidu 硅谷 AI Lab，2017，NIPS

1. 提出一种方法，采用低维的可训练的 speaker embeddings 在单一的 TTS 模型中生成不同说话人的语音
2. 提出 Deep Voice 2，基于 Deep Voice 1 的 pipelines，但是用更高性能的模块来构造，从而极大地提高了音频质量
3. 对于 Tacotron，引入 post-processing neural vocoder 也可以提高合成质量
4. 然后展示了将提出的方法用于 Deep Voice 2 和 Tacotron 的 多说话人语音合成，表明一个 TTS 模型就可以合成很多个独一无二的声音，而且每个说话人的训练音频时间少于半个小时

## Introduction

1. 开发支持多个说话人声音的 TTS 系统需要很多数据
2. 本文表明，可以在共享不同说话人的大部分的参数的情况下，构建纯神经网络的多说话人 TTS 系统，单个模型可生成不同的系统，而且每个人的数据更少（相比于单说话人）
3. 贡献如下：
	1. Deep Voice 2
	2. 引入 WaveNet-based spectrogram-to-audio neural vocoder，替换 [Tacotron- Towards End-to-End Speech Synthesis 笔记](Tacotron-%20Towards%20End-to-End%20Speech%20Synthesis%20笔记.md) 中的 Griffin-Lim
	3. 采用上面两个模型作为 baseline，通过引入 trainable speaker embeddings  实现 multi-speaker neural speech synthesis


## 相关工作（略）

## Single-Speaker Deep Voice 2

Deep Voice 2 的基本结构和 Deep Voice 1 [Deep Voice- Real-time Neural Text-to-Speech 笔记](Deep%20Voice-%20Real-time%20Neural%20Text-to-Speech%20笔记.md) 差不多，如图：
![](image/Pasted%20image%2020230830163346.png)

不同点在于  phoneme duration 和 frequency models，1 中是用一个模型联合预测这两个值，2 则先预测 phoneme durations，然后把 duration 作为输入预测  frequency。

下面给出每个模型的细节

### Segmentation model 

segmentation model 为 convolutional-recurrent 结构，采用 CTC loss 分类 phoneme pairs，和 1 差不多，Deep Voice 2 的主要变化是，在卷积层中加上了  batch normalization 和 residual connection，即 1 中每层的计算为：
$$h^{(l)}=\text{relu}\left(W^{(l)}*h^{(l-1)}+b^{(l)}\right)$$
而 2 变成：
$$h^{(l)}=\text{relu}\left(h^{(l-1)}+\text{BN}\left(W^{(l)}*h^{(l-1)}\right)\right)$$

此外，作者发现，模型在预测 silence phonemes 和 other phonemes 的边界时经常出错，于是引入一个小的 post-processing step 来纠正错误：每次模型解码到  silence boundary，采用 silence detection heuristic 调整边界的位置。

### Duration model

2 没有预测 continuous-valued duration，而是把它当作一个标签问题，将 phoneme duration  离散化为 log-scaled buckets，每个 phoneme 对应 bucket label，通过 conditional random field (CRF) 来建模序列。推理时采用维特比算法进行解码。

### Frequency Model

phoneme durations 上采样到  per-fram 形式（原来是 per-phoneme 形式），然后输入到 frequency model，包含多层：
+ 双向 GRU 先产生 hidden states
+ affine projection + sigmoid 激活基于  hidden states 判断每帧 voiced 的概率
+ hidden states 也被用于两个分开的  normalized F0 predictions：
	+ 第一个 $f_{GRU}$ 由单层的双向 GRU + ffine projection 组成
	+ 第二个 $f_{conv}$ 由多个不同 kernel size 的卷积层的组成
+ 最后，hidden state 通过 affine projection 和 sigmoid nonlinearity 预测 mixture ratio $\omega$，用于加权这两个 normalized frequency predictions：$f=\omega\cdot f_\text{GRU}+(1-\omega)\cdot f_{\mathrm{conv}}$

然后通过下式转换为 $F_0$：
$$F_0=\mu_{F_0}+\sigma_{F_0}\cdot f$$
其中 $\mu_{F_0},\sigma_{F_0}$ 为当前模型训练的这个说话人对应的 $F_0$ 的均值和方差。
> 卷积和 RNN 混合使用比只用一种的效果要好

### Vocal Model

和 Deep Voice 1 相似，用的是  WaveNet architecture with a two-layer bidirectional QRNN ，移除了 $1\times 1$ 的卷积，每层采用了相同的  conditioner bias。

## 基于可训练的 Speaker Embeddings 的多说话人模型

为了合成多个说话人的语言，对每个说话人都使用一个低维的 speaker embedding 向量，但是和之前的方法不一样，作者的方法并不依赖于 per-speaker weight matrices or layers。说话人相关的参数存在一个非常低维的向量中，所以在不同的说话人之间几乎是完全的权重共享。采用此 speaker embeddings 作为 RNN 的初始状态。

这个过程中， Speaker embeddings 是先在 $[-0.1,0.1]$ 之间均匀初始化的，然后和通过反向传播联合训练。

但是发现，把 speaker embeddings 只放在输入输入层效果并不好，而一些可以提高性能的方法有：
+ Site-Specific Speaker Embeddings：将 speaker embedding 转换为合适的维度，然后通过 affine projection 和 nonlinearity （其实就是同一个 speaker embedding 通过不同的线性层，得到一个所谓的 site-specific 的 speaker embedding）
+ Recurrent Initialization 用于初始化 recurrent layer hidden states 
+ Input Augmentation：将 speaker embedding 拼接到每个 time step 的 RNN 的输入
+ Feature Gating：使用 speaker embedding 对 layer activations 进行 elementwise 相乘

下面描述 speaker embeddings 是如何用在每个具体的模块中的。

### Multi-Speaker Deep Voice 2

Deep Voice 2 中每个模块都有一个独立的 speaker embedding，加起来可以看成一组的 speaker  embedding，且独立训练。

![](image/Pasted%20image%2020230831101612.png)

#### Segmentation Model

在卷积层的 residual connection 中使用 feature gating，此时公式为：
$$h^{(l)}=\text{relu}\left(h^{(l-1)}+\text{BN}\left(W*h^{(l-1)}\right)\cdot g_s\right)$$
其中，$g_s$ 为特定说话人的  speaker embedding，相同的 embedding 在所有层中共享。

#### Duration Model

采用 speaker-dependent recurrent initialization 和 input augmentation。一个 site-specific embedding 用于初始化 RNN 初始状态，另一个 site-specific embedding 则用于 和输入特征进行拼接得到 RNN 的输入。

#### Frequency Model

也用了  recurrent initialization。由于 Deep voice 2 预测的是 normalized frequency，其依赖说话人的 $F_0$ 的均值和方差，而此值在不同的说话人之间差异很大，因此让这两个值也是可训练的，然后通过一个和 speaker embeddings 相关的缩放因子进行缩放：
$$F_0=\mu_{\mathrm{F}_0}\cdot\left(1+\mathrm{softsign}\left(V_{\mu}^Tg_f\right)\right)+\sigma_{\mathrm{F}_0}\cdot\left(1+\mathrm{softsign}\left(V_{\sigma}^Tg_f\right)\right)\cdot f$$
这里的 $g_f$ 就是 speaker embedding，$V$ 为可训练的参数矩阵。

#### Vocal Model

只用了  input augmentation，将 site-specific speaker embedding 和条件模块的每个输入帧（如 mel 谱帧）进行拼接。但是这个不是 global condition。

其实没有 speaker embedding，模型也可以输出一定特色的音频，因为输入的特征包含了说话人相关的信息，有的话效果肯定更好。

### Multi-Speaker Tacotron

训练 multi-speaker Tacotron 时发现，模型性能受超参数影响很大，而且发现如果每个 audio clip 不是从相同的 time step 开始的话，效果很差，于是将所有开头和结尾的 silence 段去掉。

#### Character-to-Spectrogram Model

Tacotron 的 character-to-spectrogram 包含  convolution-bank-highway-GRU (CBHG) encoder、attentional decoder 和  CBHG post-processing network。

发现，将 speaker embeddings 引入 CBHG post-processing network会降低输出的质量，但是引入到 character encoder 是必须的。没有 speaker-dependent CBHG encoder，模型就不能学习 attention mechanism 而且不能生成有意义的输出。
> encoder 中，使用一个 site-specific embedding 作为每个  highway layer 的额外的输入，另一个 site-specific embedding 则用于初始化 e CBHG RNN  的状态


对于 decoder，使用一个 site-specific embedding 作为 decoder pre-net 的额外输入，另一个 site-specific embedding 作为 attentional RNN 的 初始化 attention context，再用一个 site-specific embedding 初始化 decoder GRU  的 hidden state，最后一个 ne site-specific embedding 作为 tanh 的 bias。

#### Spectrogram-to-Waveform Model

原始的 Tacotron 采用 Griffin-Lim 生成音频，但是在 spectrogram 中的轻微噪声都能造成 Griffin-Lim 算法效果变差。于是采用 WaveNet-based neural vocoder，这部分就和 Deep Voice 2 vocal model 相似，但是输入的是 linear-scaled log-magnitude spectrogram 而非  phoneme identity 和 $F_0$，如图：
![](image/Pasted%20image%2020230831112630.png)

## 结果（略）

