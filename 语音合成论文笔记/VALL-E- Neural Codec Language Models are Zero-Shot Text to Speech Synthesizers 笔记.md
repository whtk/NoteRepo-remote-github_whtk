> Microsoft

1. 训练 neural codec language model（VALL-E），把 TTS 作为一个条件语言建模任务而非之前的连续信号回归任务
2. 预训练时，用 60K 小时的语音训练模型
3. VALL-E 涌现出 in-context learning 的能力，只需要 3s 的音频作为声学 prompt，可以合成高质量的个性化的语音
4. 远超现有的 SOAT zero-shot TTS 模型，同时还可以保留说话人的情感和声学环境

## Introduction

![](image/Pasted%20image%2020230924152822.png)

如图，为了合成个性化的语音（zero-shot TTS），VALL-E 基于 3s 的音频的 acoustic token（对应 speaker information） 和 phoneme prompt 来生成 acoustic tokens（对应 content information），然后用 neural codec decoder 基于这些 acoustic token 来生成波形。

离散的 acoustic token 就类似于类似于语言模型中的 token，从而可以用上一些 prompting-based large-model techniques。

在 7000个说话人超过 60K 小时的数据集下训练。

总的贡献如下：
+ 提出 VALL-E，第一个有着 in-context learning 能力的 TTS 框架
+ 基于半监督数据构建 TTS，大力出奇迹
+ 相同的文本下可以生成多样的输出
+ 可以在 zero-shot 场景下实现 high speaker similarity

## 相关工作

## 音频量化

传统的量化采用 mu 律，这里用的是 codec。

相比于其他的量化方法，audio codec 的好处是：
+ 包含大量的说话人信息和声学信息
+ 有一个 off-the-shelf codec decoder 将 token 转为 波形，不需要 vocoder
+ 减少了 time step 的长度

采用的是预训练的  neural audio codec model，EnCodec 作为 tokenizer。结构如下：
![](image/Pasted%20image%2020230924161210.png)

## VALL-E

### 将 TTS 问题看成是 Conditional Codec Language Modeling

给定数据集 $\mathcal{D}=\{\mathbf{x}_i,\mathbf{y}_i\}$，其中 $\mathbf{y}$ 音频样本，$\mathbf{x}=\{x_0,x_1,\dots,x_L\}$
为对应的 phoneme 序列，采用预训练 的 neural codec 模型将音频编码到离散的 acoustic token，记为 $\mathrm{Encodec}(\mathbf{y})=\mathbf{C}^{T\times8}$，这里的 $\mathbf{C}$ 是一个二维的 acoustic code matrix，$T$ 为下采样后的音频长度。

这个矩阵的每行 $c_t$，表示第 $t$ 帧的 8 个codes。然后就可以用 neural codec decoder 来重构波形，$Decodec(C)\approx\hat{\mathbf{y}}$。

训练模型，基于 phoneme sequence $\mathbf{x}$ 和 acoustic prompt matrix $\tilde{\mathbf{C}}^{T^{\prime\times 8}}$ 来生成 acoustic code matrix $\mathbf{C}$，目标函数是 $\max p(\mathbf{C}|\mathbf{x},\tilde{\mathbf{C}})$。
> $\tilde{\mathbf{C}}$ 是把参考音频输入 codec 得到的 token。

推理时，得到的 acoustic code matrix $\mathbf{C}$ 通过 language model 进行处理，然后通过 neural codec decoder 进行解码以合成音频。

### 训练（Conditional Codec Language Modeling 过程）

![](image/Pasted%20image%2020230924163138.png)

由于用的是 residual 的量化，前一个量化点包含 acoustic property（如 speaker identify），而后面的量化点学习精细的 acoustic 细节。因此也以 hierarchical 的方式设计了两个 conditional language model。

对于来自第一个 quantizer 的 tokens $\mathbf{c}_{:,1}$（一共 8 个） ，训练一个 自回归的 decoder-only 的 language model，写为：
$$p(\mathbf{c}_{:,1}|\mathbf{x},\mathbf{\tilde{C}}_{:,1};\theta_{AR})=\prod_{t=0}^Tp(\mathbf{c}_{t,1}|\mathbf{c}_{<t,1},\mathbf{\tilde{c}}_{:,1},\mathbf{x};\theta_{AR})$$

对于剩下的 7 个，$\mathbf{c}_{:,j\in[2,8]}$，训练一个非自回归的 language model。模型基于 $\mathbf{x},\tilde{\mathbf{C}}$ 和前面位置的 codebook 的 acoustic token $\mathbf{C}_{:,<j}$：
$$p(\mathbf{C}_{:,2:8}|\mathbf{x},\mathbf{\tilde{C}};\theta_{NAR})=\prod_{j=2}^8p(\mathbf{c}_{:,j}|\mathbf{C}_{:,<j},\mathbf{x},\mathbf{\tilde{C}};\theta_{NAR})$$
两个模型组合，可以实现合成质量和推理速度直接的 trade off，这里的 AR 模型可以用于预测序列的长度，而 非 AR 减少了复杂度，整个的 tokens 预测可以写为：
$$p(\mathbf{C}|\mathbf{x},\tilde{\mathbf{C}};\theta)=p(\mathbf{c}_{i,1}|\tilde{\mathbf{C}}_{i,1},\mathbf{X};\theta_{AR})\prod_{j=2}^8p(\mathbf{c}_{i,j}|\mathbf{c}_{i,<j},\mathbf{x},\tilde{\mathbf{C}};\theta_{NAR})$$
#### 自回归

自回归模型用于生成第一个位置的量化点，包含 phoneme embedding $W_x$，acoustic embedding $W_a$ ，transformer  decoder 和 prediction layer。

把 phoneme 序列作为 phoneme prompt，也即输入为 $\mathbf{x}$ 和 $c_{:,1}$ 的拼接，两个特殊的 EOS token 分别放在他们后面。整个训练过程就是简单的 causal language model。

推理的时候，phoneme 序列都要和参考音频与合成音频的 token 进行拼接。然后把参考音频的 token 作为 prefix 进行推理。

### 非自回归

得到第一个位置的量化点后，用非自回归模型生成剩下的 7 个，其结构和 自回归的相同，但是包含 8 个独立的  acoustic embedding layers，在每个训练步中，随机采样 $i\in[2,8]$，然后把 embedding 的输出从 $1$ 到 $i-1$ 进行求和作为模型的输出：
$$\begin{gathered}
e_{ct,j} =W_a^j\odot c_{t,j} \\
\text{ect} =\sum_{j=1}^{i-1}e_{c_t,j} 
\end{gathered}$$
其中，$\odot$ 表示 index selection，同时把 phoneme 序列和参考音频的 acoustic tokens 都作为 prompt。参考音频的 acoustic tokens 对应的 embedding 为 $\mathbf{e_{\tilde{c}_{t}}}=\sum_{j=1}^{8}e_{\tilde{c}_{t,j}}$，然后 transformer 的输入就是 $(\mathbf{e_x},\mathbf{e_{\tilde{c}}},\mathbf{e_{c_{:,<i}}})$。

### 推理：基于 prompt 的 in-context learning

如果模型不进行 fine tune 也可以在未知说话人下合成高质量的音频，则认为这个模型有 in-context learning 的能力。

prompt 很重要，设计如下：
+ 将 text 转为 phoneme 序列，将参考语音转为 acoustic matrix
+ 对于 AR 模型，基于 prompt 采用 sampling-based decoding（beam search 会导致死循环）
+ 对于 非 AR，用 reedy decoding  每次选择概率最高的 token
+ 最后用 decoder 生成波形

由于 acoustic prompt 与合成的音频语义 相关/不相关，导致两种情况：
+ VALL-E（不相关）：目标是生成未知说话人下的给定内容的语音，输入为 文本、参考语音和其对应的文本，将参考语音的文本加到原始的文本的前面作为prompt，用参考语音的 $\tilde{c}_{:,1}$ 作为 acoustic prefix，从而可以在给定的文本下克隆说话人的音色
+ VALL-E-continual（相关）：把整个文本作为 phoneme，把前 3s 的音频作为 acoustic prompt，模型用于生成 3s 后的音频，此时文本和语音是语义相关的


## 实验（略）

