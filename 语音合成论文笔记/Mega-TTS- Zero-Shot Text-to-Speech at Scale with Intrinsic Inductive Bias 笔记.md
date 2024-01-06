> ZJU，字节，renyi，2023 preprint

inductive biases：归纳偏置
1. 之前的 TTS 用 codec 对语音进行编码，然后用自回归或者 diffusion 来生成，这样做忽略了语音的内在特性（intrinsic nature）
2. 而语音实际上可以分解成一些属性（content, timbre, prosody, and phase），每个属性都可以采用一个有着 inductive biases 的模块来建模
3. 提出 Mega-TTS，在大规模的  wild data 上训练，以不同的方式建模不同的属性：
	1. 采用 spectrogram 作为表征，可以通过 vocoder 来很好的建模，而不需要 LM
	2. 采用 global vector 来建模 timbre
	3. 采用 VQGAN-based 模型来生成 spectrogram，采用 latent code language model 来拟合韵律的分布
4. 在 2w 小时的语音上训练

## Introduction

人类的语音可以分解为一些属性：
![](image/Pasted%20image%2020231203214934.png)
用 codec 来建模整个语音通常会忽略一些内在的东特性：
+ phase 和 语义无关，人对其敏感程度更低
+ timbre 应该作为一个 global vector 在整句中保持稳定，而不需要随时间变化建模
+ prosody 通常有着 long-term 和 global 依赖，随时间变化很快，且与文本弱相关，从而可以用 LLM 来建模
+ 内容 需要和语音单调对齐，但是自回归建模不能确保

本文提出 Mega-TTS：
+ 采用 spectrogram 作为表征，可以通过 vocoder 来很好的建模，而不需要 LM
+ 采用 global vector 来建模 timbre
+ 采用 VQGAN-based 模型来生成 spectrogram，采用 latent code language model （称为 P-LLM）来拟合韵律的分布


## 背景（略）

## 方法

如图：
![](image/Pasted%20image%2020231203220021.png)
包含 VQGAN-based TTS 和 prosody large language model (P-LLM)。

推理的时候，用文本序列中提取的 content、prompt speech 中提取的 timbre 和用 P-LLM 预测的 prosody的来生成目标语音，称这种解码机制为 prosody-oriented speech decoding。

### 将语音解耦为不同的组分

采用三种类型的 encoder 来分别编码 content、prosody 和 timbre。然后采用 GAN-based decder 来基于这些表征生成 mel 谱。

具体来说，用 autoencoder 的 重构损失 + bottleneck 来解耦这三个信息：
+ mel 谱 送入 prosody encoder，然后引入一个 dimension reduction 和 phoneme-level downsampling 来约束信息流
+ content encoder 直接从 phoneme 序列编码得到 content 表征
+ 把来自同一个说话人的不同语音中采样得到的 参考 mel 谱 来解耦  timbre 和 content，然后在时间维度平均 timbre encoder 得到的输出来得到 global timbre vector
+ 设计的 bottleneck 可以从 prosody encoder 的输出中移除 content 信息和 timbre 信息

prosody encoder 包含两层卷积、phoneme-level pooling layer 和一个 VQ 的 bottleneck。第一层卷积将 mel 谱 压缩为 phoneme-level hidden states，第二层则用于捕获 phoneme-level 的相关性。VQ 则用于获得 phoneme-level 的 code $\mathbf{u}=\{u_1,u_2,...,u_T\}$ 和 hidden states $H_{prosody}$。且为了减轻解耦的难度，只采用了 mel 谱 的低频段（前 20 bin）。

content encoder 包含几层 FFT（feed-forward Transformer），为了实现单调对齐，用的是 非自回归 TTS 如 FastSpeech 中的 duration predictor + length regulator。然后把从 prosody encoder 提取的 prosody 信息送入到 duration predictor 中来减轻 one-to-many 的映射问题。

timbre encoder 包含一些卷积层，最终得到的是时间平均后的 $H_{timbre}$。

然后用 GAN-based mel-spectrogram decoder 来实现更好的感知质量。用 multi-length discriminator，输入可以是不同长度的窗口，作为判别器。训练第一阶段的总损失为：
$$\begin{gathered}\mathcal{L}_{\mathrm{VQ}}=\|y_t-\hat{y}_t\|^2+\|\mathrm{sg}[E(y_t)]-z_{\mathbf{q}}\|_2^2+\left\|\mathrm{sg}\left[z_{\mathbf{q}}\right]-E(y_t)\right\|_2^2,\\\mathcal{L}=\mathbb{E}\left[\mathcal{L}_{\mathrm{VQ}}+\mathcal{L}_{\mathrm{Adv}}\right]\:,\end{gathered}$$
其中，$y_t$ 为 target speech，$\hat{y}_t$ 为生成的语音。$\mathcal{L}_{\mathrm{rec}}=\|y_t-\hat{y}_t\|^2$ 为重构损失。$z_q$ 为 codebook，$\mathcal{L}_{\mathrm{VQ}}$ 为 VQ-VAE 损失，$\mathcal{L}_{\mathrm{adv}}$ 为 LSGAN-style 对抗损失。

### P-LLM

P-LLM 是一个 latent code language model，用于捕获 local 和 long-range 的依赖来实现 prosody 建模。

设 $(\mathbf{y}_{\mathbf{p}},\mathbf{x}_{\mathbf{p}}),(\mathbf{y}_{\mathbf{t}},\mathbf{x}_{\mathbf{t}})$ 分别为 prompt 和 target 语音-文本 对，目标是，给定未知说话人的语音 $\mathbf{y}_{\mathbf{p}}$，合成目标说话人的语音 $\mathbf{y}_{\mathbf{t}}$。
> 注意：在推理的时候，target speech 的 timbre $\tilde{H}_{timbre}$ 要和 prompt 一致

这个时候还需要 target speech 的 prosody 信息 $\tilde{\mathbf{u}}$。从而，prosody-oriented speech decoding 可以写为：
$$\begin{aligned}&\textbf{Encode}:\mathbf{u}=E_{prosody}(\mathbf{y}_{\mathbf{p}}),\quad H_{content}=E_{content}(\mathbf{x}_{\mathbf{p}}) \quad \tilde{H}_{timbre}=E_{timbre}(\mathbf{y}_{\mathbf{p}}),\quad \tilde{H}_{content}=E_{content}(\mathbf{x}_{\mathbf{t}})
\\&\textbf{Prosody prediction}:\tilde{\mathbf{u}}=f(\tilde{\mathbf{u}}|\mathbf{u},H_{content},\tilde{H}_{timbre},\tilde{H}_{content},\theta)
\\&\textbf{Decode}:\hat{y}_{t}=\hat{D}(\tilde{\mathbf{u}},\tilde{H}_{timbre},\tilde{H}_{content})\end{aligned}$$
其中，$\theta$ 为 P-LLM 的参数。

在推理的时候，需要 target speech 的 prosody $\tilde{\mathbf{u}}$，采用  LLM 的 in-context learning 能力，P-LLM 是一个 decoder-only transformer-based 的结构，其采用从 $\mathbf{y}_{\mathbf{p}}$ 中提取的 $\tilde{\mathbf{u}}$ 作为 prompt，用 $H_{content},\tilde{H}_{content},\tilde{H}_{timbre}$ 作为条件，自回归地预测：
$$p\left(\tilde{\mathbf{u}}\mid\mathbf{u},H_{content},\tilde{H}_{timbre},\tilde{H}_{content};\theta\right)=\prod_{t=0}^Tp\left(\tilde{u}_t\mid\tilde{u}_{<t},\mathbf{u},H_{conttent},\tilde{H}_{timbre},\tilde{H}_{content},\theta\right)$$
训练的时候以 teacher forcing 的方式，采用交叉熵损失进行。

### 用于推理的 speech prompt

为了把 in-context learning 用于不同的 生成 任务，设计了不同的 prompt 机制。

![](image/Pasted%20image%2020231203230940.png)

用于 TTS 任务：为了实现 zero-shot TTS，根据上式采用 $\mathbf{u},H_{content},\tilde{H}_{timbre},\tilde{H}_{content}$ 来预测 $\tilde{\mathbf{u}}$，采用 top-k 随机采样策略来从结果中采样，从而增加合成语音的多样性。然后将 $\tilde{H}_{timbre},\tilde{H}_{content},\tilde{\mathbf{u}}$ 进行拼接，用 mel decoder 来生成 target speech $y_t$。

用于 speech editing 任务：这个任务中，预测的 prosody codes 应该和其左右边界有尽可能光滑的转换过渡。之前的方法是从左往右、从右往左分别做自回归然后在 L2-norm 之间差异最小的那个点上做拼接，但是用 L2-norm 和人类的感知还是很不一样，这里由于 Mega-TTS 提取的是离散的 code，所以可以这么做：
+ 先把左边的 code 作为 prompt，用 top-k 的方法得到 $N$ 条路径
+ 把这 $N$ 条路径作为新的 prompt 来预测右边的部分，从而可以得到一个概率矩阵（因为右边的已知，为 GT）
+ 把每条路径上的对数概率求和，选择最大概率的那条作为最终的结果
用公式表达如下：
$$\begin{aligned}
\operatorname*{Max}_{i\in[1,N]}\text{Likelihood}& \begin{aligned}=\text{Max}_{i\in[1,N]}\prod_{t=L}^Rp\left(u_t^i\mid u_{<t}^i,H_{content},\tilde{H}_{timbre},\tilde{H}_{content};\theta\right)\end{aligned}  \\
&\cdot\prod_{t=R}^Tp\left(u_t^{gt}\mid u_{<t}^i,H_{content},\tilde{H}_{timbre},\tilde{H}_{content};\theta\right),
\end{aligned}$$
其中，$L,R$ 分别做左右边界，$T$ 为 mel 谱 长度，$u^i$ 为第 $i$ 条路径，$u^{gt}_t$ 为 GT prosody code。

## 实验（略）

