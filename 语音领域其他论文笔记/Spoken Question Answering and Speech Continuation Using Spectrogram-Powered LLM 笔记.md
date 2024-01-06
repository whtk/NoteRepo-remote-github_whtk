> Google，Verily，2023，preprint

1. 提出采用预训练的 LLM 实现 QA 和 speech continuation
2. 整个系统可以端到端训练，且可以直接在 spectrogram 上进行
3. 目标函数可以只用 文本-语音 数据对 联合优化 语音识别、text continuation 和 语音合成
4. 提出的方法超过了现有的 spoken language model

> 简单来说，就是用 LM 来做三个任务。LM 输入为 prompt speech，第一步先做 ASR 输出 prompt 的文本，然后做 文本预测，输出预测的文本，最后做 语音预测，输出预测的语音。三个任务是有顺序的，前面的任务的输出可以一定程度作为后面任务的输入（或者说 condition）。


## Introduction

1. 提出 Spectron，是一种新的 spoken language model，可以实现：
	1. 输入输出都是 spectrogram
	2. 从预训练的 LLM 中 迁移生成能力
2. 预训练的 speech encoder + LLM decoder 的好处就是可以实现端到端的训练，且可以实现 SOTA 性能：
	1. LLM 转录或者生成 text，用来作为音频生成的条件
	2. 还提出了一个新的 spectrogram regression loss

总体框架如图：
![](image/Pasted%20image%2020231120202739.png)

## 相关工作（略）

## 方法

### 结构

提出一种新的架构进行直接的 speech continuation，采用预训练的 speech encoder $\mathcal{E}$ 和预训练的language decoder LM，encoder以语音 prompt 作为输入，将其编码到连续的语言特征，然后送入到 decoder 作为 prefix，整个架构可以通过交叉熵损失和一个新的重构损失联合优化。

推理时，提供语音 prompt 即可得到 text 和 speech continuation。

#### 输入预处理

采用有监督的语音 $x$ 和文本 $y$ 进行训练，这里的 $x$ 为 spectrogram，把它在位置 $s$ 分成两部分：
$$x_p=x_{\leq s}, \quad x_c=x_{>s}$$

第一段 $x_p$ 称为 prompt，输入到 encoder 中的来获得连续的表征以作为 LM 的条件，第二段 $x_c$ 称为 continuation 用于计算后续的重构损失。其对应的文本也可以在位置 $\phi(s)$ 分为两部分：

$$y_p=y_{\leq \phi(s)}, \quad y_c=y_{>\phi(s)}$$

#### speech encoder

encoder 是一个有 600M 参数的 Conformer encoder（在 12M 小时数据上训练，Google USM），输入 spectrogram，输出同时包含语言和声学信息的表征，输出通过一个 layer $\mathcal{P}_s$ 将表征投影到 LM 的 embedding 的维度：
$$x_p^{\operatorname{lm}}=\mathcal{P}_s\left(\mathcal{E}\left(x_p\right)\right)$$

#### language model

LM 用的是 PaLM2，有 600M 或者 1B 的参数。输入为 encoder 的特征 $x_p^{\mathrm{lm}}$ 作为 prefix，除此之外 encoder 和 decoder 之间没有联系。

训练时，decoder 采用 teacher-forced 方法来预测 文本 $y_p$、后续文本（text continuation）$y_c$ 和 speech embedding $x^p_c$。为了将 embedding 转为 spectrogram，引入一个 lightweight 模块 $h^{\text{pre}}$ 和 $h^{\text{post}}$，此时整个模型的预测过程可以写为：
$$[\hat{y}_p,\hat{y}_c,\hat{x}_c^p]=\mathrm{LM}(x_p^\mathrm{lm},[y_p,y_c,x_c^p])$$
用相同的结构来 decode 文本和 spectrogram 的好处有两个：
+ 受益于预训练的 LM，可以从文本预测文本
+ 预测得到的文本可以看成一种中间特征，来提高合成语音的质量

#### 声学投影层（Acoustic Projection Layer）

pre-net $h^{\text{pre}}$ 采用 MLP 将 GT spectrogram $x_c$ 投影到 LM 的维度：
$$x_c^p=h^\text{pre}{ ( x _ c ) }$$
> 实际上是一个降维的过程。

同时，LM 得到的 $\hat{x}_{c}^p$ 也会通过一个 MLP（即 post-net $h^{\text{post}}$）投影回去：
$$\hat{x}_c=h^{\mathrm{post}}(\hat{x}_c^p)$$

### 目标函数

如上图的训练过程，有两个 loss：
+ 交叉熵损失，用于 speech recognition 和 transcript continuation
+ 回归损失：用于 speech continuation

训练的时候，所有的参数都会更新。

#### speech recognition 和 transcript continuation

这个损失为：
$$\mathcal{L}_{\mathrm{ASR}}(y_p,\hat{y}_p)=\mathrm{CE}(y_p,\hat{y}_p),\quad\mathcal{L}_{\mathrm{LM}}(y_c,\hat{y}_c)=\mathrm{CE}(y_c,\hat{y}_c)$$
> 这个损失只和文本有关。

#### speech continuation

speech continuation 是一个回归任务，在给定前面得到的文本后预测 spectrogram 帧。计算的时候是一个在时域和特征域（频域）的 delta loss：
$$\begin{aligned}
&\Delta_k^{\mathbf{time}}(z)=z_{[1:T-k,:]}-z_{[k:T,:]}, \\
&\begin{aligned}\Delta_k^{\text{feat}} ( z ) = z _ { [ : , 1 : F - k ] }-z_{[:,k:F]},\end{aligned} \\
&\mathcal{L}_{1+2}(z,z')=||z-z'||_1+||z-z'||_2^2.
\end{aligned}$$
对于给定的 GT spectrogram  $x_c$ 和预测的 $\hat{x}_c$，损失为下面三项：
$$\begin{aligned}
&\mathcal{L}_\text{s}{ ( x _ c , \hat { x }_c)} =\mathcal{L}_{1+2}(x_c,\hat{x}_c),  \\
&\mathcal{L}_\text{f}{ ( x _ c , \hat { x }_c)} =\mathcal{L}_{1+2}(\Delta_1^\text{feat}{ ( x _ c ) },\Delta_1^\text{feat}{ ( \hat { x }_c)}),  \\
&\mathcal{L}_\mathrm{t}(x_c,\hat{x}_c) =\sum_{k=1}^K\mathcal{L}_{1+2}(\Delta_k^{\mathrm{time}}(x_c),\Delta_k^{\mathrm{time}}(\hat{x}_c)) 
\end{aligned}$$
这部分总的损失为：
$$\mathcal{L}_\text{Recon.}(x_c,\hat{x}_c)=\mathcal{L}_\text{s}(x_c,\hat{x}_c)+\mathcal{L}_\text{f}(x_c,\hat{x}_c)+\mathcal{L}_\textbf{t}(x_c,\hat{x}_c)$$


#### 总损失

把上面两个 loss 加起来，总损失为：
$$\mathcal{L}_{\mathrm{toal}}(x,y)=\mathcal{L}_{\mathrm{ASR}}(y_p,\hat{y}_p)+\mathcal{L}_{\mathrm{LM}}(y_c,\hat{y}_c)+\mathcal{L}_{\mathrm{Recon}.}(x_c,\hat{x}_c)$$

这说明，模型可同时优化三项能力：
+ Speech recognition
+ Transcript continuation
+ Conditional speech synthesis

### 推理

推理时，speech prompt 通过 encoder $\mathcal{E}$ ，然后通过 $\mathcal{P}_s$ 投影得到 $x_p^{\operatorname{lm}}$，拼接一个 sos 的 token 后送入到 LM 中进行自回归预测：
$$\hat{y}=\mathrm{LM}([x_p^\mathrm{lm},\mathrm{sos}])$$
知道输出 eos 结束，这里的 $\hat{y}$ 时预测的文本和接续的文本的集合 $[\hat{y}_p,\hat{y}_c]$，完了之后再预测 spectrogram，即根据 $x_p^{\operatorname{lm}},\hat{y}$ 和之前预测的 $\begin{aligned}\hat{x}_c(\leq t-1)\end{aligned}$ 来进行自回归预测 $\hat{x}_{c}(t)$，同时整个过程还要考虑前面说的 pre-net 和 post-net：
$$\begin{gathered}
\hat{x}_{c}^{p}(\leq t-1)=h^{\mathbf{pre}}(\hat{x}_{c}(\leq t-1)) \\
\hat{x}_c^p(t)=\text{LM}([x_p^\mathrm{lm},\text{sos},\hat{y},\hat{x}_c^p(\leq t-1)]) \\
\begin{aligned}\hat{x}_c(t)=h^\mathrm{post}(\hat{x}_c^p(t)).\end{aligned} 
\end{gathered}$$
最后通过 vocoder 将得到的 $\hat{x}_{c}$ 转为波形。

## 实验（略）