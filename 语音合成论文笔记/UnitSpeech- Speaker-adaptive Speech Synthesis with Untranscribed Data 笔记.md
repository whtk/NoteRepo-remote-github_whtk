> 2023，interspeech，Seoul National University,

1. 提出 UnitSpeech，一个说话人自适应的语音合成模型，采用无文本的数据 fine tune 基于 diffusion 的 TTS 模型
2. 采用 self-supervised unit representation 作为 pseudo transcript，把 unit encoder 集成到预训练的 TTS 模型中
3. unit encoder 将语音的内容引入到 decoder 中，然后 fine tune decoder 以进行说话人的自适应
4. 可以实现 TTS 和 VC 任务，而不需要模型的 re-training

## Introduction

1. adaptive TTS 可以基于 target speaker 的参考音频产生个性化的音频
2. adaptive TTS 通常采用预训练的多说话人 TTS，然后使用 target speaker embedding 或者 fine tune 模型
3. 大部分基于 fine tune 的方法都需要少量数据（包括语音文本对）
4. 提出 UnitSpeech，采用 multi-speaker Grad-TTS 作为 backbone，和 AdaSpeech 2 一样，使用一个 encoder 在 diffusion-based 的 decoder 中引入语音内容信息（无需文本），而且采用的是自监督的 unit representation 作为 encoder 的输入（称为 unit encoder）
	1. 对于 speaker adaptation，采用目标说话人的 unit, speech 对来 fine tune 预训练的 diffusion 模型

## 方法

目标是使用无文本的数据来个性化基于 diffusion 的 TTS 模型。
![](image/Pasted%20image%2020230923201901.png)
### 基于 diffusion 的 TTS 模型

采用 multi-speaker Grad-TTS 作为预训练的 TTS 模型，包含：
+ text encoder
+ duration predictor
+ 基于 diffusion 的 decoder

然后采用来自 speaker encoder 的 speaker embedding 来提供说话人信息以实现 multi-speaker 的 TTS。

Grad-TTS 的先验分布是 mel-spectrogram-aligned text encoder output，这里采用正态分布，forward diffusion 写为：
$$dX_t=-\frac12X_t\beta_tdt+\sqrt{\beta_t}dW_t,\quad t\in[0,1],$$
其中，$\beta_t$ 为 noise schedule，$W_t$ 为维纳过程。

在预训练的时候，$X_0$ 通过 forward process 加噪得到 $X_t=\sqrt{1-\lambda_t}X_0+\sqrt{\lambda_t}\epsilon_t$，然后 decoder 基于 对齐后的 text encoder 输出 $c_y$ 和 speaker embedding $e_S$ 估计条件得分，其目标函数如下：
$$L_{grad}=\mathbb{E}_{t,X_0,\epsilon_t}[\|(\sqrt{\lambda_t}s_\theta(X_t,t|c_y,e_S)+\epsilon_t\|_2^2]],$$
在学习到 $s_{\theta}$ 之后，模型就可以生成 mel 谱 $X_0$（也需要给定文本和 speaker embedding），此过程即为 reverse process：
$$X_{t-\frac1N}=X_t+\frac{\beta_t}N(\frac12X_t+s_\theta(X_t,t|c_y,e_S))+\sqrt{\frac{\beta_t}N}z_t,$$
其中 $N$ 为采样步。

同时使用了 MAS 来对齐 mel 谱和 text encoder 输出，然后采用一个 encoder MSE loss 来最小化对齐后的输出 $c_{y}$ 和 $X_0$ 的距离。 

### Unit Encoder

将 unit encoder 和 前面预训练的 TTS 模型组合起来提高自适应的生成能力。

unit encoder 在结构和功能上等效于 text encoder，但是 text encoder 输入的是文本，而 unit encoder 使用离散的表征，称为 unit。

unit 是通过 HuBERT 得到的离散表征，提取过程如图最左边。HuBERT 输入为语音波形，输出表征通过 k-mean 聚类得到 unit cluster，从而得到 unit sequence。然后把 unit sequence 上采样到 mel 谱 的长度，拓展后的序列也可以进一步分成两个部分： unit duration $d_u$ 和压缩后的 unit $u$（看图就能明白）。

把 $u$ 作为 unit encoder 的输入，其他的就完全和 text encoder 一样了（可以把 unit sequence 看成 phoneme sequence）。最后通过 $d_u$ 拓展得到 encoder 之后的 $c_u$。

### 说话人自适应的语音合成

将前面两个模块组合，就可以采用无文本的 target speaker 的语音来自适应合成音频。采用从 target speaker 中提取的 $u^{\prime},d_{u^\prime}$ 来 fine tune TTS 中的 diffusion decoder 模型（冻住 unit encoder 的参数），简单把 $c_y$ 替换为 $c_{u^\prime}$ 即可。
> 其实就是 voice conversion。

采样时用的是 classifier-free guidance 的方法。classifier-free guidance 需要一个 unconditional embedding $e_{\Phi}$ 来估计 unconditional score，选择数据集中所有的 mel 谱 的均值 $c_{mel}$ 作为这个 null token，修改后的 score 变为：
$$\begin{gathered}
\begin{aligned}\hat{s}(X_t,t|c_c,e_S)=s(X_t,t|c_c,e_S)+\gamma\cdot\alpha_t,\end{aligned} \\
\begin{aligned}\alpha_t=s(X_t,t|c_c,e_S)-s(X_t,t|c_{mel},e_S).\end{aligned} 
\end{gathered}$$
其中的 $c_c$ 即为 unit 或 text encoder 对齐后的输出，也就是真的 condition，$\gamma$ 为梯度的 scale。


## 实验
