> preprint 2024.09，Johns Hopkins University、Tencent AI Lab

1. 提出 SSR-Speech，基于 codec 的自回归模型，实现 zero-shot 语音编辑和 TTS
2. SSR-Speech 基于 Transformer decoder，引入 CFG 增强生成过程的稳定性
3. 提出 watermark Encodec，将 frame-level watermark 嵌入 speech 的编辑区域，从而检测哪些部分被编辑
4. 重构 speech waveform 时，利用原始未编辑的 speech 段，相比 Encodec 模型有更好的恢复效果
5. 在 RealEdit 语音编辑任务和 LibriTTS TTS 实现 SOTA

## Introduction

1. 给定未知说话人，zero-shot SE 修改特定单词或短语，zero-shot TTS 生成整个 speech
2. 本文关注 zero-shot SE 和 TTS 的 AR 模型，提出 SSR-Speech，贡献如下：
	+ SSR-Speech 可以实现稳定的推理
	+ SSR-Speech 生成的语音包含 frame-level 水印，提供音频是否由 SSR-Speech 生成以及哪部分被编辑的信息
	+ SSR-Speech 对多段编辑和背景声音具有鲁棒性
	+ SSR-Speech 在 zero-shot SE 和 TTS 任务上显著优于现有方法

## SSR-Speech

模型为 causal Transformer decoder，输入 text tokens 和 audio neural codec tokens，预测 masked audio tokens。

### Modeling

给定语音信号，[EnCodec- High Fidelity Neural Audio Compression 笔记](../语音领域其他论文笔记/EnCodec-%20High%20Fidelity%20Neural%20Audio%20Compression%20笔记.md) 将其量化为离散 token $A=\{a_1,a_2,...,a_T\}$，其中 $T$ 为 tokens 长度，每个 token $a_i=\{a_{i,1},a_{i,2},...,a_{i,K}\}$ 对应 $K$ 个 Encodec 的 codebooks。

如图：
![](image/Pasted%20image%2020240930110457.png)

在训练时，随机 mask $P$ 个连续 span 的 audio（如图中 $P=1$）。被 mask 的 tokens 与特殊 token $[m_1],[m_2],...,[m_P]$ 连接，每个后面跟着特殊 token $[eog]$。未被 mask 的 tokens，也称为 context tokens，同样与特殊 token $[m_1],[m_2],...,[m_P]$ 连接，序列开头和结尾分别有特殊 token $[sos]$ 和 $[eos]$。整个 audio tokens 组合成新的 audio 序列 $A'=\{a'_1,a'_2,...,a'_{T'}\}$，其中 $T'$ 为新的长度。

使用 Transformer decoder 自回归地建模 masked tokens，条件是 phoneme 序列 $Y=\{y_1,y_2,...,y_L\}$，其中 $L$ 为 phoneme tokens 长度。在 $A^\prime$ 的每个时间步 $t$，模型基于 phoneme 序列 $Y$ 和 $A^\prime$ 中直到 $a^\prime_t$ 的所有前面 tokens（记为 $X_t$）预测 $a^\prime_t$：
$$\mathbb{P}_\theta(A^{\prime}\mid Y)=\prod_t\mathbb{P}_\theta\left(a_t^{\prime}\mid Y,X_t\right)$$
其中 $\theta$ 表示模型参数。损失为负对数似然：
$$\mathcal{L}(\theta)=-\log\mathbb{P}_\theta(A^{\prime}\mid Y)$$

采用 causal masking、delayed stacking，且第一个 codebook 的权重较大。只在 masked tokens 上计算预测损失，而不是所有 tokens。
> 为了进一步增强 TTS 训练，通过一定概率一直 mask 到语音的结尾，以实现 speech continuation。

### Inference

对于 SE 任务，比较原始 transcript 和目标 transcript，确定需要 mask 的单词。使用原始 transcript 的 word-level forced alignment，定位对应的 audio tokens。将 target transcript 的 phoneme tokens 和未被 mask 的 audio tokens 拼接后输入模型，预测新的 audio tokens。编辑语音时，需要略微调整 span 周围的相邻单词，以准确建模协同效应。因此，引入小的 margin 超参数 $\alpha$，在左右两侧将 masked span 的长度扩展 $\alpha$。

对于 TTS 任务，将 voice prompt 的 transcript 与要生成的语音的 transcript 结合，加上 voice prompt 的 audio tokens 输入模型。
> 这里做的其实是一个 speech continuation 的任务，即在 voice prompt 的基础上继续生成 speech。

由于自回归生成的随机性，模型偶尔会产生过长的 silence 或拉长某些声音，导致语音不自然。本文提出使用 CFG 解决这个问题。

但是发现传统 CFG 不能很好地解决 AR 模型的 dead loop，并且可能使训练在开始时不稳定。为了解决这个问题，提出使用 inference-only CFG，不需要 unconditioned 训练。具体来说，在推理时，使用随机文本序列作为无条件输入，从由条件和无条件概率的线性组合得到的分布中采样：
$$\gamma\mathbb{P}_\theta(A^{\prime}\mid Y)+(1-\gamma)\mathbb{P}_\theta(A^{\prime}\mid Y^{\prime})$$
其中 $\gamma$ 是 guidance scale，$Y^\prime$ 是与 $Y$ 长度相同的随机 phoneme 序列。

### Watermark Encodec

watermark Encodec 是专为 SE 设计的 neural codec，能够为生成的 audio 加水印。Watermark Encodec 也可以应用于 TTS 任务。如图：
![](image/Pasted%20image%2020240930154602.png)

模型包含：
+ speech encoder
+ quantizer
+ speech decoder
+ masked encoder
+ watermark predictor

### Watermarking (WM)

speech encoder 与 Encodec 中的 encoder 共享相同的网络架构。watermark predictor 采用与 Encodec encoder 相同的架构，增加一个用于二元分类的线性层。首先预训练 Encodec 模型，使用预训练的 Encodec encoder 参数初始化 speech encoder 和 watermark predictor 的参数。quantizer 与 Encodec quantizer 相同，参数相同。

speech decoder 输入为 watermarks 和 audio codes，重构语音，与 Encodec decoder 架构想通过。唯一的区别是用线性层将特征投影到与 audio 特征相同的维度。speech decoder 的参数也从 Encodec 模型初始化。在训练时，speech encoder 和 quantizer 固定。watermark 是一个二进制序列，与 speech encoder 输出的 audio frames 长度相同，其中 mask 的 frames 为 1，未 mask 的 frames 为 0，通过 embedding 层之后得到 watermark 特征。

### Context-aware Decoding (CD)

Encodec 使用 audio codes 重构 waveform。但是对于 SE 任务，重要的是 speech 的未编辑 span 保持不变。为了在解码过程中更好地利用这些未编辑 span 的信息，提出了 context-aware decoding 方法，把原始未编辑的 waveform 作为 watermark Encodec decoder 的额外输入。

具体来说，用 silence clips mask 原始 waveform 的编辑部分，然后使用 masked encoder 从这个 masked waveform 中提取特征。masked encoder 与 Encodec encoder 共享相同的架构，参数从 Encodec 初始化。因此，speech decoder 的输入包括 audio codes、watermarks 和 masked 特征。

此外，发现使用 skip connections 提高了重构质量并加速了模型收敛。因此在每个 block 之间类似于 UNet 的方法融合多尺度特征。

## 实验（略）
