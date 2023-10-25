> preprint，上海交大、思必驰，俞凯

1. 离散 speech  token 可以分为 语义 token 和 声学 token
2.  VALL-E 和 SPEAR-TTS 可以通过自回归的连续声学 token 实现 zero-shot speaker adaptation
3. 但是 AR 模型是有序的预测，不适用于 speech editing 这种可以看到之前和之后上下文的任务
4. 本文提出一种 unified context-aware TTS 框架，称为 UniCATS，可以实现 speech continuation 和 editing，包含两个部分：
	1. 声学模型 CTX-txt2vec，采用 contextual VQ-diffusion 从文本预测语义 token
	2. vocoder CTX-vec2wav，采用 contextual vocoding 将语义 token 转为 波形
5. 在 speech continuation and editing 上实现了 SOTA

## Introduction

1. speech continuation 和 editing 任务描述

![](image/Pasted%20image%2020231013154129.png)

2. 当前基于 discrete speech tokens 的 TTS 方法有三个限制：
	1. 大部分都是自回归模型，从而只能是  left-to-right 方向，不适用于 speech editing
	2. 声学 token 的构建涉及 RVQ，需要多次索引
	3. 音频质量受限于 audio codec 模型
3. 提出 UniCATS

![](image/Pasted%20image%2020231013154633.png)

其中：
+ CTX-txt2vec 采用 contextual VQ-diffusion 从输入文本中预测语义 token
+ vocoder CTX-vec2wav 用 contextual vocoding 将语义 token 转为 波形，同时也会考虑 acoustic context（尤其是 speaker  identity）

## UniCATS

### CTX-txt2vec with Contextual VQ-diffusion

采用 vq-wav2vec token 作为语义 token。

考虑离散样本序列 $\boldsymbol{x}_0=[x_0^{(1)},x_0^{(2)},...,x_0^{(l)}]\mathrm{~where~}x_0^{(i)}\in\{1,2,...,K\}$，在每个 forward step 中吗，对 $\boldsymbol{x}_0$ 的每个样本执行 mask、替换 或保持不变 三种操作，最终得到的样本记为 $\boldsymbol{x}_t$，此时 forward 过程为：
$$q(x_t|x_{t-1})=\boldsymbol{v}^\top(x_t)\boldsymbol{Q}_t\boldsymbol{v}(x_{t-1})$$
其中 $\boldsymbol{v}(x_t)\in\mathbb{R}^{(K+1)}$ 表示 $x_t=k$ 的位置 为 1（其他位置为 0）的 one-hot 向量。$K+1$ 的 $1$ 表示 $[mask]$ token，$\boldsymbol{Q}_t\in\mathbb{R}^{(K+1)\times(K+1)}$ 为传输矩阵。多个 forward 过程得到：
$$q(x_t|x_0)=\boldsymbol{v}^\top(x_t)\overline{\boldsymbol{Q}}_t\boldsymbol{\upsilon}(x_0)$$
采用贝叶斯规则也可以计算：
$$\begin{aligned}
\begin{aligned}q(x_{t-1}|x_t,x_0)=\frac{q(x_t|x_{t-1},x_0)q(x_{t-1}|x_0)}{q(x_t|x_0)}\end{aligned} \\
=\frac{\left(\boldsymbol{v}^\top(x_t)\boldsymbol{Q}_t\boldsymbol{v}(x_{t-1})\right)\left(\boldsymbol{v}^\top(x_{t-1})\overline{\boldsymbol{Q}}_{t-1}\boldsymbol{v}(x_0)\right)}{\boldsymbol{v}^\top(x_t)\overline{\boldsymbol{Q}}_t\boldsymbol{v}(x_0)}.
\end{aligned}$$
 采用堆叠的 Transformer blocks  来构建 VQ-diffusion，模型用于估计分布 $p_\theta(\tilde{\boldsymbol{x}}_0|\boldsymbol{x}_t,\boldsymbol{y})$，推理的时候采样过程如下：
 $$p_\theta(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t,\boldsymbol{y})=\sum_{\tilde{\boldsymbol{x}}_0}q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t,\tilde{\boldsymbol{x}}_0)p_\theta(\widetilde{\boldsymbol{x}}_0|\boldsymbol{x}_t,\boldsymbol{y}).$$
这里的 $\boldsymbol{y}$ 为输入的文本。

由于任务是 speech editing and continuation，还需要考虑额外的 context token $\boldsymbol{c}^A,\boldsymbol{c}^B$，也就是要建模概率：
$$p_\theta(\tilde{\boldsymbol{x}}_0|\boldsymbol{x}_t,\boldsymbol{y},\boldsymbol{c}^A,\boldsymbol{c}^B)$$
于是提出以时间顺序将 $\boldsymbol{x}_t$ 和 $\boldsymbol{c}^A,\boldsymbol{c}^B$ 进行拼接，得到 $[\boldsymbol{c}^A,\boldsymbol{x}_t,\boldsymbol{c}^B]$ ，然后再送入到 VQ-diffusion 模型中，此时后验分布计算为：
$$\begin{aligned}
&p_\theta(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t,\boldsymbol{y},\boldsymbol{c}^A,\boldsymbol{c}^B) \\
&=\sum_{\widetilde{\boldsymbol{x}}_0}q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t,\widetilde{\boldsymbol{x}}_0)p_\theta(\widetilde{\boldsymbol{x}}_0|\boldsymbol{x}_t,\boldsymbol{y},\boldsymbol{c}^A,\boldsymbol{c}^B).
\end{aligned}$$
> 这里的 $\boldsymbol{c}^A,\boldsymbol{c}^B$ 其实就是前面、后面的语音。

其模型架构如下：
![](image/Pasted%20image%2020231013202848.png)
采用了一个长度和输入相同的 binary indicator sequence 来区分数据和 context。

VQ- diffusion 的结构和 [Vector Quantized Diffusion Model for Text-to-Image Synthesis 笔记](../图像合成系列论文阅读笔记/Vector%20Quantized%20Diffusion%20Model%20for%20Text-to-Image%20Synthesis%20笔记.md) 一样，但是没有使用 cross attention，而是直接把 $\boldsymbol{h}$ 加入到 self-attention layers 的输出，从而实现 $\boldsymbol{h}$ 和语义 token 之间严格的对齐。最后通过 softmax 来得到 $p_\theta(\tilde{\boldsymbol{x}}_0|\boldsymbol{x}_t,\boldsymbol{y},\boldsymbol{c}^A,\boldsymbol{c}^B)$ 的分布。

损失函数为：
$$\mathcal{L}_{\text{CTX-txt}2\text{vec}} = \mathcal{L}_{\text{duration}} + \gamma \mathcal{L}_{\text{VQ-diffusion}}$$


推理流程如下：
![](image/Pasted%20image%2020231013210003.png)

### CTX-vec2wav with Contextual Vocoding

模型架构如图：
![](image/Pasted%20image%2020231013210145.png)

首先将语义 token 通过两层 semantic encoders，然后通过 generator（卷积+上采样）得到波形。中间还有一个辅助的 feature adaptor，其功能类似于 FastSpeech 2 中的 variance adaptor。训练的时候，还使用了 ground-truth auxiliary features 作为条件，模型学习从第一个 encoder 的输出来预测这些条件。推理的时候，把这些辅助特征作为条件。

语义 token 主要捕获发音信息但是缺乏足够的声学细节，尤其是 speaker identify，于是提出采用  mel 谱 $m$ 来 prompt 声学 context。从而 semantic encoder 中的 cross attention 可以将来自 mel 谱 中的 声学 context 进行集成，从而可以提高 speaker similarity。
> 简单来说就是通过 mel 谱 来引入一些说话人的信息。

### Unified framework for context-aware TTS

speech continuation and editing 和唯一区别在于 context $B$ 是否存在。前面的描述中去掉 $B$ 之后就可以实现 speech continuation。

## 实验和结果