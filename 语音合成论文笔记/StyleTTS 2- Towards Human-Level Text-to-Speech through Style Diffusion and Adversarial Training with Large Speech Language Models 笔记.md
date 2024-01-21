> NIPS 2023，preprint，哥伦比亚大学

1. 提出 StyleTTS 2，采用 style diffusion 和 对抗训练 和 speech language model 实现 human-level 的 TTS
2. 通过 diffusion 把 style 建模为 latent random variable 来从文本中生成 style 而无需参考语音
3. 采用 Large pre-trained SLMs（如 WavLM）作为 discriminators，和一个新的可微分的 duration modeling 实现端到端的训练

> 其实就是在 1 的基础上，用 WavLM 模型作为 discriminator，然后用 diffusion 从文本中建模韵律，从而不需要参考语音作为输入；然后还引入了一种新的可微的 duration 建模。

## Introduction

1. 提出 StyleTTS 2：
	1. 通过 diffusion 把风格建模为  latent random variable
	2. 采用 预训练的 speech language models (SLMs) 作为 discriminators
2. 在 LJSpeech 上 CMOS 很高，比 NaturalSpeech 效果也好，在多说话人数据集上效果也很好，在自然度上超过了 VALL-E

## 相关工作（略）

## 方法

### StyleTTS 概览

见 [StyleTTS- A Style-Based Generative Model forNatural and Diverse Text-to-Speech Synthesis 笔记](StyleTTS-%20A%20Style-Based%20Generative%20Model%20forNatural%20and%20Diverse%20Text-to-Speech%20Synthesis%20笔记.md)

StyleTTS 有几个缺点：
+ 两阶段可能会降低合成质量和表达性
+ 需要一个 参考语音，从而不能实时应用

### StyleTTS 2

引入一个端到端的训练方法，可以联合优化所有的模块，包括波形合成和对抗训练。

style 建模为从 diffusion 中采样的 latent variable，从而可以实现多样性的合成而无需参考音频。
![](image/Pasted%20image%2020231122163518.png)

### 端到端训练

将原来的 decoder $G$ 修改成可以从 style vector, aligned phoneme representations, 和 pitch、energy curves 中直接生成波形（而非生成 mel 谱），其实就是移除最后一层的 投影层，然后接上 waveform decoder，有两种选择：
+ HifiGAN-based
+ iSTFTNet-based

然后训练 decoder 的时候，将 mel-discriminator 替换为 multi-period discriminator (MPD) 和 multi-resolution discriminator (MRD)，损失函数替换为 LSGAN 的损失。同时加入 truncated pointwise relativistic loss function 来增强声音的质量。

发现预训练一部分声学模块可以加速训练过程（对应图中 蓝色 部分）。于是首先预训练 升学模块 和pitch extractor 和 text aligner。
> 作者在论文中自己也承认了，这其实就和 style tts 中的 阶段 1 的训练一致。
> 但是又解释道，这种预训练不是必须的，从零训练会变慢，但是也会收敛。

然后联合优化原来的所有的损失来训练整个模型（所谓端到端训练）。

但是训练过程中，style encoder 需要同时编码声学和韵律信息，因此会不稳定。于是引入 prosodic style encoder $E_p$ （这个是新引入的）+ acoustic style encoder $E_a$（对应之前的 $E$），此时 Duration Predictor $S$ 和 Prosody Predictor $P$ 的输入不是 $\boldsymbol{s}_a=E_a(\boldsymbol{x})$，而是 $\boldsymbol{s}_{p}=E_{p}(\boldsymbol{x})$。style diffusion 模型生成增强后的 style vector $\boldsymbol{s}=\begin{bmatrix}\boldsymbol{s}_p,\boldsymbol{s}_a\end{bmatrix}$。

为了进一步解耦声学模块和 predictors，将来自 Text Encoder $T$ 的 $\boldsymbol{h}_{\text{text}}$ 替换为来另一个来自 BERT 模型的 Text encoder $B$，从而得到所谓的 $\boldsymbol{h}_{\text{bert}}$，这里称之为 prosodic text encoder。
> 注意：这里是新增了一个模块，不是替换原来的模块。
> 具体就是用了一个在 phoneme-level 上预训练的 BERT 模型。

### Style Diffusion

语音 $\boldsymbol{x}$ 可以通过 latent variable $\boldsymbol{s}$ 建模为条件分布 $p(\boldsymbol{x}|\boldsymbol{t})=\int p(\boldsymbol{x}|\boldsymbol{t},\boldsymbol{s})p(\boldsymbol{s}|\boldsymbol{t})$，而 $\boldsymbol{s}$ 也是一个关于文本的条件分布 $p(\boldsymbol{s}|\boldsymbol{t})$，称之为  generalized speech style，代表了除语音内容 $\boldsymbol{t}$ 之外的任何特征，包括但不限于 prosody, lexical stress, formant transitions, 和 speaking rate。

通过组合  probability flow 和 time-varying Langevin dynamics，利用 EDM 采样得到 $\boldsymbol{s}$：
$$s=\int-\sigma(\tau)\left[\beta(\tau)\sigma(\tau)+\dot{\sigma}(\tau)\right]\nabla_s\log p_\tau(\boldsymbol{s}|\boldsymbol{t})\:d\tau+\int\sqrt{2\beta(\tau)}\sigma(\tau)\:d\tilde{W}_\tau$$
> 讲这么多，其实就是，这是一个 diffusion 模型，输入为某种噪声，条件为 $\boldsymbol{h}_{\text{bert}}$，输出为 $\boldsymbol{s}=\begin{bmatrix}\boldsymbol{s}_p,\boldsymbol{s}_a\end{bmatrix}$，其他的操作只是在改进 diffusion 或者提高采样速度，而且也都不是作者提出的。

### SLM Discriminators

采用  12-layer WavLM $W$ 作为 discriminator。但是固定其参数，然后加上一个 CNN $C$ 的 discriminative head，记为 $D_{SLM}=C\circ W$。输入语音下采样到 16K $h_{\mathrm{SLM}}=W(\boldsymbol{x})$，然后通过 linear map 将其转为 256 维。

优化的对抗损失为：
$$\mathcal{L}_{slm}=\min_{\boldsymbol{G}}\max_{D_{SLM}}\left(\mathbb{E}_{\boldsymbol{x}}[\log D_{SLM}(\boldsymbol{x})]+\mathbb{E}_{\boldsymbol{t}}[\log\left(1-D_{SLM}(\boldsymbol{G}(\boldsymbol{t}))\right)]\right)$$
其中 $\boldsymbol{G}(\boldsymbol{t})$ 为生成的语音，$\boldsymbol{x}$ 为真实的语音。
> 由于  discriminator 的大部分参数都是固定的，这个过程其实相当于在训练 generator ？

训练的时候，为了避免过拟合，从 分布内 和 OOD 文本中以相同的概率采样。

### Differentiable Duration Modeling

由于采用了 MAS，计算对齐的过程中是不可微的。

NaturalSpeech 用的是 attention-based upsampler，但是这种方法在对抗训练的时候不稳定。于是需要一个非参数化（non-parametric）的上采样方法。

Gaussian upsampling 是 non-parametric 的，可以使用高斯核，以超参数 $\sigma$ 将预测的 duration $d_{\mathrm{pred}}$ 转为 $a_\text{pred}{ [ n , i ]}$：
$$\mathcal{N}_{c_i}(n,\sigma):=\exp\left(-\frac{(n-c_i)^2}{2\sigma^2}\right),\quad\ell_i:=\sum_{k=1}^id_{\mathrm{pred}}[k]$$
其中心为 $c_i:=\ell_i-\frac12\boldsymbol{d}_{\text{pred}} [ i ]$，但是高斯核受限于其固定的宽度，从而无法正确地建模对齐。
> 见论文 [End-to-End Adversarial Text-to-Speech 笔记](End-to-End%20Adversarial%20Text-to-Speech%20笔记.md)。

于是提出一种新的 non-parametric differentiable upsampler，对于每个 phoneme $t_i$，将对齐建模为随机变量 $a_i\in\mathbb{N}$，用于表明 $t_i$ 对应的 语音帧 的索引，定义第 $i$ 个phoneme 的 duration为另一随机变量 $d_i\in\{1,\dots,L\},L=50$ 为最大的 duration。于是 $a_i=\sum_{k=1}^id_k$，$d_k$ 相互独立。

但是，通过估计 $a_i=d_i+\ell_{i-1}$，此时 $a_i$ 的 PMF 为：
$$f_{a_i}[n]=f_{d_i+\ell_{i-1}}[n]=f_{d_i}[n]*f_{\ell_{i-1}}[n]=\sum_kf_{d_i}[k]\cdot\delta_{\ell_{i-1}}[n-k]$$
> 相当于直接估计加权的权重的离散分布。

## 实验（略）