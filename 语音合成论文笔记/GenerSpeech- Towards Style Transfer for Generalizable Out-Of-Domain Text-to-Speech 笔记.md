> NIPS 2022，RenYi

1. out-of-domain（OOD）场景下的风格迁移语音合成用于从 acoustic reference 生成未知风格下的语音样本，挑战如下：
	1. expressive voice 中，style feature 的动态范围很高，难以建模和迁移
	2. TTS 模型应该在 OOD 条件下鲁棒
2. 提出 GenerSpeech，用于 OOD 场景下的，高保真的 zero-shot 风格迁移 TTS
3. 通过引入两个模块，将语音分为 style-agnostic 和 style- specific 两个部分：
	1. multi-level style adaptor 来建模 style，包含 global speaker、emotion characteristics、local (utterance, phoneme, and word-level) fine-grained prosodic representations
	2. generalizable content adaptor，采用 Mix-Style Layer Normalization 来消除内容表征中的 style 信息
4. zero-shot style transfer 实验表明，在合成质量和风格相似度方面超过 SOTA 

## Introduction

1. 现有的风格建模和迁移工作：
	1. Global style token
	2. Fine-grained latent variables：用 VAE 来表征 fine-grained prosody variable，从而可以对 prosody 进行采样
	3. 但是这些模型在捕获不同的风格特性时都受限，无法同时反应出正确的 speaker identity, emotion 和 prosody
2. 现有的面向 OOD custom voice 的模型泛化工作：
	1. 数据驱动：在大规模数据集中做预训练
	2. 风格自适应：在 diverse acoustic 条件下用有限的 adaptation data 做 finetuning；也有采用 meta-learning 的，但是这个不能实现 zero-shot
3. 提出 GenerSpeech，可以分别控制语音中的 style-agnostic (linguistic content) 和 style-specific (speaker identity, emotion, and prosody) variations：
	1. Multi-level style adaptor，用 wav2vec 2.0 encoder 来生成 global latent representations 以控制 speaker 和 emotion characteristics；然后用三个不同的 local style encoders 来建模 fine-grained frame, phoneme 和 word-level prosodic representation
	2. Generalizable content adaptor，提出 mix-style layer normalization (MSLN) 来消除 content representation 中的风格属性，使其预测 style-agnostic 特征

## 相关工作（略）

## GenerSpeech

Style transfer of out-of-domain (OOD) custom voic（用于 OOD 定制化语音的风格迁移）：从 reference utterance 提取未知的 style，基于此 style 来合成高质量、高相似度的语音。

### 概览

采用 FastSpeech 2 作为 backbone，整体架构如图：
![](image/Pasted%20image%2020240124110302.png)

启发：通过 disentangled representation learning 将模型分为 domain-agnostic 和 domain-specific 的部分。

于是设计模型来独立建模 style-agnostic 和 style-specific variation。

### Generalizable Content Adaptor

通过 Mix-Style Layer Normalization 消除 phonetic sequences 中的 style information。

#### Mix-Style Layer Normalization

已有工作表明，layer normalization 会通过 learnable scale vector $\gamma$ 和 bias $\beta$ 极大地影响 hidden activation 和 final prediction：$\mathrm{LN}(x)=\gamma\frac{x-\mu}\sigma+\beta$，其中 $\mu,\sigma$ 为 hidden vector 的均值和方差。而用于 speaker adaptation 的 conditional layer normalization CLN 则可以自适应地对输入特征进行缩放和偏移，其中 $w$ 为 style embedding：$\mathrm{CLN}(x,w)=\gamma(w)\frac{x-\mu}\sigma+\beta(w)$。两个简单的线性层 $E_\gamma$ 和 $E_\beta$ 以 style embedding 为输入，输出 scale 和 bias vector：
$$\gamma(w)=E^\gamma*w,\quad\beta(w)=E^\delta*w$$

但是 source 和 target domain 之间的差异会阻碍 TTS 模型的泛化能力。为了分离 style infomation 和学习 style-agnostic representation，一种方法是在 mismatched style information 的条件下 refine sequence，可被视为注入噪声来 confuse 模型并防止其生成一致的 style representation。本文设计了 Mix-Style Layer Normalization，通过 perturbe style information 来规范化 TTS 模型的训练：
$$\gamma_\mathrm{mix}(w)=\lambda\gamma(w)+(1-\lambda)\gamma(\tilde{w})\quad\beta_\mathrm{mix}(w)=\lambda\beta(w)+(1-\lambda)\beta(\tilde{w})$$
其中 $w$ 表示 style vector，$\tilde{w}$ 通过 $\tilde{w}=Shuffle(w)$ 得到（图 b）。$\lambda\in\mathbb{R}^B$ 从 Beta 分布中采样，$B$ 为 batch size。$\lambda\sim\mathrm{Beta}(\alpha,\alpha)$，其中 $\alpha\in(0,\infty)$ 为原始 style 和 shuffle style 之间的 trade-off，本文中设置 $\alpha=0.2$。最终，generalizable style-agnostic hidden representations 变为：
$$\text{Mix-StyleLN}(x,w)=\gamma_{\mathrm{mix}}(w)\frac{x-\mu}\sigma+\beta_{\mathrm{mix}}(w)$$
模型通过 perturbed style 来 refine input features 并学习 generalizable style-invariant content representation。为了进一步确保 diversity 和避免过拟合，还通过随机混合 shuffle vectors 来 perturb style information，其中 shuffle rate $\lambda$ 从 Beta 分布中采样。

通过在 generalizable content adaptor 中使用 Mix-Style Layer Normalization，linguistic content-related variation 可以从 global style attributes（speaker 和 emotion）中分离出来，从而提高了模型 OOD custom style 的泛化性。

### Multi-level Style Adaptor

OOD custom voice 通常包含高动态的 style attributes（speaker identities, prosodies, and emotions），提出 multi-level style adaptor （图 d）来进行 global 和 local stylization。

#### Global Representation

使用 generalizable wav2vec 2.0 模型来捕获 global style characteristics，包括 speaker 和 emotion acoustic conditions。
> [wav2vec 2.0- A Framework for Self-Supervised Learning of Speech Representations 笔记](../语音自监督模型论文阅读笔记/wav2vec%202.0-%20A%20Framework%20for%20Self-Supervised%20Learning%20of%20Speech%20Representations%20笔记.md)

在 wav2vec 2.0 encoder 顶部添加 average pooling layer 和 fully-connected layers，从而可以在 speaker 和 emotion classification 任务上 finetuning 模型。采用 AM-softmax 作为分类损失函数。由于 multi-speaker 和 multi-emotion 的数据有限，发现分别使用不同的数据集来 finetuning global encoders 效果更好。
> 其实就是 finetuned wav2vec 2.0 来生成 global representations $\mathcal{G}_s$ 和 $\mathcal{G}_e$，分别用于建模 speaker 和 emotion characteristics

#### Local Representation

为了捕获细粒度的 prosodic details，考虑 frame, phoneme, and word-level 三种 differential acoustic conditions。这些 multi-level style encoders 共享一个通用的架构：
+ 输入序列通过几个卷积层进行 refine。也可进行 pooling 操作来进行不同层次的 stylization。pooling 操作根据输入边界对每个 representation 中的 hidden states 进行平均
+ 然后将 refine 后的序列作为 bottleneck 传入 VQ 层来消除 non-prosodic information。

三种 prosodic conditions 为：
1. Frame level：去掉 local style encoder 中的 optional pooling layer（因为是最小的 level，不需要 pooling） 来得到 frame-level latent representation $\mathcal{S}_p$
2. Phoneme level：考虑到 pitch 和 stress 的升降，phoneme 之间的 style pattern 动态范围很大。将 phoneme boundary 作为额外的输入，在通过 VQ 层之前做 pooling，得到 phoneme-level style latent representation $\mathcal{S}_p$
3. Word level：类似于 phoneme-level stylization，每个 word 上的 acoustic conditions（如 pitch 和 stress）也是高度变化的。将 word boundary 作为额外的输入，在通过 VQ 层之前做 pooling，得到 word-level style latent representation $\mathcal{S}_w$

在 VQ 层，refined sequences 作为 bottleneck 传入 VQ 层来消除 non-style information。VQ 层采用 information bottleneck，定义 latent embedding space $e\in\mathbb{R}^{K\times D}$，其中 $K$ 为离散 latent space 的大小（embedding 的个数），$D$ 为每个 latent embedding vector $e_i$ 的维度。然后和之前的 VQ 一样添加了一个 commitment loss：
$$\mathcal{L}_c=\left\|z_e(x)-\operatorname{sg}[e]\right\|_2^2$$
其中 $z_e(x)$ 为 VQ 层的输出，$\operatorname{sg}$ 为 stop gradient operator。

#### Style-To-Content Alignment Layer

为了将可变长度的 local style representations（即 $\mathcal{S}_u,\mathcal{S}_p,\mathcal{S}_w$）与 phonetic representations $\mathcal{H}_c$ 对齐，引入 Style-To-Content Alignment Layer 来学习 style 和 content 之间的 alignment（图 e）。

采用 Scaled Dot-Product Attention 作为 attention module。以 frame-level style encoder 中的 module 为例，其中 $\mathcal{H}_c$ 作为 query，$\mathcal{S}_u$ 作为 key 和 value：
$$\mathrm{Attention}(Q,K,V)=\mathrm{Attention}(\mathcal{H}_c,\mathcal{S}_u,\mathcal{S}_u)=\mathrm{Softmax}(\frac{\mathcal{H}_c\mathcal{S}_u^T}{\sqrt{d}})\mathcal{S}_u$$
还为 style representations 添加了 positional encoding embedding。为了高效训练，使用 residual connection 将 $\mathcal{H}_c$ 添加到 $\mathrm{Attention}(\mathcal{H}_c,\mathcal{S}_u,\mathcal{S}_u)$ 中。

最后，使用 pitch predictor 来生成 style-specific prosodic variations。

### Flow-based Post-Net

expressive custom voices 通常包含丰富的高动态变化，而 transformer decoder 很难生成这样细节丰富的 mel-spectrogram。为了进一步提高合成 mel-spectrogram 的质量和相似度，引入 flow-based post-net 来 refine mel-spectrogram decoder 的输出。

post-net 的架构用的是 Glow 中的。训练时，flow post-net 将合成的 mel-spectrogram 转换为 gaussian prior distribution，并计算精确似然。推理时，从 prior distribution 中采样 latent variables，并将其反向传入 post-net 以生成 expressive mel-spectrogram。

### 预训练、训练和推理

预训练阶段，采用 AM soft-max loss 来 finetune global style encoder wav2vec 2.0，然后将练好的 wav2vec 2.0 模型的知识进行迁移来生成 global style features。

训练 GenerSpeech 时，reference 和 target speech 保持不变。最终的损失包括：
+ duration prediction loss $\mathcal{L}_\mathrm{dur}$：预测的 phoneme-level duration 和 GT 之间的 MSE
+ mel 重构损失 $\mathcal{L}_\mathrm{mel}$：GT mel 谱和 transformer decoder 生成的 mel 谱之间的 MAE
+ pitch 重构损失 $\mathcal{L}_\mathrm{p}$：GT 和 style-agnostic 和 style-specific pitch predictor 预测的 joint pitch spectrogram 之间的 MSE
+ post-net 的负对数似然 $\mathcal{L}_\mathrm{pn}$
+ commit loss $\mathcal{L}_\mathrm{c}$

推理时，GenerSpeech 的 pipeline 如下：
+ text encoder 编码 phoneme sequence，根据 inference duration 得到 expanded representations $\mathcal{H}_c$
+ style-agnostic pitch (SAP) predictor 生成 invariant to custom style 的 linguistic content speech variation
+ 给定参考语音，通过 forced alignment 得到 word 和 phoneme boundaries，然后输入 multi-level style adaptor 来建模 style latent representations：
    + wav2vec 2.0 生成 speaker $\mathcal{G}_s$ 和 emotion $\mathcal{G}_e$ 来控制 global style
    + local style encoder 捕获 frame, phoneme 和 word-level fine-grained style representations $\mathcal{S}_u,\mathcal{S}_p,\mathcal{S}_w$
    + style-specific pitch (SSP) predictor 生成 style-sensitive variation
+ mel decoder 生成 coarse-grained mel-spectrograms $\tilde{\mathcal{M}}$，然后 flow-based post-net 将随机采样的 latent variables 转换为以 $\tilde{\mathcal{M}}$ 和 mel decoder input 为条件的 fine-grained mel-spectrograms $\mathcal{M}$

## 实验（略）