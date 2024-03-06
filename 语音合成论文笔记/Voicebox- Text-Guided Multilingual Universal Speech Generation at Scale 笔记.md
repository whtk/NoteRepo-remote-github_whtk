> NIPS 2023，Fundamental AI Research (FAIR), Meta

1. 提出 Voicebox，文本引导的大规模语音生成模型
2. 本质为自回归的 flow-matching 模型，给定音频上下文和文本来做填空题
3. 可以通过 in-context learning 实现多种任务，可以用于单语言或者跨语言的 zero shot TTS，效果超过 VALLE，且速度快 20 倍
<!-- 翻译&理解 -->
## Introduction
<!--  Previous works consider highly curated datasets such as VCTK [Yamagishi et al., 2019], which contains only clean audio recorded in studio from about 100 speakers with little speaking style and text variation. Such models struggle to synthesize speech with rich variation in emotion, voice, background noise, acoustic condition, and have not been tested on the abilities to generalize to tasks not explicitly trained on. -->
1. 之前的工作都是 VCTK 这种干净数据，风格和文本变化小，训练的模型难合成具有丰富情感、声音、背景噪音、声学条件变化的语音
<!-- There had been a few attempts of using in-the-wild data such as CommonVoice [Ardila et al., 2019], Librispeech [Panayotov et al., 2015], and LibriTTS [Zen et al., 2019] for training text-to-speech (TTS) models. It led to huge quality degradation compared to training on curated datasets [Hsu et al., 2019, Wang et al., 2021]. In particular, while in-the-wild data are generally of lower quality, the gap between synthesized and training speech is big compared to that of the models trained on curated speech [Wang et al., 2021], which suggests that previous models terribly underfit in-the-wild data. -->
2. 而用 CommonVoice、Librispeech、LibriTTS 训练的模型由于数据质量低，合成质量不行
<!-- This paper presents Voicebox, the most versatile text-conditioned speech generative model at scale. Voicebox is trained on a text-guided speech infilling task, where the goal is to generate masked speech given its surrounding audio and text transcript. This can be considered as a guided in-context learning problem, where audio style is inferred from the audio context and textual content is specified through transcript. Voicebox does not require any audio style labels (e.g., speaker, emotion, and noise), which differentiates Voicebox from the majority of prior work where such labels are used extensively. Prior work uses labels to make the mapping between input (text and audio style) and output (speech) more deterministic to reduce underfitting [Wang et al., 2021, Popov et al., 2021]. We show that Voicebox’s text-guided speech infilling approach is much more scalable in terms of data while subsuming many common speech generative tasks. -->
3. 提出 Voicebox，文本引导的语音生成模型，通过填空题的方式训练，不需要音频风格标签，可以用于多种任务
<!-- In terms of modeling, Voicebox is a non-autoregressive (NAR) continuous normalizing flow (CNF) model [Chen et al., 2018]. Similar to diffusion models [Ho et al., 2020], CNFs model the trans- formation from a simple distribution to a complex data distribution (p(missing data | context)), parameterized by a neural network. We train Voicebox with flow-matching [Lipman et al., 2023], a recently proposed method that enables efficient and scalable training of CNFs via a simple vector field regression loss. In contrast to auto-regressive models, Voicebox can consume context not only in the past but also in the future. Moreover, the number of flow steps can be controlled at inference time to flexibly trade off quality and runtime efficiency. -->
4. Voicebox 是一个非自回归的 continuous normalizing flow (CNF) 模型，通过 flow-matching 训练，可以在推理时控制流步数，灵活权衡质量和效率
<!-- Voicebox is trained on 60K hours of English audiobooks and 50K hours of multilingual audiobooks in 6 languages for the mono and multilingual setups. Voicebox achieves SOTA performance on mono-lingual/cross-lingual zero-shot TTS, speech denoising, speech editing, diverse speech sampling and an application to data creation for speech recognition. To tackle the lack of comparability due to the use of subjective metrics, this paper presents a series of metrics using public models to facilitate reproducible comparison and model development for speech generation studies. -->
5. Voicebox 在 60K 小时英文有声书和 6 种语言的 50K 小时有声书上训练，效果超过 SOTA，可以用于单语言/跨语言的 zero-shot TTS、语音降噪、语音编辑等
6. 本文还提出了一系列公开模型的指标，以便于 speech generation 研究的可重现性和模型开发
<!-- The contribution of this work can be summarized as follows:
1. Voicebox represents a breakthrough in generative modeling for speech. By learning to solve a text-guided speech infilling task with large scale data, Voicebox can solve tasks it was not explicitly trained to accomplish via in-context learning.
2. Voicebox outperforms VALL-E and achieves a new SOTA English zero-shot TTS result (5.9% → 1.9% on word error rate (WER) and 0.580 → 0.681 on audio similarity).
3. Voicebox is the first model that can perform high-quality cross-lingual zero-shot TTS across six languages. It does not use any style labels, pre-trained embedders, or multilingual samples. Compared to the prior cross-lingual SOTA YourTTS, Voicebox reduces the average WER from 10.9% to 5.2%, and improves audio similarity from 0.335 to 0.481.
4. Voicebox is capable of infilling speech of any length and outperforms the prior SOTA A3T, on text guided denoising with -8.8% WER, +0.450 similarity, and +0.80 mean opinion score.
5. Voicebox can generate diverse and realistic speech. An ASR system can be trained solely on synthetic speech generated by Voicebox, resulting in only 0.4%/1.7% absolute WER increase on Librispeech test-other/test-clean compared to training on real data. In contrast, previous TTS models suffer from at least 18.2%/44.5% absolute WER increase.
 -->
7. 贡献总结：
    1. Voicebox 通过大规模数据学习文本引导的语音填充任务，可以通过 in-context learning 解决未经训练的任务
    2. Voicebox 超过 VALL-E，实现了新的 SOTA 英文 zero-shot TTS 结果
    3. Voicebox 是第一个可以跨 6 种语言实现高质量 zero-shot TTS 的模型
    4. Voicebox 可以填充任意长度的语音，超过了 A3T 在文本引导降噪任务上的效果
    5. Voicebox 可以生成多样且逼真的语音

## 相关工作（略）

##  方法
<!-- Background: Flow Matching with an optimal transport path -->
### 背景：流匹配与最优传输路径
<!-- Let Rd be the data space with data points x ∈ Rd drawn from some unknown distribution q(x). Continuous Normalizing Flows (CNFs) Chen et al. [2018] are a family of generative models that learn the transformation from a simple prior distribution p0 (e.g., normal distribution) to the data distribution p1 ≈ q. CNFs parameterize a time-dependent vector field vt : [0, 1] × Rd → Rd that is used to construct a flow: φt : [0, 1] × Rd → Rd that pushes points from the prior towards the target distribution. The relationship between a vector field and a flow is defined via the ordinary differential equation (ODE) as: -->
设 $\mathbb{R}^d$ 为数据空间，$x \in \mathbb{R}^d$ 从未知分布 $q(x)$ 中采样。连续正则化流（CNFs）学习从简单先验分布 $p_0$（如正态分布）到数据分布 $p_1 \approx q$ 的变换。CNFs 参数化一个时间相关的向量场 $v_t: [0, 1] \times \mathbb{R}^d \rightarrow \mathbb{R}^d$，用于构建一个流 $\phi_t: [0, 1] \times \mathbb{R}^d \rightarrow \mathbb{R}^d$，将先验分布中的点推向目标分布。向量场和流的关系通过常微分方程（ODE）定义为：
$$\frac d{dt}\phi_t(x)=v_t(\phi_t(x));\quad\phi_0(x)=x$$
<!-- For a flow φt , the probability path (time-dependent probability density function) p : [0, 1] × Rd → R>0 can be derived via the change of variables formula: -->
对于流 $\phi_t$，概率路径（时间相关概率密度函数）$p: [0, 1] \times \mathbb{R}^d \rightarrow \mathbb{R}_{>0}$ 可以通过 change of variables formula 得到：
$$p_t(x)=p_0(\phi_t^{-1}(x))\det\left[\frac{\partial\phi_t^{-1}}{\partial x}(x)\right].$$
<!-- To sample from pt(x), we first draw x0 from p0 and then solve the initial value problem (IVP) for
φt(x0) given dφt(x)/dt = vt(φt(x)) and φ0(x) = x0. We use xt and φt(x0) interchangeably. -->
为了从 $p_t(x)$ 中采样，先从 $p_0$ 中采样 $x_0$，然后求解初值问题（IVP）$\phi_t(x_0)$ 给定 $\frac{d\phi_t(x)}{dt}=v_t(\phi_t(x))$ 和 $\phi_0(x)=x_0$。可以互换 $x_t$ 和 $\phi_t(x_0)$。
<!-- Let pt be a probability path and ut be the corresponding vector field that generates pt. The vector
field vt(x; θ) parameterized by a neural network θ can be trained with the Flow Matching objective: -->
设 $p_t$ 为概率路径，$u_t$ 为生成 $p_t$ 的对应向量场。由神经网络参数化的向量场 $v_t(x;\theta)$ 可以通过 Flow Matching 训练：
$$\mathcal{L}_{FM}(\theta)=\mathbb{E}_{t,p_t(x)}||u_t(x)-v_t(x;\theta)||^2$$
<!-- where t ∼ U [0, 1] and x ∼ pt (x). While the objective appears simple, in practice we do not have the prior knowledge of pt or vt, and cannot directly compute the loss or its gradient estimator. -->
其中 $t \sim \mathcal{U}[0, 1]$，$x \sim p_t(x)$。但由于没有 $p_t$ 或 $v_t$ 的先验知识，不能直接计算损失或其梯度估计。
<!-- Let x1 be a random variable distributed according to data distribution q. Lipman et al. [2023] first notes that a probability path pt(x) can be constructed via a mixture of simpler conditional paths pt(x | x1) whose vector field ut(x | x1) can be easily computed. To construct pt(x), a conditional path is defined such that 1) p0(x | x1) = p0(x) and 2) p1(x | x1) = N(x | x1,σ2I), a Gaussian distribution centered at x1 with a sufficiently small σ (typically 10−5). The marginal path is computed as R pt(x | x1)q(x1)dx1, which closely approximates q(x1) at t = 1. With that, [Lipman et al., 2023] presents the Conditional Flow Matching (CFM) objective, -->
Lipman 指出，可以通过较简单的条件路径 $p_t(x | x_1)$ 的混合构造概率路径 $p_t(x)$，其向量场 $u_t(x | x_1)$ 可以很容易计算。为了构造 $p_t(x)$，定义条件路径使得 1) $p_0(x | x_1) = p_0(x)$，2) $p_1(x | x_1) = N(x | x_1, \sigma^2I)$，一个以 $x_1$ 为中心的高斯分布，$\sigma$ 足够小（通常 $10^{-5}$）。边缘路径计算为 $\int p_t(x | x_1)q(x_1)dx_1$，在 $t=1$ 时近似 $q(x_1)$。此时，条件流匹配（CFM）目标函数为：
$$\mathcal{L}_{CFM}(\theta)=\mathbb{E}_{t,q(x_1),p_t(x|x_1)}||u_t(x\mid x_1)-v_t(x;\theta)||^2.$$
<!-- It is proven that FM and CFM have identical gradients w.r.t. θ. More importantly, one can easily
draw samples from pt(x | x1) and compute ut(x | x1) to derive an unbiased gradient estimator. -->
已被证明，FM 和 CFM 对 $\theta$ 的梯度相同。而且可以很容易从 $p_t(x | x_1)$ 中采样并计算 $u_t(x | x_1)$ 来得到无偏的梯度估计器。
<!-- The next question is how to choose a conditional flow. A flow defines trajectories, describing how each point moves between p0 and p1. Intuitively, a simpler trajectory (e.g., a straight line) can be learned faster and the IVP can be solved more accurately and efficiently. Lipman et al. [2023] presents a conditional flow called optimal transport (OT) path, which has the form of pt(x | x1) = N(x | tx1, (1 − (1 − σmin)t)2I) and ut(x | x1) = (x1 − (1 − σmin)x) / (1 − (1 − σmin)t). The flow is arguably simple because points move with a constant speed and direction. We adopt it for Voicebox
Lipman et al. [2023] also presents another flow that recovers the path of diffusion models [Song and Ermon, 2019], which is more complex than the OT path. We will present ablation studies comparing different paths (OT vs diffusion) and different objectives (CFM vs score-matching). Results show the superiority in performance and efficiency of CFM with OT path -->
下一个问题是如何选择条件流。流定义了轨迹，描述了每个点在 $p_0$ 和 $p_1$ 之间的移动。直观地，更简单的轨迹（如直线）可以更快地学习，IVP 可以更准确和高效地求解。Lipman 提出了一种称为最优传输（OT）路径的条件流，其形式为 $p_t(x | x_1) = N(x | tx_1, (1 - (1 - \sigma_{\min})t)^2I)$，$u_t(x | x_1) = (x_1 - (1 - \sigma_{\min})x) / (1 - (1 - \sigma_{\min})t)$。这种流是简单的，因为点以恒定的速度和方向移动。这里将其用于 Voicebox。
<!-- Problem formulation -->
### 问题定义
<!-- Given a dataset of transcribed speech (x, y) where x and y denote an audio sample and its transcript, respectively, the goal is to build a single model that can perform many text-guided speech generation tasks through in-context learning. We propose to train such a generative model on the text-guided speech infilling task, which predicts a segment of speech given its surrounding audio and the complete text transcript. Let m be a binary temporal mask which is of the same length as x, 3 and xmis = m⊙x and xctx = (1 − m) ⊙ x be the complementary masked versions of x. The generative model learns p(xmis | y, xctx). In other words, y and xctx are the context and xmis is the missing data.
 -->
给定 文本-语音 数据集 $(x, y)$，其中 $x$ 和 $y$ 分别表示音频样本和其文本，目标是构建一个模型，可以通过 in-context learning 执行多种文本引导的语音生成任务。提出在文本引导的语音填充任务上训练这样的生成模型，给定其周围音频和完整文本，预测语音段。设 $m$ 为二值时间掩码，与 $x$ 长度相同，$x_{\text{mis}} = m \odot x$ 和 $x_{\text{ctx}} = (1 - m) \odot x$ 为 $x$ 的互补掩码。模型学习 $p(x_{\text{mis}} | y, x_\text{ctx})$。换句话说，$y$ 和 $x_{\text{ctx}}$ 是上下文，$x_{\text{mis}}$ 是缺失数据。

### 模型和训练
<!-- Motivated by the need that some applications require fine-grained alignment control between speech and text, we decouple Voicebox into two components: an audio model and a duration model. Let x = (x1,x2,··· ,xN) be an audio sample of N frames, y = (y1,y2,··· ,yM) be a text sequence of M phones, and l = (l1, l2, · · · , lM ) be the per-phone duration where lj denotes how many audio
frames yj correspond to and PMj=1 lj = N. We further define z = rep(y,l) = (z1,z2,··· ,zN) to be the frame-level phone transcript, which repeats each yj by lj times such that zi denotes the phone
label of the audio frame xi. For a pair of (x, y), l and z can be estimated through forced alignment using a speech recognition model. The estimation of q(xmis | y, xctx) is then broken down into the audio model q(xmis | z, xctx) and the duration model q(lmis | y, lctx), where lmis and lctx denote l masked by m′ and 1 − m′, and m′ is downsampled from m based on l where m = rep(m′, l) -->
将 Voicebox 分为两部分：音频模型和时长模型。

设 $x = (x_1, x_2, \cdots, x_N)$ 为 $N$ 帧的音频，$y = (y_1, y_2, \cdots, y_M)$ 为 $M$ 个音素的文本序列，$l = (l_1, l_2, \cdots, l_M)$ 为每个音素的持续时间，$l_j$ 表示 $y_j$ 对应多少音频帧，且 $\sum_{j=1}^M l_j = N$。定义 $z = \text{rep}(y, l) = (z_1, z_2, \cdots, z_N)$ 为每帧对应的音素，即将 $y_j$ 重复 $l_j$ 次（$z_i$ 其实就是音频帧 $x_i$ 的音素标签）。对于一对 $(x, y)$，$l$ 和 $z$ 可以使用语音识别模型估计。$q(x_{\text{mis}} | y, x_{\text{ctx}})$ 的估计分解为音频模型 $q(x_{\text{mis}} | z, x_{\text{ctx}})$ 和时长模型 $q(l_{\text{mis}} | y, l_{\text{ctx}})$，其中 $l_{\text{mis}}$ 和 $l_{\text{ctx}}$ 表示 $l$ 被 $m'$ 和 $1 - m'$ 掩码，$m'$ 是从 $m$ 中基于 $l$ 下采样的，其中$m = \text{rep}(m', l)$。

#### 音频模型
<!-- Given a context z and xctx of length N, the distribution of xmis is highly stochastic especially when xmis has a large temporal span. Hence, we parameterize it with a CNF and train it using the flow matching objective with the optimal transport path. Audio x is represented as an 80-dimensional log Mel spectrogram (xi ∈ R80) extracted at a 100Hz frame rate.4 The audio contextxictx =0wheremi =1andxictx =xi wheremi =0. -->
给定长度为 $N$ 的上下文 $z$ 和 $x_{\text{ctx}}$，$x_{\text{mis}}$ 的分布是高度随机的，特别是当 $x_{\text{mis}}$ 有很大的时间跨度时。因此，使用 CNF 参数化，并使用最优传输路径的流匹配目标训练。音频 $x$ 表示为 80 维对数 Mel 频谱图（$x_i \in \mathbb{R}^{80}$），在 100Hz 帧率下提取。音频上下文 $x_{\text{ctx}} = 0$ 当 $m_i = 1$，$x_{\text{ctx}} = x_i$ 当 $m_i = 0$。