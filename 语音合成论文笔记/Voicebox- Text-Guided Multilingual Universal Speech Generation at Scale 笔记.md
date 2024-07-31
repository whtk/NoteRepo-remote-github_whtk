> NIPS 2023，Fundamental AI Research (FAIR), Meta

1. 提出 Voicebox，文本引导的大规模语音生成模型
2. 本质为自回归的 flow-matching 模型，给定音频上下文和文本来做填空题
3. 可以通过 in-context learning 实现多种任务，可以用于单语言或者跨语言的 zero shot TTS，效果超过 VALL-E，且速度快 20 倍

## Introduction

1. 之前的工作都是 VCTK 这种干净数据，风格和文本变化小，训练的模型难合成具有丰富情感、声音、背景噪音、声学条件变化的语音
2. 而用 CommonVoice、Librispeech、LibriTTS 训练的模型由于数据质量低，合成质量不行
3. 提出 Voicebox，文本引导的语音生成模型，通过填空题的方式训练，不需要音频风格标签，可以用于多种任务
4. Voicebox 是一个非自回归的 continuous normalizing flow (CNF) 模型，通过 flow-matching 训练，可以在推理时控制流步数，灵活权衡质量和效率
5. Voicebox 在 60K 小时英文有声书和 6 种语言的 50K 小时有声书上训练，效果超过 SOTA，可以用于单语言/跨语言的 zero-shot TTS、语音降噪、语音编辑等
6. 本文还提出了一系列公开模型的指标，以便于 speech generation 研究的可重现性和模型开发
7. 贡献总结：
    1. Voicebox 通过大规模数据学习文本引导的语音填充任务，可以通过 in-context learning 解决未经训练的任务
    2. Voicebox 超过 VALL-E，实现了新的 SOTA 英文 zero-shot TTS 结果
    3. Voicebox 是第一个可以跨 6 种语言实现高质量 zero-shot TTS 的模型
    4. Voicebox 可以填充任意长度的语音，超过了 A3T 在文本引导降噪任务上的效果
    5. Voicebox 可以生成多样且逼真的语音

## 相关工作（略）

##  方法

### 背景：流匹配与最优传输路径

设 $\mathbb{R}^d$ 为数据空间，$x \in \mathbb{R}^d$ 从未知分布 $q(x)$ 中采样。连续正则化流（CNFs）学习从简单先验分布 $p_0$（如正态分布）到数据分布 $p_1 \approx q$ 的变换。CNFs 参数化一个时间相关的向量场 $v_t: [0, 1] \times \mathbb{R}^d \rightarrow \mathbb{R}^d$，用于构建一个流 $\phi_t: [0, 1] \times \mathbb{R}^d \rightarrow \mathbb{R}^d$，将先验分布中的点推向目标分布。向量场和流的关系通过常微分方程（ODE）定义为：
$$\frac d{dt}\phi_t(x)=v_t(\phi_t(x));\quad\phi_0(x)=x$$

对于流 $\phi_t$，概率路径（时间相关概率密度函数）$p: [0, 1] \times \mathbb{R}^d \rightarrow \mathbb{R}_{>0}$ 可以通过 change of variables formula 得到：
$$p_t(x)=p_0(\phi_t^{-1}(x))\det\left[\frac{\partial\phi_t^{-1}}{\partial x}(x)\right].$$

为了从 $p_t(x)$ 中采样，先从 $p_0$ 中采样 $x_0$，然后求解初值问题（IVP）$\phi_t(x_0)$ 给定 $\frac{d\phi_t(x)}{dt}=v_t(\phi_t(x))$ 和 $\phi_0(x)=x_0$。可以互换 $x_t$ 和 $\phi_t(x_0)$。

设 $p_t$ 为概率路径，$u_t$ 为生成 $p_t$ 的对应向量场。由神经网络参数化的向量场 $v_t(x;\theta)$ 可以通过 Flow Matching 训练：
$$\mathcal{L}_{FM}(\theta)=\mathbb{E}_{t,p_t(x)}||u_t(x)-v_t(x;\theta)||^2$$
其中 $t \sim \mathcal{U}[0, 1]$，$x \sim p_t(x)$。但由于没有 $p_t$ 或 $v_t$ 的先验知识，不能直接计算损失或其梯度估计。

Lipman 指出，可以通过较简单的条件路径 $p_t(x | x_1)$ 的混合构造概率路径 $p_t(x)$，其向量场 $u_t(x | x_1)$ 可以很容易计算。为了构造 $p_t(x)$，定义条件路径使得 1) $p_0(x | x_1) = p_0(x)$，2) $p_1(x | x_1) = N(x | x_1, \sigma^2I)$，一个以 $x_1$ 为中心的高斯分布，$\sigma$ 足够小（通常 $10^{-5}$）。边缘路径计算为 $\int p_t(x | x_1)q(x_1)dx_1$，在 $t=1$ 时近似 $q(x_1)$。此时，条件流匹配（CFM）目标函数为：
$$\mathcal{L}_{CFM}(\theta)=\mathbb{E}_{t,q(x_1),p_t(x|x_1)}||u_t(x\mid x_1)-v_t(x;\theta)||^2.$$
已被证明，FM 和 CFM 对 $\theta$ 的梯度相同。而且可以很容易从 $p_t(x | x_1)$ 中采样并计算 $u_t(x | x_1)$ 来得到无偏的梯度估计器。

下一个问题是如何选择条件流。流定义了轨迹，描述了每个点在 $p_0$ 和 $p_1$ 之间的移动。直观地，更简单的轨迹（如直线）可以更快地学习，IVP 可以更准确和高效地求解。Lipman 提出了一种称为最优传输（OT）路径的条件流，其形式为 $p_t(x | x_1) = N(x | tx_1, (1 - (1 - \sigma_{\min})t)^2I)$，$u_t(x | x_1) = (x_1 - (1 - \sigma_{\min})x) / (1 - (1 - \sigma_{\min})t)$。这种流是简单的，因为点以恒定的速度和方向移动。这里将其用于 Voicebox。

### 问题定义

给定 文本-语音 数据集 $(x, y)$，其中 $x$ 和 $y$ 分别表示音频样本和其文本，目标是构建一个模型，可以通过 in-context learning 执行多种文本引导的语音生成任务。提出在文本引导的语音填充任务上训练这样的生成模型，给定其周围音频和完整的对应的文本，预测语音段。
> 这个思路和 VALL-E 是一样的，和 GPT-SoVITS 也是一样的。

设 $m$ 为二值时间掩码，与 $x$ 长度相同，$x_{\text{mis}} = m \odot x$ 和 $x_{\text{ctx}} = (1 - m) \odot x$ 为 $x$ 的互补掩码。模型学习 $p(x_{\text{mis}} | y, x_\text{ctx})$。换句话说，$y$ 和 $x_{\text{ctx}}$ 是上下文，$x_{\text{mis}}$ 是缺失数据。

### 模型和训练

将 Voicebox 分为两部分：音频模型和时长模型。

设 $x = (x_1, x_2, \cdots, x_N)$ 为 $N$ 帧的音频，$y = (y_1, y_2, \cdots, y_M)$ 为 $M$ 个音素的文本序列，$l = (l_1, l_2, \cdots, l_M)$ 为每个音素的持续时间，$l_j$ 表示 $y_j$ 对应多少音频帧，且 $\sum_{j=1}^M l_j = N$。定义 $z = \text{rep}(y, l) = (z_1, z_2, \cdots, z_N)$ 为每帧对应的音素，即将 $y_j$ 重复 $l_j$ 次（$z_i$ 其实就是音频帧 $x_i$ 的音素标签）。对于一对 $(x, y)$，$l$ 和 $z$ 可以使用语音识别模型估计。$q(x_{\text{mis}} | y, x_{\text{ctx}})$ 的估计分解为音频模型 $q(x_{\text{mis}} | z, x_{\text{ctx}})$ 和时长模型 $q(l_{\text{mis}} | y, l_{\text{ctx}})$，其中 $l_{\text{mis}}$ 和 $l_{\text{ctx}}$ 表示 $l$ 被 $m'$ 和 $1 - m'$ 掩码，$m'$ 是从 $m$ 中基于 $l$ 下采样的，其中$m = \text{rep}(m', l)$。

#### 音频模型

 如图：
 ![](image/Pasted%20image%2020240307103520.png)

给定上下文 $z$ 和长度为 $N$ 的 $x_{\text{ctx}}$，可以假设 $x_{\text{mis}}$ 的分布是随机的。于是可以采用 CNF 来参数化，采用 flow matching 的目标函数进行训练。

将音频 $x$ 在 100Hz 帧率下提取为 80 维对数 Mel 谱（$x_i \in \mathbb{R}^{80}$）。当 $m_i = 1$，$x_{\text{ctx}} = 0$，当 $m_i = 0$，$x_{\text{ctx}} = x_i$ 。

建模的是所有帧 $x$ 的条件分布 $q(x | z, x_{\text{ctx}})$，而非仅掩码帧 $x_{\text{mis}}$。使用神经网络参数化条件向量场 $v_t(x_t, x_{\text{ctx}}, z; \theta)$，输入为 $x_{\text{ctx}}$ 和 $z$。

给定输入 $x_{\text{ctx}} \in \mathbb{R}^{N \times F}$，$x_t \in \mathbb{R}^{N \times F}$，音素序列 $z \in [K]^N$，$K$ 表示音素类别数，时间步 $t \in [0, 1]$，用 Transformer 模型参数化向量场 $v_t$。使用 look table $L \in \mathbb{R}^{K \times H}$ embed 音素序列 $z$，得到 embedding $z_{\text{emb}} \in \mathbb{R}^{N \times H}$，其中 $z_i = L(z_i)$。然后，三个序列（$x_t$、$x_{\text{ctx}}$ 和 $z_{\text{emb}}$）逐帧连接，并通过矩阵 $W_p \in \mathbb{R}^{(2F+H) \times D}$ 投影，得到序列 $H_c \in \mathbb{R}^{N \times D}$，其中 $D$ 表示 Transformer 模型的嵌入维度。

为了 embed flow step，用正弦位置编码将 $t \in [0, 1]$ 映射到 $h_t \in \mathbb{R}^D$。将 $H_c$ 与向量 $h_t$ 沿时间维度拼接得到输入序列 $\tilde{H}_c \in \mathbb{R}^{(N+1) \times D}$。Transformer 的输出为 $v_t(x_t, x_{\text{mis}}, z; \theta) \in \mathbb{R}^{N \times F}$，计算损失为：
$$\mathcal{L}_\text{audio-CFM}{ ( \theta ) }=\mathbb{E}_{t,m,q(x,z),p_0(x_0)}||u_t(x_t\mid x)-v_t(x_t,x_{ctx},z;\theta)||^2,$$
训练时，给定音频样本 $x$ 和先验样本 $x_0$，有 $x_t = (1 - (1 - \sigma_{\min})t)x_0 + tx$ 和 $u_t(x_t | x) = x - (1 - \sigma_{\min})x_0$。该函数计算所有帧的损失（包括未掩码的帧）。

mask 版本的 $\mathcal{L}_\text{audio-CFM}$ 为：
$$\mathcal{L}_{\text{audio-CFM-m}}(\theta)=\mathbb{E}_{t,m,q(x,z),p_0(x_0)}||m\odot(u_t(x_t\mid x)-v_t(x_t,x_{ctx},z;\theta))||^2,$$
其中 loss 仅在 mask 帧上计算。

#### 时长模型

时长模型有两种解决方案。一种和音频模型一样，通过条件向量场建模 $q(l | y, l_{\text{ctx}})$，将 $(x, x_{\text{ctx}}, z)$ 简单替换为 $(l, l_{\text{ctx}}, y)$，其中 $l, l_{\text{ctx}} \in \mathbb{R}^{M \times 1}$，$y \in [K]^M$。训练时使用 mask 版本的 CFM 损失。

第二种是，给定上下文时长 $l_{\text{ctx}}$ 和音素转录 $y$，回归预测掩码时长 $l_{\text{mis}}$。使用相同的 Transformer 模型，只有两个输入序列，不使用时间嵌入。模型使用掩码音素的 L1 回归损失训练：
$$\mathcal{L}_{\text{dur-regr-m}}(\theta)=\mathbb{E}_{m,q(l,y)}||m^{\prime}\odot(l_{mis}-g(l_{ctx},y;\theta))||_1,$$

其中 $g$ 表示基于回归的时长模型。
> 类似于 [FastSpeech 2- Fast and High-Quality End-to-End Text to Speech 笔记](FastSpeech%202-%20Fast%20and%20High-Quality%20End-to-End%20Text%20to%20Speech%20笔记.md) 中使用的 duration model，但额外使用了时长上下文 $l_{\text{ctx}}$ 作为输入。

### 推理

采样时，先从 $p_0$ 中采样噪声 $x_0$，然后使用 ODE solver 计算 $\phi_1(x_0)$，给定 $\frac{d\phi_t(x_0)}{dt} = v_t(\phi_t(x_0), x_{\text{ctx}}, z; \theta)$ 和初始条件 $\phi_0(x_0) = x_0$。

如图：
![](image/Pasted%20image%2020240307103542.png)

ODE solver 通过在多个 $t$ 处评估 $v_t$ 来计算 $\phi_1(x_0)$ 从 $t=0$ 到 $t=1$ 的积分（给定初始条件 $\phi_0(x_0) = x_0$）。函数评估次数（NFE）定义为 $d\phi_t(x_0)/dt$ 的计算次数。NFE 越高，解 $\phi_1(x_0)$ 越准确，但时间越长。
> 实验发现，Voicebox 可以在不到 10 次 NFE 的情况下生成高质量的语音，比自回归模型快得多。

### Classifier-Free Guidance

Classifier guidance（CG）是 diffusion 模型在 mode coverage 和 sample fidelity 之间进行 trade-off 的一种方法。主要通过修改对数似然梯度来实现。CG 近似于从 $p(x | c)p(c | x)^\alpha$ 中采样，其中 $c$ 是条件。此过程可以通过混合条件模型和无条件模型的得分估计来模拟。而无条件模型可以通过以一定概率丢弃 $c$ 联合训练。

将 CFG 扩展到 flow-matching 模型。音频模型的条件 $c$ 就是 $(z, x_{\text{ctx}})$，对于时长模型是 $(y, l_{\text{ctx}})$，在训练时以概率 $p_{\text{uncond}}$ 丢弃。推理时，修改后的音频模型向量场 $v_t$ 变为：
$$\tilde{v}_t(w,x_{mis},z;\theta)=(1+\alpha)\cdot v_t(w,x_{ctx},z;\theta)-\alpha\cdot v_t(w;\theta),$$
其中 $\alpha$ 是 guidance 强度，$v_t(w; \theta)$ 是丢掉  $x_{\text{ctx}}$ 和 $z$ 后得到的。用 $\alpha$ 和 $\alpha_{\text{dur}}$ 分别作为音频和时长模型的 CFG 强度。

### 应用

![](image/Pasted%20image%2020240307160828.png)


#### Zero-shot TTS & alignment-preserved style transfer

给定目标文本 $\hat{y}$ 和参考音频 $(x, y)$，zero-shot TTS 合成类似参考音频风格的语音。将参考音频和目标语音视为一个整体的 utterance，但是目标语音被 mask。设 $l$ 和 $z$ 为 $(x, y)$ 的音素持续时间和 frame-level phoneme。目标持续时间 $\hat{l}$ 在给定持续时间上下文 $l$ 和拼接的音素序列 $\text{cat}(y, \hat{y})$ 下进行采样得到。目标语音 ${\hat{x}}$ 在给定上下文 $x$ 和拼接的帧级音素 $\text{cat}(z, \text{rep}(\hat{y}, \hat{l}))$ 下进行采样得到。

#### Transient noise removal & content editing

Voicebox 可以通过重新生成受噪音污染的段落来进行噪音去除，给定原始帧级转录和周围的干净音频。具体来说，给定带噪音的音频 $(x, y)$ 的帧级 phoneme $z$，创建一个 mask $m$ 来指示有噪音的段。然后在给定 $z$ 和 $x_{\text{ctx}} = (1 - m) \odot x$ 下采样 $x_{\text{mis}}$。音频模型可能会生成干净的语音，因为在训练时，干净音频上下文大部分时间与干净目标音频同时出现。新的音频 $\hat{x} = x_{\text{mis}} + x_{\text{ctx}}$。

对于内容编辑，设 $\hat{y}$ 为新的文本，其中一些单词替换了原始 $y$ 的单词，$l$ 为原始的持续时间。用户首先通过从 $l$ 复制未替换的音素的长度构造 $l_{\text{ctx}}$（与 $\hat{y}$ 长度相同），并将新音素的长度设置为 0。给定 $l_{\text{ctx}}$ 和 $\hat{y}$，采样新音素的持续时间 $\hat{l}_{\text{mis}}$，新持续时间 $\hat{l} = \hat{l}_{\text{mis}} + l_{\text{ctx}}$。新的帧级 phoneme 为 $\hat{z}= \text{rep}(\hat{y}, \hat{l})$。同样，音频上下文 $x_{\text{ctx}}$ 与 $\hat{z}$ 长度相同，通过将未替换的音素映射到 $x$ 中的对应帧，新音素的帧设置为 0。给定 $\hat{z}$ 和 $x_{\text{ctx}}$，采样新音素 $x_{\text{mis}}$。编辑后的语音为 $\hat{x} = x_{\text{mis}} + x_{\text{ctx}}$。

#### Diverse speech sampling & alignment-preserved style shuffling

Voicebox 可以通过填充整个 utterance 生成多样的语音样本。首先使用时长模型在给定音素phoneme  $\hat{y}$ 下采样 $\hat{l}$。然后使用音频模型在给定 $\hat{z} = \text{rep}(\hat{y}, \hat{l})$ 下采样 $\hat{x}$。

类似于风格转移，Voicebox 通过在给定目标音频片段 $\bar{x}$ 的帧级转录 $\bar{z}$ 下采样 $\hat{x}$，在保持对齐的同时对音频风格进行 shuffle。
