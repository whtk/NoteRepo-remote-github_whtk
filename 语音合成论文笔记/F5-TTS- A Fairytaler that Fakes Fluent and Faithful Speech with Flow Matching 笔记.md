> preprint 2024，上交、剑桥

1. 提出 F5-TTS，基于 flow matching 和 DiT 的非自回归 TTS，文本通过 filler token 来 pad 到和输入语音一样的长度，同时提出：
    1. 使用 ConvNeXt 对输入进行建模，使文本表征更容易与语音对齐
    2. 提出 inference-time Sway Sampling 策略，提高模型性能和效率
    3. 可以实现快速训练，推理 RTF 为 0.15，比现有基于 diffusion 的 TTS 模型有很大提升
2. [代码和模型开源](https://SWivid.github.io/F5-TTS)

## Introduction

1. [E2 TTS- Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS 笔记](E2%20TTS-%20Embarrassingly%20Easy%20Fully%20Non-Autoregressive%20Zero-Shot%20TTS%20笔记.md) 使用字符填充的方式，不需要音素和 duration predictor，直接将字符填充到 mel 谱长度，得到自然的合成结果

2. 提出 F5-TTS，使用 DiT 和 ConvNeXt V2，不需要 phoneme alignment、duration predictor、text encoder 和 codec，但是可以更好地处理文本-语音对齐；提出 inference-time sampling 策略，提高自然度、可理解度和说话人相似度，且可以集成到现有的 flow matching 模型中

## 预备知识

### Flow Matching

FM 目标为将简单分布 $p_0$ （标准正态分布 $N(x|0,I)$）的概率路径 $p_t$ 匹配到近似数据分布 $q$ 的 $p_1$，FM 通过神经网络 $v_t$ 拟合向量场 $u_t$，损失为：
$$\mathcal{L}_{FM}(\theta)=\,E_{t,p_{t}(x)}\vert\vert v_{t}(x)-u_{t}(x)\vert\vert^{2}$$

其中 $\theta$ 为神经网络，$t\sim U[0,1]$，$x\sim p_t(x)$，模型 $v_t$ 在整个 flow step 和数据范围上训练。

在实际训练中，考虑条件概率路径 $p_t(x|x_1)=N(x|\mu_t(x_1),\sigma_t(x_1)^2I)$，Conditional Flow Matching (CFM) 损失对 $\theta$ 的梯度相同，其中 $x_1$ 是对应训练数据的随机变量，$\mu$ 和 $\sigma$ 是时间相关的高斯分布均值和标准差。

目标是从初始简单分布（如高斯噪声）构建目标分布（数据样本），flow map $\psi_t(x)=\sigma_t(x_1)x+\mu_t(x_1)$，其中 $\mu_0(x_1)=0$，$\sigma_0(x_1)=1$，$\mu_1(x_1)=x_1$，$\sigma_1(x_1)=0$，使所有条件概率路径在开始和结束时收敛到 $p_0$ 和 $p_1$，flow 向量场为 $d\psi_t(x_0)/dt=u_t(\psi_t(x_0)|x_1)$，用 $x_0$ 重新参数化 $p_t(x|x_1)$，得到：
$$\mathcal{L}_{\mathrm{CFM}}(\theta)=E_{t,q(x_{1}),p(x_{0})}||v_{t}(\psi_{t}(x_{0}))-\frac{d}{d t}\psi_{t}(x_{0})||^{2}.$$

进一步利用最优传输形式 $\psi_t(x)=(1-t)x+tx_1$，得到 OT-CFM 损失：
$${\mathcal{L}}_{\mathrm{CFM}}(\theta)=E_{t,q(x_{1}),p(x_{0})}\|v_{t}((1-t)x_{0}+t x_{1})-(x_{1}-x_{0})\|^{2}.$$

更一般地，用 log 信噪比（log-SNR） $\lambda$ 替代 flow step $t$，预测 $x_0$（在扩散模型中常用的 $\epsilon$）替代预测 $x_1-x_0$，CFM 损失等价于具有 cosine schedule 的 v-prediction 损失。

推理时，给定从初始分布 $p_0$ 中采样的噪声 $x_0$、flow step $t\in[0,1]$和条件，使用 ODE solver 求解 $\psi_1(x_0)$，即 $d\psi_t(x_0)/dt$ 的积分，其中 $\psi_0(x_0)=x_0$。函数评估次数（NFE）表示神经网络计算次数，可以通过多个 flow step 来近似积分。更高的 NFE 会产生更准确的结果，但需要更多计算时间。

### Classifier-Free Guidance

CFG 用隐式分类器替换显式分类器，不直接计算显式分类器及其梯度。分类器的梯度可以表示为条件生成概率和无条件生成概率的组合。在训练时以一定比例丢弃条件，用有/无条件的推理分别得到结果，得到最终的结果：
$$v_{t,C F G}=v_{t}(\psi_{t}(x_{0}),c)+\alpha(v_{t}(\psi_{t}(x_{0}),c)-v_{t}(\psi_{t}(x_{0})))$$

$\alpha$ 是 CFG 强度。

## 方法


采用类似 [E2 TTS- Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS 笔记](E2%20TTS-%20Embarrassingly%20Easy%20Fully%20Non-Autoregressive%20Zero-Shot%20TTS%20笔记.md) 的 pipeline，提出 F5-TTS，解决 E2 TTS 的收敛速度慢和鲁棒性差的问题；提出 inference 时的 Sway Sampling 策略，提高模型在生成时对参考文本和说话人相似度的性能。

框架如图：
![](image/Pasted%20image%2020241022113642.png)

### Pipeline



#### 训练

infiling 任务是给定周围音频和完整文本（包括周围的和要生成的）来预测语音片段。用 $x$ 表示音频样本，$y$ 表示对应的文本。训练时，输入是从 $x$ 提取的 mel 谱 $x_1\in \mathbb{R}^{F\times N}$，其中 $F$ 是 mel 维度，$N$ 是序列长度。CFM 的输入是噪声语音 $(1-t)x_0+tx_1$ 和 mask 的语音 $(1-m)\odot x_1$，其中 $x_0$ 表示采样的高斯噪声，$t$ 是采样的 flow step，$m\in\{0,1\}^{F\times N}$ 表示 01 mask。

对于英文，直接使用字母和符号；对于中文，使用 full pinyin（全拼）得到字符序列，并用 filler token 填充到和 mel frames 一样的长度，得到扩展序列 $z$，其中 $c_i$ 表示第 $i$ 个字符：
$$z=(c_1,c_2,\ldots,c_M,\underbrace{\langle F\rangle,\ldots,\langle F\rangle}_{(N-M)\text{ times}}).$$

模型训练时，基于 $(1-m)\odot x_1$ 和 $z$ 重建 $m\odot x_1$，等效于以 $P(m\odot x_1|(1-m)\odot x_1,z)$ 形式学习目标分布 $p_1$，近似真实数据分布 $q$。

#### 推理

推理时，给定 prompt 音频的 mel 谱 $x_{\text{ref}}$、对应文本 $y_{\text{ref}}$ 和文本 prompt $y_{\text{gen}}$ 来生成带有所需内容的语音。其中 prompt 音频提供说话人特征，文本 prompt 包含生成的内容。

序列长度 $N$ 是需要告知模型生成样本的期望长度。可以训练一个单独的模型基于 $x_{\text{ref}}$、$y_{\text{ref}}$ 和 $y_{\text{gen}}$ 预测和提供持续时间。这里简单地基于 $y_{\text{gen}}$ 和 $y_{\text{ref}}$ 中字符数量的比例估计持续时间。

从学习的分布中采样时，转换后的 mel 特征 $x_{\text{ref}}$ 和拼接并扩展的字符序列 $z_{\text{ref}\cdot\text{gen}}$ 作为条件，得到：
$$v_t(\psi_t(x_0),c)=v_t((1-t)x_0+tx_1|x_{ref},z_{ref\cdot gen}),$$

从采样的噪声 $x_0$ 开始，目标是 $x_1$。使用 ODE solver 从 $\psi_0(x_0)=x_0$ 逐渐积分到 $\psi_1(x_0)=x_1$，给定 $d\psi_t(x_0)/dt=v_t(\psi_t(x_0),x_{\text{ref}},z_{\text{ref}\cdot\text{gen})}$。推理时，flow step，从 0 到 1 均匀采样。

得到 mel 后，丢弃 $x_{\text{ref}}$ 的部分，使用 vocoder 将 mel 转换为语音信号。

### F5-TTS

E2 TTS 直接将 pad 过的字符序列和输入语音拼接，导致 semantic 和 acoustic 特征耦合，从而导致训练困难。为了解决收敛速度慢和鲁棒性差的问题，提出 F5-TTS。引入 inference-time Sway Sampling，允许更快的推理（使用更少的 NFE），同时保持性能。

#### 模型

模型如上图，使用 zero-initialized adaptive Layer Norm（adaLN-zero）的 DiT 作为 backbone，使用 ConvNeXt V2 blocks 增强模型的对齐能力。

模型输入是字符序列、噪声语音和 mask 语音。在特征拼接前，字符序列先经过 ConvNeXt blocks。这里没有显式引入文本的边界，semantic 和 acoustic 特征与整个模型一起学习。

CFM 的 flow step $t$ 作为 adaLN-zero 的条件。
> 作者发现，对于 TTS 任务，adaLN 中的额外的 text sequence 的 mean pooled token 不是必要的，可能是因为 TTS 任务需要更严格的 guided，但 mean pooled text token 更粗糙。

采用 [Voicebox- Text-Guided Multilingual Universal Speech Generation at Scale 笔记](Voicebox-%20Text-Guided%20Multilingual%20Universal%20Speech%20Generation%20at%20Scale%20笔记.md) 中的一些位置编码，flow step 用正弦位置编码，输入序列和卷积位置编码相加，对于 self-attention 使用 RoPE，对于扩展后的字符序列，也加上绝对正弦位置编码。

不使用 U-Net 的 skip connection，而使用 DiT 和 adaLN-zero。没有 phoneme-level duration predictor 和显式的对齐。

#### 采样

CFM 可以看作具有 cosine schedule 的 v-prediction。图像合成中，有人提出使用单峰 logit-normal sampling，从而可以给中间的 flow step 更多权重。
> 作者猜测这种采样可以更均匀地分配模型的学习难度。


本文训练时使用均匀采样的 flow step $t\sim U[0,1]$，但在推理时使用非均匀采样。定义 Sway Sampling 函数：
$$f_{sway}(u;s)=u+s\cdot(\cos(\frac\pi2u)-1+u),$$

其中 $s\in[-1, \frac{2}{\pi-2}]$，首先采样 $u\sim U[0,1]$，然后根据这个函数得到 sway sampled flow step $t$。$s<0$ 时，采样向左；$s>0$ 时，采样向右；$s=0$ 时，等同于均匀采样。

CFM 模型在早期更关注从纯噪声中得到语音的 contours，后期更关注细节。因此，前几个 step 的生成结果决定了 speech 和 text 之间的对齐。$s<0$ 时，更多地使用小的 $t$，从而告诉 ODE 更多的 startup 信息，在初始的积分过程中实现更精确的 evaluation。

## 实验设置

数据集：
+ Emilia，过滤掉 transcription failure 和 misclassified language speech 后，得到 95K 小时的英文和中文数据；
+ WenetSpeech4TTS Premium subset，包含 945 小时的普通话语料。
+ 三个测试集：LibriSpeech-PC test-clean，Seed-TTS test-en，Seed-TTS test-zh

训练：
+ 基础模型训练 1.2M 次迭代，batch size 307,200，8 NVIDIA A100 80G GPU，AdamW 优化器，最大学习率 7.5e-5，线性 warm up 20K 次迭代，剩余部分线性衰减，梯度裁剪到 1
+ F5-TTS 基础模型有 22 层，16 attention heads，1024/2048 embedding/FFN dimension for DiT，4 层，512/1024 embedding/FFN dimension for ConvNeXt V2，共 335.8M 参数

+ 英文使用字母和符号，中文使用 jieba 和 pypinyin 处理原始中文字符为全拼，字符嵌入词汇表大小 2546，包括特殊的 filler token 和 Emilia 数据集中的所有其他语言字符
+ 音频样本使用 100 维 log mel-filterbank 特征，采样率 24 kHz，hop length 256，随机 70% 到 100% 的 mel frames 用于 infilling 任务训练
+ CFG 训练时，首先丢弃 0.3 的 mask 语音输入，然后再次丢弃 0.2 的 mask 语音和文本输入，假设 CFG 训练的两阶段控制可以让模型更多地学习文本对齐


推理：
+ 使用 EMA 权重，使用 Euler ODE solver 进行推理
+ 使用预训练的 vocoder Vocos 将生成的 log mel 谱转换为音频信号

baseline：
+ autoregressive 模型：VALL-E 2，MELLE，FireRedTTS，CosyVoice
+ non-autoregressive 模型：Voicebox，NaturalSpeech 3，DiTTo-TTS，MaskGCT，Seed-TTS(DiT)，E2 TTS


评价指标：
+ WER
+ speaker Similarity（SIM-o）
+ CMOS
+ SMOS

## 实验结果（略）
