
1. 提出了端到端的 AVSD 方法，将 音频特征、多说话人的 lip ROI 和 多说话人的 i-vector 作为多模态输入，输出为一系列的二分类结果表明某个说话人的语音活动
2. 端到端结构可以处理重叠语音，同时利用多模态信息准确区分语音段和非语音段；同时 i-vector 用于解决视觉模态误差引起的对准问题
3. 模型在没有视觉模态下是鲁棒的，但是仅使用视觉模态性能大幅下降


## Introduction

1. 单模态的 SD 非常有挑战性，基于音频模态时，混合语音包含多种噪声和混响；基于视频模态时，说话人可能移动或者被遮挡（嘴唇或面部信息并不总是可靠的）
2. 语音重叠是 SD 的一个难点，[[End-to-End Neural Speaker Diarization with Permutation-Free Objectives 笔记]] 和 [[Target-Speaker Voice Activity Detection- a Novel Approach for Multi-Speaker Diarization in a Dinner Party Scenario 笔记]] 通过直接预测每帧的说话人的活动性来处理重叠，但是仍难以对低质量高重叠的语音进行分类
3. 仅用视觉方法时，有多说话人的面部或嘴唇跟踪、基于视觉的 VAD。多说话人的视觉跟踪也是一项有挑战的任务。
4. 面部信息和嘴唇运动与语音高度相关，通过采用互信息 mutual information、canonical correlation analysis 和 深度学习等来进行协同；通过线性组合、时间对齐、贝叶斯等方法来融合不同的模态
5. 本文研究嘴唇运动（高清摄像头录制的 lip ROI）和语音（单通道音频）对 SD 的影响。和基于 deep audio-visual synchronisation network 、clustering on audio-visual pairs scored by audio-visual relation network 等网络不同，本文提出的端到端方法直接预测语音活动概率
	1. 一方面将提取的帧级音频 embedding 与同步的帧级多说话人嘴唇 embedding 进行拼接
	2. 另一方面将所有说话人的 i-vector 作为附加输入以解决由于遮挡、离开或不可靠检测导致的输入嘴唇缺失时的对齐问题

## AVSD 数据集

1. AVDIAR
2. VoxConverse
3. AMI
4. AVA-AVD
5. MISP

## 提出的方法

网络结构：![](./image/Pasted%20image%2020221115220935.png)



### 概率模型

令 $N$ 为最大的说话人数量，在时间 $t$ 有 $N$ 个 lip ROI $\mathbf{X}_t=\left(\boldsymbol{X}_{t, 1}, \ldots, \boldsymbol{X}_{t, n}, \ldots, \boldsymbol{X}_{t, \hat{N}}, \ldots, \boldsymbol{X}_{t, N}\right) \in \mathbb{R}^{W \times H \times N}$ ，$\hat{N}$ 为当前 session 的说话人数量，随机变量 $\boldsymbol{X}_{t, n} \in \mathbb{R}^{W \times H}$ 是 person $n$ 在时间 $t$ 的 lip figure（$W\times H$ 表示图像长宽）。随机从 silent lips 中选择一个 fake lip 来补全 $n>\hat{N}$ 和 lip 丢失的情况。使用 $\mathbf{Y}_t=\left(Y_{t, 1}, \ldots, Y_{t, f}, \ldots, Y_{t, F}\right) \in \mathbb{R}^F$ 表示 $F$ 维的单通道音频的 FBANKs 特征。

时间序列 $\mathbf{X}_{1: T}=\left\{\mathbf{X}_1, \ldots, \mathbf{X}_t, \ldots, \mathbf{X}_T\right\}$ 和 $\mathbf{Y}_{1: T}=\left\{\mathbf{Y}_1, \ldots, \mathbf{Y}_t, \ldots, \mathbf{Y}_T\right\}$ 分别表示视频和音频特征序列。

SD 的任务是对于每个语音信号都分配一个说话人，引入 $\mathbf{S}_{1: T}=\left\{\boldsymbol{S}_1, \ldots, \boldsymbol{S}_t, \ldots, \boldsymbol{S}_T\right\} \in\{0,1\}^{N \times T}$ ，其中向量 $\boldsymbol{S}_t=\left(S_{t, 1}, \ldots, S_{t, n}, \ldots, S_{t, N}\right) \in\{0,1\}^N$ 中的元素非 $0$ 即 $1$，也就是 $S_{t, n}=0$ 表示说话人 $n$ 在时间 $t$ 是 silent（$n\leq\hat{N}$）, 对于 fake person（$n>\hat{N}$），$S_{t, n}$ 全为 $0$。

引入 i-vector $\mathbf{I}=\left\{\boldsymbol{I}_1, \ldots, \boldsymbol{I}_n, \ldots, \boldsymbol{I}_{\hat{N}}, \ldots, \boldsymbol{I}_N\right\} \in \mathbb{R}^{D_I \times N}$ 来避免对齐问题（即由于lip确实导致模型不知道谁在当前时刻说了话），其中 $\boldsymbol{I}_n \in \mathbb{R}^{D_I}$ 为第 $n$ 个人的 i-v
ector。当 $n>\hat{N}$ 时，随机选择不属于此 session 的 fake speaker 的 i-vector。

SD 问题为，给定以上条件来找到最可能的 $\hat{\mathbf{S}}$  ：$$\hat{\mathbf{S}}=\underset{\mathbf{S} \in \mathcal{S}}{\arg \max } P(\mathbf{S} \mid \mathbf{X}, \mathbf{Y}, \mathbf{I})$$其中，$P(\mathbf{S} \mid \mathbf{X}, \mathbf{Y}, \mathbf{I})$ 基于条件概率可以写为：$$P(\mathbf{S} \mid \mathbf{X}, \mathbf{Y}, \mathbf{I})=P\left(\mathbf{S} \mid \mathbf{E}_{\boldsymbol{V}}, \mathbf{E}_{\boldsymbol{A}}, \mathbf{I}\right) P\left(\mathbf{E}_{\boldsymbol{V}} \mid \mathbf{X}\right) P\left(\mathbf{E}_{\boldsymbol{A}} \mid \mathbf{Y}\right)$$其中，$\mathbf{E}_{\boldsymbol{V}}=\left\{\boldsymbol{E}_{V_1}, \ldots, \boldsymbol{E}_{V_n}, \ldots, \boldsymbol{E}_{V_N}\right\} \in \mathbb{R}^{T \times D_V \times N}$ 为 $D_{V}$ 维 的帧级视觉隐变量 V-embedding；$\mathbf{E}_{\boldsymbol{A}} \in \mathbb{R}^{T \times D_A}$ 同理为 $D_{A}$ 维的音频隐变量 A-embedding，这里假设两者条件独立。

### V-embedding

建立 视觉网络来计算 V-embedding。网络通过在原始的 lipreading TCN 网络中直接添加几层 conformer blocks 和 一个 BLSTM 层来实现。

然后把 V-embedding 通过全连接层投影到语音活动概率向量 $\hat{\boldsymbol{S}}_n^V=\left(\hat{S}_{1, n}^V, \ldots, \hat{S}_{t, n}^V, \ldots, \hat{S}_{T, n}^V\right) \in(0,1)^T$ 中，整个网络可以被视为视觉 VAD，得到的概率也可以被单独用于 VAD。

视觉网络需要预训练。

视觉网络如上图中的第一行。

### A-embedding

音频网络首先对输入音频采用 NARA-WPE 工具进行去混响，然后提取 FBANKs 作为 2D 卷积、Batch Norm 和 ReLU 的输入。然后一个全连接投影层把高维的 CNN 输出投影到低维的 A-embedding。

音频网络不需要预训练。

音频网络如上图中的最后一行。

### 说话人 embedding

没有说话人 embedding 的时候存在两种问题：
+ 有人在说话人，但是 lip 信息丢失，这种情况下，要么两个网络都输出没有语音，要么随机一个网络输出有语音；
+ 有人在说话人且其他人没有说话人，但是两个人的 lip 都在运动 ，这种情况下，要么两个网络都输出有语音，要么随机一个网络输出有语音；

纯音频的 SD 中，将 i-vector 作为输入可以估计多说话人的 VAD，因此本文也引入了 i-vector 来处理 VASD 中的多说话人。

在训练阶段使用 oracle 标签中每个说话人的非重叠片段来计算 i-vector；在推理阶段；仅使用视觉 SD 的结果来估计 i-vector。

### embedding 组合

首先，$\mathrm{V} \text {-embedding } \mathbf{E}_{\boldsymbol{V}}$ 重复 $K$ 次进行拼接（因为音视频中的帧移不同），$\text { A-embedding } \mathbf{E}_A$ 重复 $N$ 次，$\text { i-vectors I }$ 重复 $T$ 次，然后，音视频说话人 embedding （AVSE）通过拼接三者为：$$\mathbf{E}_{\mathbf{S}}=\left\{\boldsymbol{E}_{S_1}, \ldots, \boldsymbol{E}_{S_n}, \ldots, \boldsymbol{E}_{S_N}\right\} \in \mathbb{R}^{T \times D_S \times N}$$其中，$\boldsymbol{E}_{S_{n}} \in \mathbb{R}^{T \times D_S}$    为帧级 的 AVSE，且 $D_S=D_V+D_A+D_I$。

然后组合二两层 BLSTMP 进一步提取音视频特征。

然后采取一层 BLSTMP + FC 层来获得最终的 $N$ 个说话人的语音活动概率 $\hat{\boldsymbol{S}}^{A V}=\left(\hat{\mathbf{S}}_1^{A V}, \ldots, \hat{\mathbf{S}}_n^{A V}, \ldots, \hat{\mathbf{S}}_N^{A V}\right) \in (0,1)^{T \times N}$ 。

### 优化

AVSD 网络通过以下三个步骤进行优化：
1. 复制预训练的 lipreading 模型，然后用 lr=1e-4 训练 V-VAD 模型，损失函数定义为 $$J_{V_n}=\frac{1}{T} \sum_{t=1}^T B C E\left(S_{t, n}, \hat{S}_{t, n}^V\right)$$其中，$B C E(\cdot, \cdot)$ 为交叉熵函数。
2. freeze 视觉网络的参数，用 lr=1e-4 同步训练 音频网络和 audio-visual decoding block，损失函数定义为：$$J_{A V}=\frac{1}{N T} \sum_{n=1}^N \sum_{t=1}^T B C E\left(S_{t, n}, \hat{S}_{t, n}^{A V}\right)$$
3. 最后，unfreeze 视觉网络参数，同步、联合训练整个网络，lr=1e-5，损失函数为：$$J_{J o i n t}=\lambda \cdot \frac{1}{N} \sum_{n=1}^N J_{V_n}+J_{A V}$$实验中，$\lambda=0.1$。

上述实验都在 middle-field audio and video 数据集中进行训练。

## 实验

### 实验设置

40 维的 FABNKs（$F=40$），帧长 25ms，帧移 10ms。

视频每秒 25 帧，也就是 40ms一帧，lip ROI 图像大小为 $96 \times 96(W \times H)$。

V-embedding 维度为 $D_V=256$，每帧重复 $4K$ 次。
> $4=\frac{40ms}{10ms}$ 

$D_I=100$ 的 i-vector 在 CN-CELEB 数据集上训练。$D_A=256$。

视觉网络中，3 个 conformer block，encoder 的维度为 256，attention head 为 4，kernel size 为 32，BLSTM 有 256 个 cell。

音频网络中，4 层 2D CNN，audio-visual decoding block 中，所有的  BLSTMP 有 896 个 cell。

还有一个阈值用来决定每个帧的语音活动。在 DEV 集中找最佳的然后应用到 EVAL 集。

采用了 post-processing 方法。

采用 DER 作为评估指标。

### 实验结果

结果如图：![](./image/Pasted%20image%2020221116113837.png)

仅音频系统在MISP上表现不佳，主要是因为对话和背景电视谈话者的重叠率较高。
纯视觉系统的 MISS和FA 较高。
通过 i-vector 和联合训练，视听系统可以实现更好的性能。
视听系统的 DOVER Lap 可以带来进一步的改进。
没有 pre-processing 时，AVSD受 无参考VAD 的影响较小。

Lip ROI 确实对系统的影响：![](./image/Pasted%20image%2020221116114600.png)当视觉模态缺失时，所提出的AVSD方法比VSD方法鲁棒得多。同时，i-vector 在AVSD模型中起着重要作用。

一个很好的示例：![](./image/Pasted%20image%2020221116114837.png)