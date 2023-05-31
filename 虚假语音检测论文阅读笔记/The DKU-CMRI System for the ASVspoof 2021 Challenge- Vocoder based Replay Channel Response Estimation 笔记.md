
1. 认为 声码器可以部分消除 PA 欺诈语音中的 重放信道信息
2. 在 PA 任务中，使用 原始音频和声码器过滤后的音频之间的差异作为特征，采用离群点检测模型作为后端分类器
3. 在 LA 任务中，通过数据增强 以及 RawNet2 和  LFCC-LCNN 模型来进行欺诈检测

## Introduction

1. 欺诈检测方法的泛化性能是一个很大的挑战
2. 离群点检测（类似于 One-Class Classification ）具有很好的潜力，可用于增强模型对未知攻击的泛化性能，因为在训练阶段仅使用一类数据，所有数据都带有正标签
3. 离群点检测 已经被用于 LA 任务，用于学习 嵌入空间中 真实音频的表征点的边界，如 [[One-class Learning Towards Synthetic Voice Spoofing Detection 笔记]]
4. 本文仅基于ASVspoo2019真实数据为PA任务构建异常值检测系统，提出基于声码器的重放信道响应估计方法，采用声码器来突出重放信道信息，使用 原始音频和声码器过滤后的音频之间的差异作为特征，采用 GMM 和 VAE 作为后端分类器。
5. 对于LA任务，使用 LFCC-LCNN和RawNet2 baseline。同时两个任务都使用了 数据增强

## PA 任务
由于 ASVspoo2019 训练数据和2021评估数据之间存在很大的域不匹配，因此不采用深度学习建模方法。而是提出了基于声码器的重放信道响应估计方法。

相比于基于 DNN 的二分类模型能够实现更好的泛化性能来对抗未知攻击。模型如图：
![[Pasted image 20221209215754.png]]
设计了 utterance 和 frame level 的模型。

### 基于 声码器的 重放信道响应估计

#### frame level 建模

录制音频可以表示为：$$x(n)=s(n) * h(n)+v(n)$$
$x(n)$ 为录制信号，$s(n)$ 为真实信号，$h(n)$ 为 重放信道脉冲响应，$v(n)$ 为加性噪声。假设噪声为 0，在信号经过 STFT 之后，频域表示为：$$X(k, l)=S(k, l) H(k)$$
其中，$k$ 为 frequency bin，$l$ 为 time frame，对数幅度谱表示为：$$\log (|X(k, l)|)=\log (|S(k, l)|)+\log (|H(k)|)$$
但是很难估计 $S(k, l)$ 。之前一般使用 GMM 来建模，本文使用 vocoder 来进行滤波以减轻 $S(k, l)$ 的影响。

假设 vocoder 是一个信道无关的模型因此可以用于消除重放信道的信息（响应）。因此经过 vocoder 的信号 $X_{\text {vocoder }}(k, l)$ 满足：$$X_{\text {vocoder }}(k, l)=S(k, l) H_{\text {vocoder }}(k) H_{\text {res }}(k)$$
其中，$H_{\text {vocoder }}$ 表示 vocoder channel，$H_{\text {res}}$ 表示剩下的 重放信道响应（因为只能消除一部分。。。），计算对数幅度谱之后有：$$\begin{aligned}
\log \left(\left|X_{\text {vocoder }}(k, l)\right|\right)= & \log (|S(k, l)|)+\log \left(\left|H_{\text {vocoder }}(k)\right|\right) \\
& +\log \left(\left|H_{\text {res }}(k)\right|\right)
\end{aligned}$$
定义提出的 基于 vocoder 的 重放信道估计特征为 $H_{v r}$，计算为：$$\begin{aligned}
\log \left(H_{v r(k, l)}\right) & =\log \left(\left|X_{v o c o d e r}(k, l)\right|\right)-\log (|X(k, l)|) \\
& =\log \left(\left|H_{v o c o d e r}(k)\right|\right)+\log \left(\left|H_{r e s}(k)\right|\right)-\log (|H(k)|)
\end{aligned}$$
其中，$H_{v o c o d e r}(k)$ 可以被视为常数，对于真实语音，$h(n)$ 为单位脉冲响应，所以另外两个 log 项都是0，整体就是常数；对于重放信号，$H_{v r}(k, l)$ 时刻在变。

通过这种变换和计算，能够突出重放信号的 重放信道信息。

#### utterance level 建模

就是对 frame level 的值在时间维度进行求平均：$$\log \left(H_{v r}\right)=\frac{1}{L} \sum_{l=1}^L \log \left(\left|X_{\text {vocoder }}(k, l)\right|\right)-\frac{1}{L} \sum_{l=1}^L \log (|X(k, l)|)$$
### 边界检测
> 采用 GMM 和 VAE 方法来训练 one-class outlier detection 模型

#### GMM 模型

在 one class 模型中，低于 GMM 概率的某个阈值被认为是 异常点。

在训练 GMM 模型时，只使用了 真实语音的数据。

在 eval 阶段，GMM 的输出的概率就是score。

#### VAE 模型

使用重构概率值作为 score。

VAE 的好处就是 score 是基于分布采样得到的，而不是基于输入本身，因此适合更 generized 的场景。

## LA 任务

采用给的 baseline 来实现，即 RawNet2 和 LFCC-LCNN。

唯一的改动是，训练 LCNN 的时候采用 angular-margin softmax 损失来增加类间距离，减少类内距离。

## 实验

训练数据：ASV spoof 2019 LA、PA（仅用真实语音训练）

指标：min t-DCF、EER

数据增强：speed perturbation 对于重放检测任务有效，使用 Pyroomacoustic 工具包来添加混响来模拟远场真实语音

探索了 WORLD vocoder、MelGAN vocoder、HiFi-GAN vocoder 的效果。

模型：PA 中，512 维特征，对于 utterance level 的特征使用 PCA 进行降维；LA 任务就用的 baseline，LCNN 用了 AM-softmax 损失

## 结果

不同 vocoder 、不同后端的PA检测结果：![[Pasted image 20221210090153.png]]

结论：
1. 最好的 vocoder 是 WORLD
2. utterance level 效果优于 frame level，PCA 可以提高 frame level 的性能
3. 数据增强可以提升性能，但是在后评估阶段反而可能使效果变差

LA 结果：![[Pasted image 20221210093245.png]]
没用，就是 baseline。






