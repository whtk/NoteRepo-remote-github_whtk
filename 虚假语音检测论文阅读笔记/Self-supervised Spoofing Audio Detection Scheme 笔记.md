
1. 提出一种自监督欺骗音频检测方案（SSAD），使用八个卷积块来捕获音频信号的局部特征。时间卷积网络（TCN）用于捕获上下文特征并实现并行操作。
2. 三个 regression workers 和 一个 binary worker 用于欺诈检测
3. ASVspoof 2019数据集 实验表明，SSAD 优于现有技术


## Introduction

1. 基于 DNN 的语音欺诈检测方案：
	1. [[A Light Convolutional GRU-RNN Deep Feature Extractor for ASV Spoofing Detection 笔记]] 通过合并light CNN 和基于门控递归单元的RNN，提出了一种轻卷积门控递归神经网络（LC-GRNN），并将LC-GRN用作深度特征提取器。
	2. [[Anti-spoofing speaker verification system with multi-feature integration and multi-task learning 笔记]] 提出了一种基于多特征集成和多任务学习（MFMT）的欺骗音频检测架构
2. 自监督学习已广泛应用于 NLP 和 CV，如 BERT 在许多 NLP 任务上提供最先进的结果。
3. MoCo 、SimCLR 等模型在自监督视觉表征中取得了有竞争力的结果，[[Learning Problem-agnostic Speech Representations from Multiple Self-supervised Tasks 笔记]] 提出了一种称为PASE的多任务自监督方法，用于学习语音信号处理中的问题无关的高级语音表示。
4. 本文提出 SSAD，包含一个 encoder 和两种 workers，
	1. encoder 将原始音频encoder 成 representations
	2. regression worker 用于从输入波形预测 target feature
	3. 完成训练后，workers 最小化网络预测和 target feature 的 MSE 误差
	4. Binary task worker 判断正负样本
5. SSAD 包含了一个二分类任务 congener info max，CIM旨在最小化两种相似音频之间的距离，并最大化两种不同音频之间的间距。
6. SSAD 是多任务自监督学习在欺骗音频检测中的首次应用且效果很好。


## 使用 SSAD 进行自监督学习

SSAD 结构如图：![[Pasted image 20221115153745.png]]包括：
+ encoder
+ 三个 regression workers
+ 一个 binary worker

### encoder

首先是 八个卷积块，包括：
+ 1-d 卷积
+ Batch Norm
+ 多参数rele（ PReLU）

中间卷积层还引入了 skip connect，用来 transfer 不同 levels 的 abstractions 到最终的 representations，同时修改了 PASE+ 中的结构：
1.   时间卷积网络（TCN）：将TCN放置在卷积层的顶部，SSAD可以更有效地学习长期依赖关系，对一维输入序列 $x \in \mathbb{R}^n$ 和 滤波器 $f:\{0, \cdots, k-1\} \rightarrow \mathbb{R}$ ，元素 $s$ 上的空洞卷积定义为：$$F(s)=\left(x \cdot{ }_d f\right)(s)=\sum_{i=0}^{k-1} f(i) \cdot x_{s-d \cdot i}$$其中，$d$ 为 dilation factor，$k$ 为 filter size。Dilation 相当于在相邻两个 filter 之间引入固定的步长。TCN 可以并行计算，灵活性也更大。
2. 非线性 projection：在对比学习任务中，在编码器的输出层中，非线性投影比线性投影表现得更好。

### workers

workers 的输入为 encoded representation。

workers 把 regression 任务也视作 自监督任务，然后传回平均 error 来帮助 encoder 更好地发掘 high-level 的 representation。

workers 由一个带PReLU 激活的隐藏层组成。之所以这么简单是为了 激励encoder，就说是，即使 workers 这么简单，encoder 也有能力发掘出能够被 decode 的 high-level representation。

1. regression task：目标是预测 target feature，包含以下三种 target feature：
	1. 对数功率谱
	2. LFCC
	3. CQCC
2. binary task：encoder 和 binary task 合作以获得更好的 representation。通过定义采样策略，从训练集中可用的 SSAD representation 中提取 chor $s_a$，positive $s_r$，negative $s_f$ ，然后引入 CIM，用于最小化 similar 语音之间的距离最大化 different 语音之间的距离，采用交叉熵来定义距离，有：$$\begin{gathered}
L 1=E_{S_r}\left[\log \left(d\left(s_a, s_r\right)\right)\right] \\
L 2=E_{S_f}\left[\log \left(1-d\left(s_a, s_f\right)\right)\right] \\
L=L 1+L 2
\end{gathered}$$其中，$d$ 为判别函数，$E_{S_r},E_{S_f}$ 为正负样本的期望，通过最小化 $L$ ，binary worker 可以实现区分真实和虚假语音。

### 训练

初始 lr 为 $0.5 \times 10^{-3}$，采用多项式调度器。Adam 优化器，batch size 为 16，每段长度为 2s，在 Tesla V100 训练 100 epoch 共五天。

## 数据集、结果

数据集：ASVspoof 2019 LA

模型采用了 LCNN-big, LCNN-small, and SENet12 三种分类器，训练时，LCNN 使用 A-softmax 损失，SENet12 使用 softmax + 交叉熵损失。

1. 和 PASE+ 进行比较：![[Pasted image 20221115162003.png]]后端分类器为 SENet12 时，TCN 的泛化性更好；后端使用 LCNN-big 时，方法最优。
2. 和常见的语音特征进行比较：![[Pasted image 20221115162251.png]]说明，使用不同的分类器时，SSAD 的特征泛化性更强。