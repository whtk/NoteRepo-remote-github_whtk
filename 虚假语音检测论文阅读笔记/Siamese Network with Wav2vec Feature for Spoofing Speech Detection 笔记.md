1. 提出了一种基于孪生网络的两阶段表征学习系统用于欺诈语音检测。
2. 在表征学习阶段，使用 wav2vec 特征训练嵌入孪生网络，以区分一对语音样本是否属于同一类。
3. 在ASVspoof 2019评估集上，提出的系统将EER从最新的4.07%降低到1.15%


## Introduction
1. 本文目标，研究针对语音转换和合成的欺骗攻击对策
2. 检测性能高度依赖于特征，[[CQCC]] 优于传统的 MFCC，LFCC、GD、对数谱 和 [[x-vector、i-vector]] 、MGD 等也取得了改进的性能；在后端分类器中，GMM 用作 baseline；
3. [[Improving Replay Detection System with Channel Consistency DenseNeXt for the ASVspoof 2019 Challenge 笔记]] 提出了一种模型信道一致性 DenseNeXt，在不损失性能的情况下减少了参数数量和计算能力；
4. [[ASSERT- Anti-Spoofing with Squeeze-Excitation and Residual neTworks 笔记]] 提出一种基于 [[SENet]] 和 ResNet 的模型，相比于之前的模型有很大的改进。
5. 语音处理中，无监督预训练模型wav2vec在语音识别方面优于最佳的基于特征的模型。研究表明，从大规模数据集上预训练的模型中提取的特征可以用于可能没有足够标记数据或不可能从每个可能的领域收集训练数据的任务。因此，本文使用wav2vec特征代替传统的声学特征来提高模型对未知攻击的泛化能力。
6. 孪生网络已经称为表征学习方法中的基本结构，本文使用它来学习用于欺诈语音检测任务中的相似性度量
7. 本文贡献：提出了基于孪生网络的 embedding 提取器 - 从wav2vec特征中提取表示向量。在表征学习阶段，孪生网络被训练来区分两种类型的语音属于同一类别的正样本和两种类型语音属于不同类别的负样本。在分类器训练阶段，使用从孪生网络获得的表征对两层全连接层进行训练，以最小化欺骗语音检测的交叉熵损失。


## Wav2vec 特征
用于表征学习的预训练作为一种获得音频、图像和文本的更好区分特征的方法，在标记数据稀少的任务中受到越来越多的关注。Wav2vec是一种自监督的预训练模型，输入音频信号，输出编码特征。

模型包括 encoder 网络和 context 网络。

encoder 将语音信号转换为低维特征表示 $f: X \mapsto Z$，两两间隔 10 ms，每个表征都包含了 30ms 的 16KHz 信号。

context 将 encoder 的输出转换为 context-level 的向量：$g: Z \mapsto C$，即 $C_i=g\left(Z_i \ldots Z_{i-v}\right)$ 表示感受野大小为 $v$，实际中，context 网络有 210 ms 的感受野。

两个网络都用于计算对比损失。计算如下：
$$\begin{gathered}
\left.\operatorname{sim}_k\left(Z_i, c_j\right)=\log \left(\sigma\left(Z_i^T H_k c_j\right)\right)\right) \\
L_K=-\Sigma\left(\log \sigma\left(\operatorname{sim}_k\left(Z_{i+k}, c_i\right)\right)+\underset{\widetilde{z} \sim p_n}{\lambda \mathbb{E}}\left(\operatorname{sim}_k\left(-\widetilde{z}, c_i\right)\right)\right.
\end{gathered}$$
在实际中，选择 $p_n(z)=\frac 1 T$ ，其中，$T$ 代表音频长度，$H_k$ 代表每一步的仿射变换，总损失为每步损失的求和：$L=\Sigma_{k=1}^K L_k$

预训练过程中，采用了其变体 wav2vesc large 网络，使用额外的线性变换和更大的上下文网络。

## 检测方法

### 表征学习

在表征学习阶段，模型以两个随机选择的wav2vec特征作为输入，通过由 backbone network 组成的孪生网络，backbone network 可以是 LCNN 或 ResNet。通过最小化对比损失来优化参数。

> 对比损失：相似输入对之间的距离接近0 或 不相似输入对间的距离超过某个阈值时，损失都是 0

设 $f(x_i)$ 为 wav2vec 特征，则距离定义为 $D_{i j}=\left\|f\left(x_i\right)-f\left(x_j\right)\right\|$ 表示两个特征之间的欧距离。设 $Y$ 为二元标签，$Y=0$ 表示属于相同的类，$Y=1$ 表示属于不同的类，对比损失函数定义为：
$$L_{\text {contrast }}\left(Y, x_i, x_j\right)=(1-Y) \frac{1}{2}\left(D_{i j}\right)+(Y) \frac{1}{2} \max \left(0, m-D_{i j}\right)$$
表征学习之后，孪生网络生成嵌入，使相似的语音数据点保持接近，同时将不相似的语音点分开，整个结结构如图：
![[Pasted image 20221027100954.png]]

### 孪生网络结构

可以采用不同的结构作为孪生网络的 backbone ：
在以下的高维特征提取网络中，仅使用前端 CNN 进行特征提取。使用一个全连接层从高维原始特征中获得512维嵌入。

1. LCNN：使用 9 层 LCNN，5个卷积层，4个最大池层，kernel size 为2×2，步长为2，4个 NIN 层，以及最大特征映射层。
2. ResNet18
3. SENet18、SENet34和SENet50

### 分类器

孪生网络的 embeddding 作为 分类器的输入，分类器为一个 MLP，包含：
+ 带有 BN 的 hidden layer，使用 全连接网络（输入 512 维）
+ 输出全连接网络（输出 2 维）

## 实验
数据集：ASVspoof 2019 LA

训练表征学习的时候，batch 为 64，且正负样本均分，然后随机选 50 个进行训练。

一个 epoch 的结果：
![[Pasted image 20221027102454.png]]

50个 epoch 后的结果：
![[Pasted image 20221027102523.png]]

adam 优化器，大概前 1.5 个epoch 进行warmup，lr 逐渐增加，后面的 epoch 再逐渐减小。


##  结果分析

1. 和 baseline 还有一些其他系统比：效果很好，相比于最优的 FG-LCNN [[Light Convolutional Neural Network with Feature Genuinization for Detection of Synthetic Speech Attacks 笔记]] 把 ERR 从 4.07 降到了 1.15
![[Pasted image 20221027102921.png]]
2. 和原始模型对比（也就是不用孪生网络和对比学习）：效果也有提升
![[Pasted image 20221027103115.png]]
3. 对比学习中参数 $m$ 的选择：$m=2$ 最好
![[Pasted image 20221027103310.png]]

