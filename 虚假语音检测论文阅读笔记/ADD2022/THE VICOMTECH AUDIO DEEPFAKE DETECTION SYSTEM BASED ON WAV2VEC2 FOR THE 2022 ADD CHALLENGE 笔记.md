>  ADD 2022

1. 提交在 ADD 2022 赛道 1 和 赛道 2 的系统
2. 使用预训练的 wav2vec2.0 作为特征提取 + 下游分类器 检测伪造音频，利用 Transformer 不同层的上下文语音表征来捕获有区分性的信息
3. 同时使用了不同的数据增强技术来适用新场景

## Introduction

1. 现在的 DF 检测重点在于真实场景下的鲁棒性
2. ASV spoof 2021 考虑通过不同电话网络传输的合成语音，以及一些压缩算法修改的伪造语音
3. ADD 2022 的主要目标是检测深度合成和操纵的音频
4. 本文使用基于 wav2vec2 的方法进行虚假语音检测，采用预训练的模型作为特征提取器，同时使用了不同的 Transformer 层的特征。最后用一个简单的下游分类器模型进行检测

## 方法
![[Pasted image 20230319204607.png]]

### Wav2vec2 特征提取器

分析了 large 的模型，分别采用了 53 和 128 种语言训练，对应 XLS-53，XLS-128。

原始音频首先通过几个 CNN 层的特征提取器，每 20 ms 抽取 1024 维的特征，感受野的范围为 25 ms，然后特征通过 24 层 Transformer获得语音的上下文表征。模型通过对比损失采用自监督学习进行训练。目标是通过上下文表征预测被 mask 的表征的量化值（不是原始值）。

### 分类模型
![[Pasted image 20230319205545.png]]
预训练模型的最后一层 Transformer 的输出可以用于某些语音任务，但是，先前的工作表明，对于一些其他任务使用第一层或者中间层可以得到更多有区分性的信息。

本文使用不同 Transformer 层的表征作为下游模型的输入：
1. 首先，在每层的 Transformer 的输入应用  temporal normalization
2. 对于每个时间步 $t$，每层的输出表征为 $\mathbf{h}_{t, l}$，计算总的输出 $\mathbf{o}_t=\sum_{l=0}^L \alpha_l \mathbf{h}_{t, l}$，$l$ 代表层索引，$\alpha_l$ 为可训练的参数（加起来为1）。
3. 然后将总输出送到两个带 ReLU 和 dropout 的 FF 层中，通过一个 attentive statistical pooling 得到一个单一的 表征
4. 再接 FF 层得到最终的 embedding $\mathbf{e}$
5. 计算余弦相似度 $S=\cos (\mathbf{w}, \mathbf{e}) \in [-1,1]$，其中 $\mathbf{w}$ 是真实语音的 embedding。

模型通过 one-class softmax 损失，训练使得真实语音的得分尽可能地高。

## 实验

对每个 challenge，只使用对应的数据集进行训练。

### ADD 2022 数据集

基于 AISHELL-3 的 clean 语音，包含合成和转换。training 和 dev set 包含 28K 不重叠说话人的语音。

赛道 1 和 2 都包含一个 1K 语音的 adaptation set，100K 语音的 test set（无标签）。adaptation set 和 test set 的音频条件相似，用于使系统适应 test set。

### ASVspoof 2021 数据集

略

### 数据增强

在训练的时候，on the fly 地应用数据增强。主要的增强就是对语音使用 低通 FIR 滤波器，用于估计传输伪影和编解码器的影响。同时mask信号的一部分频率来提高泛化性能。

类似于 [[STC Antispoofing Systems for the ASVspoof2021 Challenge 笔记]] 中的过程，评估了窄带和宽带 FIR 滤波器。

对于 ADD2022，把训练和对应的 adaptation set 合并来提高泛化性。

对于赛道2 ，在每个 epoch 中，选择 20 % 的真实语音，然后选择对应的set中的不同话语的一段可变语音段，将该片段和原始语音随机重叠。

### 训练设置

Adam 优化器，默认学习率，dropout = 0.2，训练时 wav2vec2 的参数固定，只更新分类器参数，batch 为 8，梯度累计为 8，在 dev set 上性能连续10个 epoch不变时停止训练。

## 结果

在 ADD 2022 数据集上的结果：![[Pasted image 20230319214745.png]]
1. XLS-28 的总体效果更好
2. 主要改进还是来自于 adaptation set 的引入
3. 采用 FIR 可以将 EER 降低越 1%

在 ASV spoof 数据集上的效果：
![[Pasted image 20230319215152.png]]
1. NB 对于 LA 的效果很好，DF 更喜欢 WB

和其他系统比：![[Pasted image 20230319215415.png]]

不同层的权重：
![[Pasted image 20230319215459.png]]

> 整个过程中，wav2vec2.0 的模型是用现成的也不会被训练也不需要 fine tune 吗？？？
> 那如果 fine tune 了效果岂不是会更好？