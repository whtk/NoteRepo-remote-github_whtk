> Odyssey 2020 The Speaker and Language Recognition Workshop


1. 本文通过使用 large margin cosine loss 和 online frequency masking 增强 技术 使得神经网络学习更鲁棒的特征进行欺诈检测

## Introduction

1. 传统方法中，通过研究设计不同的特征来进行欺诈检测，如 CQCC、MGD
2. Towards robust audio spoofing detection: a detailed comparison of traditional and learned features 论文中 研究了不同的声学特征和学习到的特征
3. 本文重点在于提高模型的泛化能力，采用 LMLC 损失函数最大化真实类和欺骗类之间的差异，同时最小化类内差异；同时使用 FreqAugment 以进一步提高DNN模型的泛化能力
4. 还研究了音频增强技术的有效性，使用公开可用的噪声来增强音频以进一步降低 EER
5. 最后 研究了所提出的欺骗检测系统在呼叫中心环境中的性能


## 数据

采用 ASV sspoof 2019 数据集，同时使用两种数据增强：
1. 混响：RIR
2. 背景噪声：四种噪声，来自于 MUSAN噪声语料库，还有一部分电视噪声

为了模拟呼叫中心环境中的语音欺骗，重放 LA 数据集中的语音，使得所得数据集具有VoIP信道特性，并将带宽从16kHz降低到8kHz采样率。在训练和测试期间，数据集被上采样到16kHz。

得到的数据集如图：![[Pasted image 20221224093152.png]]


## 方法

整个模型如图：![[Pasted image 20221224094701.png]]

### Low-level 特征

使用 linear filter banks（LFB）作为特征，为 SFT 对的压缩版本，降低了过拟合的风险。

使用 60 维的特征，进行了均值和方差归一化，没有进行 VAD 操作。

### Frequency Masking

在训练的时候进行 Online frequency masking，丢弃一定 频率范围 $\left[f_0, f_0+f\right)$ ，$f$ 来自于均匀分布。$f_0$ 来自于 $[0, v-f]$，$v$ 为 LFB 总的 frequency channel 数量。在每个 batch 中这两个值都随机选择。

### LMLC

使得模型可以最大化类间距离最小化类内距离，定义如下：$$L_{l m c}=\frac{1}{N} \sum_i-\log \frac{e^{s\left(\cos \left(\theta_{y_i, i}\right)-m\right)}}{e^{s\left(\cos \left(\theta_{y_i, i}\right)-m\right)}+\sum_{i \neq y_i} e^{s \cos \left(\theta_{j, i}\right)}}$$
且有 $$\begin{aligned}
W & =\frac{W^*}{\left\|W^*\right\|}, \\
x & =\frac{x^*}{\left\|x^*\right\|}, \\
\cos \left(\theta_j, i\right) & =W_j^T x_i
\end{aligned}$$
其中，$N$ 为样本数，$s,m$ 为超参数，用来决定在 cosine space 中的 margin。

### 网络

baseline 是基于 ResNet 的模型，提出的模型基于 baseline 做改进，具体的模型如图：![[Pasted image 20221224102538.png]]

FC2 层的输出作为 feature embedding。

ResNet embedding 提取器和后端分类器是分开训练的，都采用 adam 优化器 训练 50 epoch。

## 实验&结果 

不同系统训练的结果：![[Pasted image 20221224102944.png]]

不同 protocol 训练的结果：![[Pasted image 20221224103015.png]]

结论：
1. LCML能够迫使模型学习具有更好泛化能力的更鲁棒的特征
2. frequency masking 可以进一步降低 EER
3. 采用增强的数据进行训练可以降低 EER

