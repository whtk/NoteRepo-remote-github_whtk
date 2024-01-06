> interspeech 2022

1. 提出了一种基于 VIB 的wav2vec 2.0 预训练模型的迁移学习方法进行语音反欺诈
2. ASVspool 2019 LA 中优于最好的系统，同时在低资源和跨数据集的情况下也能显著提高性能

## Introduction

最近的反欺诈系统通过构建复杂的网络来提高对未知攻击的检测性能。

特别是，很多系统研究各种网络架构，并使用原始波形作为输入以期实现更好的性能，如 [[RW-Resnet- A Novel Speech Anti-Spoofing Model Using Raw Waveform 笔记]]、[[End-to-End Spectro-Temporal Graph Attention Networks for Speaker Verification Anti-Spoofing and Speech Deepfake Detection 笔记]]、[[Towards End-to-End Synthetic Speech Detection 笔记]]

  
但是训练数据有限的问题仍然限制了表征的泛化能力。为了解决这一问题，一些系统侧重使用预训练模型通过迁移学习来找到表征，如 [[Investigating self-supervised front ends for speech spoofing countermeasures 笔记]]、[[Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation 笔记]]

  
[[Siamese Network with Wav2vec Feature for Spoofing Speech Detection 笔记]] 使用预训练wav2vec模型进行迁移学习，并采用孪生网络进行反欺诈。

本文使用预训练的模型将 IB 与迁移学习相结合。采用变分信息瓶颈（VIB）来实现IB。VIB为损失函数提供了一个正则化项，它抑制了潜在表示中的无关信息。在预训练模型之后添加VIB模块有助于通过仅压缩与任务相关的有意义的信息来提取广义表示。  

总的来说，本文旨在通过使用基于有 VIB 的wav2vec 2.0预训练模型的迁移学习方法来提高语音反欺诈的泛化性能。我们利用 VIB 将语音嵌入映射到潜在特征 z，通过抑制表面和冗余信息来正则化潜在特征以学习广义信息。

## 方法

### Wav2vec 2.0

原理见 [[../语音自监督模型论文阅读笔记/wav2vec 2.0- A Framework for Self-Supervised Learning of Speech Representations 笔记]]

### Variational information bottleneck

原理见 [[VIB]]

### Wav2vec 2.0 + VIB 架构

架构如图：![[Pasted image 20221203140119.png]]
Wav2vec 2.0 模型使用自监督方法进行预训练。假设模型的参数为 $\psi$，输入原始波形 $s$ 结果 Wav2vec 2.0 模型 $f_\psi(\cdot)$ 后转换成 speech embedding $x=f_\psi(s)$，然后将 $x$ 通过MLP 进行降维，通过 MLP 之后，使用两个线形层来建模后验分布 $p_\theta(z \mid x)$ 的均值 $\mu(g(x))$ 和方差 $\Sigma(g(x))$，同时使用重参数技巧计算 $z=\mu(g(x))+\epsilon \odot \Sigma(g(x))$，其中 $\epsilon$ 服从标准正态分布。

得到 $p_\theta(z \mid x)=\mathcal{N}(z \mid \mu(g(x)), \Sigma(g(x)))$ 后，从分布中进行采样得到压缩后的语音表征 $z$，将其送入到另一个 MLP 分类器 $q_\phi(y \mid z)$ 来判断语音的真假。  

因为 $z$ 它是从方差为 $\Sigma(g(x))$ 的后验高斯分布采样的，所以 $z$ 包含一定的噪声。这种随机性使得网络不学习依赖于数据集的表面信息，而是学习反欺诈任务的广义信息。

## 实验设置

数据集：ASVspool 2019 LA，同时使用 ASVspoof 2015 和 ASVspouf 2021 作为 cross-dataset

评估指标：EER 和 min-t-DCF

baseline：RawNet2 [[End-to-End Anti-Spoofing with RawNet2 笔记]] 和 AASIST [[AASIST- Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks 笔记]]，同时比较了低资源和跨数据条件下的效果。

使用 PyTorch Lightning 库构建实验，采用了 Huggingface 中的 wav2vec 2.0 预训练模型，模型的输出在时间轴进行 mean-pooled ，最终生成 768 维的 speech embedding。

使用交叉熵损失，且分别为 真实语音和虚假语音分配0.9和0.1的权重。使用 Adam 优化器进行学习。

## 结果

1. 和当前最好的模型进行比较![[Pasted image 20221203154456.png]]仅用 Wav2vec 2.0 的效果就很好了。

论文中还比较了低资源和跨数据集的效果。



