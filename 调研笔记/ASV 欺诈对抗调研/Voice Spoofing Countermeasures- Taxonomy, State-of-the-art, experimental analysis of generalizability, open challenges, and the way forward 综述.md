> 2022 年 11 月最新综述

1. 回顾了有关使用手工特征、深度学习、端到端和通用欺诈对策解决方案来检测语音合成（SS）、语音转换（VC）和重放攻击的欺骗检测的文献
2. 回顾了语音欺诈评估和说话人验证、针对语音对抗的对抗性和反取证攻击以及ASV的集成解决方案
3. 介绍了现有欺诈对策的局限性和挑战性，对其进行了跨语料库的评估

## Introduction

1. ASV 欺骗攻击可以分为：
	1. PA：如 重放攻击
	2. LA：如 SS 和 VC
2. 重放攻击更容易实现，包括：
	1. single-hop：使用一个记录设备重放音频
	2. multi-hop：使用多个记录设备，也就是多次记录和重放
3. 还要一种最新的威胁是对抗样本
4. 已有的一些反欺诈措施可以分为两个部分：
	1. 前端特征提取
	2. 后端分类器
5. 本文贡献：
	1. 提出了一个语音反欺诈的 baseline survey，包含各种因素
	2. 全面分析了现有的 SOTA 攻击+对策、数据集、评估参数
	3. 使用VSDC、ASVspoof 2019和2021的数据集 再多个分类器上对最先进的不同对策进行了实验分析，跨语料库评估结果
	4. 在 min-tDCF and EER 这两个指标中跨语料库评估，测试了四种不同的机器学习和深度学习分类器

## 语音欺诈攻击
ASV 系统可能受到的欺诈攻击如图：![[Pasted image 20230308164817.png]]
1. PA 更简单也更常见
2. 虚假语音检测（PAD）旨在检测以下攻击：
	1. 直接攻击，可以分为设备伪影和算法伪影，分别指 PA 和 LA：
		1. PA：通过麦克风将样本输入到 ASV 系统
			1. 重放攻击：最容易实现也最容易欺骗
			2. 模拟攻击：欺骗人通过改变说话方式来模仿合法用户
		2. LA，绕过sensor（如麦克风），直接送到模型中，包含 VC、SS、对抗攻击：
			1. VC 使用 imposter 的声音来生成 人造的声音，目的是匹配目标说话人的声音，但是这些攻击仍然可以被检测，因为不是完美的匹配
			2. SS 攻击，也称 deepfake 攻击，使用文本作为模型输入，生成目标说话人的声音
	2. 不直接攻击：包括对抗攻击，音频信号保持不变的，攻击者在 ASV 处理期间修改信号的特征
	3. 总结：![[Pasted image 20230308165747.png]]

### 针对对抗攻击的对策
Defense against adversarial attacks on spoofing countermeasures of asv 使用自监督学习，利用无标签数据的知识。通过自监督模型创建高层的表征，然后l使用 ayer-wise noise to signal ratio (LNSR) 来评估模型的检测质量，仅在 PGD 和 FGSM 两个攻击上进行了测试。

[[Defense for Black-box Attacks on Anti-spoofing Models by Self-Supervised Learning 笔记]] 使用空间平滑、滤波和对抗训练，但是仅在 PGD 攻击上效果最好。

Light convolutional neural network with feature genuinization for detection
of synthetic speech attacks 使用模型拟合真实语音的分布，模型的输入是真实语音输出也是真实语音，而对于对抗的虚假语音，模型的输出非常不同，从而使得输入和输出的差异比较大。模型采用的是 Transformer + LCNN 的结构，但是没有跨语料库中测试。

Long-term high frequency features for synthetic speech detection 引入 inversion module ，输入四种特征，ICQC、ICQCC、ICBC 和 ICLBC，在 ASVspoof 2015、17、19 的 LA 上进行测试。
> 这和 对抗攻击 有关系吗 ？？

Phoneme specific modelling and scoring techniques for anti spoofing system 


## 语音欺诈检测分类
> 详细分析了用于检测欺骗攻击的技术，以及为对抗这些攻击开发的对策
> 对策的分类： ![[Pasted image 20230325195149.png]]
> 

### 传统、手工欺诈检测

特征通常是 MFCC、CQCC，分类器通常是 GMM 或 SVM。



### 讨论

### 基于深度学习的对策

[[Robust Deep Feature for Spoofing Detection - The SJTU System for ASVspoof 2015 Challenge 笔记]] 提出了一种新颖且简单的模型，从音频中提取关键特征构建 compact, abstract 和 resilient 表征，训练了一个 spoofing-discriminant 网络提取所谓的 s-vector，最终使用马氏距离和归一化进行欺诈检测。

[[Replay Attack Detection using DNN for Channel Discrimination 笔记]] 提出 HFCC 特征 + DNN 分类进行检测，重点分析了高频区域。

[[Attentive Filtering Networks for Audio Replay Attack Detection 笔记]] 提出一种 attentive filtering 网络，提出的方法同时在时域和频域进行特征增强，模型由两个 AF 层组成，效果优于 CQCC-GMM。

Audio replay spoof attack detection by joint segment-based linear filter bank feature extraction and attention-enhanced densenet-bilstm network 提出基于分段的线性滤波器特征和 attention-enhanced DenseNet-BiLSTM 模型，特征从音频信号的静音段提取，采用三角滤波器估计高频的噪声（3-8KHz），但是在低信噪比下容易出错。

[[Light Convolutional Neural Network with Feature Genuinization for Detection of Synthetic Speech Attacks 笔记]] 采用 LPS 特征，创建了一个模型来拟合真实语音的分布，其输入和输出都是真实语音（AE？），而如果输入是虚假的语音，输出则相差很大。

[[RW-Resnet- A Novel Speech Anti-Spoofing Model Using Raw Waveform 笔记]] 提出带有残差连接的 Conv1D Resblock，使模型可以从原始语音学习更好的特征表示。

[[End-to-End Spectro-Temporal Graph Attention Networks for Speaker Verification Anti-Spoofing and Speech Deepfake Detection 笔记]] 是 GAT-ST 的拓展，输入为原始波形，模型捕获谱时特征，采用特征表示学习和 GAT 来学习不同子带和时间间隔的欺诈线索。

Improving anti-spoofing with octave spectrum and short-term spectral statistics information 提出一种新的特征，eCQCC-STSSI，使用 CQT 获得频域特征，然后用两个 DCT 分别用于去除特征维度相关性和能量拼接，两个 DCT 的输出作为 eCQCC 特征向量。

Study on feature complementarity of statistics, energy, and principal information for spoofing detection 使用四个特征，STSSI，OPI，FPI 和 MPEI 融合生成 delta acceleration coefficients 作为欺诈检测的特征。

Extraction of octave spectra information for spoofing attack detection 作者提出了一种基于多级变换（MLT）的启发式特征提取方法，从倍频程功率谱中提取有价值的信息，用于欺骗攻击检测。

Phoneme specific modelling and scoring techniques for anti spoofing system 比较了真实和虚假语音的不同音素，表明特定的音素在检测重放攻击时信息更丰富，创建了四种不同的融合评分方法，使用音素特定模型来整合语音信息。

Voicepop- A pop noise based anti-spoofing system for voice authentication on smartphones 提出 VoicePop，通过检测说话人在靠近麦克风说话时呼吸自然产生的 pop noise 来识别 live user，采用的是 GFCC 特征

[[STC Antispoofing Systems for the ASVspoof2019 Challenge 笔记]] 
基于最大特征图（MFM）激活的轻量级 CNN 来检测重放攻击。

### 端到端的对策



### 讨论
