
1. 提出 ResNet18、LCNN9、RawNet2 三个 DNN 模型得分加权融合系统，同时也考虑了各种前端，设计 LA、PA和 DF 三个赛道
2. 本文主要关注用于提高系统泛化能力的几种方法，包括：
	1. 基于FIR滤波器的编解码器频率失真仿真增强
	2. 应用 混合技术 部分解决了过拟合问题
	3. 为了解决域外数据问题并适应PA赛道中的真实重放攻击，使用了麦克风和房间脉冲响应增强

## Introduction

1. 本文考虑了几种增强技术，目的是提高训练数据的泛化性防止过拟合、提高系统对域外数据的鲁棒性，增强技术有：
	1. 噪声增强
	2. 声学编解码器
	3. 高、低通滤波器
	4. timewarping 和 SpecAug
	5. 混合增强
	6. 脉冲响应滤波器
2. 同时关注两种类型的输入，传统的谱特征和一些新颖的特征；模型则同时考虑了 [[STC Antispoofing Systems for the ASVspoof2019 Challenge 笔记]] 中的 LCNN 模型和 一个修改的 ResNet 架构


## 提出的系统

> 详细的结果见论文。

### 特征

使用 MSTFT 特征的单系统在所有的赛道都表现较好的性能（提取的参数形式可能略有不同）。

在 PA 中，LEAF 和 LSTFT 对特征映射的扩展被证明是合理的。

RawNet2使用原始信号作为输入

### LA 系统

发现去掉输入特征归一化这一步可以提高性能；使用完整语音进行训练可以提高性能，但是使用固定长度的语音可以方便融合，因此使用了 固定的 6s 训练。

使用 MSTFT 特征，使用 特征级混合和 FIR 滤波器 作为数据增强方法。

使用修改的 LCNN9、ResNet18 和 RawNet2 作为 embedding 提取器，训练时使用 Adam 优化器和 center loss 作为损失函数来惩罚类内距离。

提取到的特征使用 biLSTM 求 average sum，然后通过全连接层得到二分类的结果，实验使用了 MHA、GAT  和 ASP 几种方法来聚合，但是效果不好。 

### PA 系统

PA 使用固定 1s 的语音，使用 MSTFT 特征，同时将 LEAF 作为特征映射的第二个通道添加到输入特征。使用 加性 MUSAN 噪声增强，同时使用混合特征图 和 RIR, IR and MIR 增强（具体见下一节）。

模型基于 ResNet18 （也用了 LCNN 和 RawNet，但是泛化性不行，有过拟合倾向），使用 Adam 优化器，也探索了不同的聚合方法，最好的是 ASP。

### DF 系统

数据用的是 LA 的，因此特征和 LA 差不多。

使用了 LCNN9, ResNet18 and RawNet2 like 模型（类似于 LA），对于 RawNet 在原始信号中使用了一些特别的混合方法，具体见下一节。

## 增强技术

### 基于 FIR 滤波器的编解码器仿真

通过使用编解码器和信号压缩方法进行信道仿真是在信号处理领域(如说话人和语音识别)训练基于 DNN 系统的一种常见方法。

作者发现并非所有电话信道仿真都能与真实的欺骗场景相匹配 ，因此应该仔细选择增强技术。编解码器可以大致分为四部分: 窄带(NB) ，宽带(WB) ，超宽带(UWB)和全带编解码器。

根据 2019 的数据集特性，使用了NB 和 WB，同时进行 在线增强以节约时间。
具体而言，使用随机选择的低频或高频滤波核与原始信号卷积来模拟编解码器的幅值响应，如图：![[Pasted image 20221214221006.png]]
使用低通来模拟NB，使用高通来模拟 WB，本质就是减轻频率范围 0-300 和 3.4K-8K 之间的频率的影响，有点类似于 SpecAug。

### Mixup

Mixup 是一种数据增强技术，通过标签间以及信号间的平滑插值来提高模型的通用性和鲁棒性。
两种mixup方法：
1. 原始信号级，使用两部分信号的拼接
2. 特征级，使用  time-spectral maps 求加权和
用数学公式描述为：$$S=\lambda \times S_1+(1-\lambda) \times S_2$$
如果使用 mixup 增强，最后的损失应该改为：$$L=\lambda \times \operatorname{loss}\left(T_1\right)+(1-\lambda) \times \operatorname{loss}\left(T_2\right)$$

### RIRs and Microphone IRs

PA 中使用 room impulse responses (RIR) 或者 microphone impulse
responses (MicIR) 可以提高性能，有两个 RIRs 数据库：
+ MIRaGe
+ 包括真实 RIR 的数据集：RWCP、REVERB、AIR等

Microphone IR database 是作者自己收集的，包含不同麦克风的脉冲响应。

对于一些模型，额外的噪声增强可以提高性能，从 MUSAN 数据集中随机采样然后加到原始数据中。

## 实验

具体的结果见论文（主要是图太多了。。）。

下面说一些结论：
1. LA 中，单独使用 mixup 效果反而变差，但是和 FIR 滤波器一起使用效果会变好，其中 RS-mixup+FIR 效果最好
2. PA 中，使用噪声增强的效果不大，RIR 的增强是最有效的，MirAGE RIR 是最好的，组合增强技术可以进一步提高性能
3. VAD 在 LA 和 DF 中无效，且降低了 PA 的效果
4. DF 单模型在  progress eval 中的 EER 很低但是在 post-eval 变得很差，可能是有点过拟合
5. LEAF 只在 PA 赛道中有改进，在 LA 和 DF 中反而性能变差

作者关于 增强技术的一些总结：
1. mixup 技术在三个赛道中都可以防止过拟合
2. mixup+FIR 一起通常比分开用的效果好
3. 在 PA 中使用 RIR、IR 和 噪声 增强
4. 在 LA 和 DF 中用 FIR 滤波器模拟编解码器的幅值响应