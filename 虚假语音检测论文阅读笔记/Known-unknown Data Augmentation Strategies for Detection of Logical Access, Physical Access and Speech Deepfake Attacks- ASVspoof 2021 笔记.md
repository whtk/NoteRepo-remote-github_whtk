
1. 本文研究了一些数据增强方法来实现更为鲁棒性的欺诈检测
2. 考虑基于 CQT 变换的 logspec，使用 LCNN  网络进行数据增强

## Introduction

1. 现有的对策要么侧重于新的前端，要么侧重于有效的分类器
2. 一些能够捕获伪影的特征包括：cochlear filter cepstral coefficient and instantaneous frequency、instantaneous phase、LFCC、subband spectral flux coefficients 和 spectral centroid frequency coefficients，同时基于 CQT 的 CQCC 特征也有一定的优势
3. 在分类器方面，LCNN、SEnet、capsule network、VGG 和 SincNet 都有着较大的潜力
4. 本文将基于 CQT 的 lopspec 和 LCNN 作为 test bench 研究数据增强，现有的研究包括 [[The DKU Replay Detection System for the ASVspoof 2019 Challenge- On Data Augmentation, Feature Representation, Classification, and Fusion 笔记]]、[[Data augmentation with signal companding for detection of logical access attacks 笔记]]
5. 本文将提出的方法分为已知攻击的数据增强和未知攻击的数据增强，在三个赛道上调研各种数据增强策略 ，已知的方法基于攻击的具体信息，未知的方法基于赛道类型

## Known-unknown Data Augmentation

### LA 赛道

ASV spoof 2021 的LA数据是通过了特定的编解码器得到的，即使用了两种信号压缩算法，G.7112 (alaw) 和  G.7223（称基于这两个编码器的方法为已知的数据增强方法），同时还考虑了其他信号压缩算法（对应未知的数据增强算法。）

### PA 赛道

ASV spoof 2019 的重放数据集是模拟出来的，而 2021 的  PA 数据集是真实的。因此在 2019 年的数据集训练在 2021 的数据集测试会导致很大的不匹配。

所以使用数据增强使2019的数据更接近真实重放场景。使用了 外部噪声数据库的噪声样本和训练数据相加，将这种方法称为未知数据增强。

### DF 赛道

DeepFake 攻击的目的是欺骗人类而非 ASV 系统，其数据也是使用 TTS 和 VC 方法生成的，也使用了各种通用的压缩算法。

作者将 mp3和m4a压缩方法作为 DF 赛道中的 已知数据增强方法。

其他压缩方法 mp2、ogg、tta、wma、aac和ra 为未知数据增强方法。

## 实验

ASV2019 LA 是作为 2021 DF 赛道的训练集的。其他关于数据集的信息见 ASVspoof2021的官网。

除了使用 ASVspoof2019的数据进行训练，还用了NoiseX-9212语料库进行数据增强。

NoiseX-92语料库包含各种噪声类别，本文只使用了 factory1, factory2 and volvo noise 这三种噪声。

实验设置：基于 CQT 的 logspec + LCNN，同时考虑 2021 给的baseline 进行得分融合

结果：

1. 在 progress phase 中的结果：![[Pasted image 20221213110628.png]]结论：基于 DNN的模型在 LA 和 DF 效果更好，传统分类器在 PA 效果更好

最终提交的结果：![[Pasted image 20221213111733.png]]

其他的效果见论文，反正总结起来就是，使用数据增强可以提高检测性能。
