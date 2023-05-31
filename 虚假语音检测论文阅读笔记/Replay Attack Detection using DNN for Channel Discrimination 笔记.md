> interspeech 2017

1. 本文提出一种结合 low-level 倒谱特征的深度学习架构进行虚假语言检测，low-level 特征有：
	1. CQCC
	2. HFCC
2. 使用 DNN 来区分ASVSpoof 2017数据集中的不同信道条件，即 recording, playback 和 session conditions。
3. 从网络导出的 high-level 特征向量用于区分真实音频和欺骗音频

## Introduction


提出的 重放攻击检测 如图：![[Pasted image 20221202153859.png]]
本文的贡献包括两点：
+ 提出一种新的特征，即高频倒谱系数（HFCC），能够捕获频谱的非语音区域中的信道特性，当与CQCC一起使用时，能实现更高的精度
+ 提出 DNN特征提取器，训练以区分由于回放、录制和环境条件的变化而导致的不同条件

## Low-Level 特征提取

分析了真实特征和伪造音频之间的频谱差异，如图：![[Pasted image 20221202154226.png]]
在重放音频中，高频子带的特征的功率更低。

### HFCC（High-Frequency Cepstral Coefficients）

HFCC 提取过程：![[Pasted image 20221202163616.png]]

作者认为， recording and playback devices 会产生 channel artifacts。本文将研究重点放在浊音频率之外的高频区域。

HFCC 提取时，使用 二阶高通巴特沃斯滤波器对信号进行滤波，其他过程都和倒谱提取差不多。最终特征包含 30 维的 static、delta 和 delta-delta。

经验发现，最佳截止频率为3500Hz。


### Tandem features

实际上，低频和中频阶段也有一些区分真实和乘法语音的内容，因此将 HFCC 和 CQCC 特征结合使用。

CQCC 特征提取时，使用 zero mean 和 unit variance normalized 30 维的 CQCC 和 delta 和 delta delta。

## 前端 DNN 特征提取器

训练 DNN 时候，考虑两种不同的分类策略：
+ 真实/重放攻击的二分类
+ 区分各种信道条件的多分类，输出的类别数等于重放的配置数

经验认为第二种更好，能够更好的泛化到未知信道下。

使用的 DNN 架构为：![[Pasted image 20221202214201.png]]
输入为 $d\times N$ 的二维特征，$d$ 是特征维度，$N$ 是帧数。

三个卷积层 + 三个全连接层，训练时 dropout 为 0.3，训练 2000 个 epoch，最后一层生成语音 feature embedding。

## 实验

模型的后端基于 两个 512-component GMM 模型，使用 EM 算法进行训练。

性能：![[Pasted image 20221202215535.png]]
说明：
1. 融合 CQCC 和 HFCC （S3）确实有改进，说明两者有互补，但是 在 eval 中不明显，说明可能过拟合
2. 使用 CQCC 作为 DNN的输入，后端采用 SVM 进行分类（S5），表明 提出的 DNN 后端比 GMM 效果好（有点怪，这个 DNN 到底是前端还是后端）
3. 串联 CQCC 和 HFCC （S6）在 eval 上的结果很好，但是不知道为什么。。。