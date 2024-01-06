1. 最先进的音频生成方法在 时域和频域都存在 fingerprint artifacts and repeated inconsistencies 的问题，这种伪影可以通过频域分析捕获。
2. 本文提出一种新的长范围谱-时间调制特征，用于音频深伪检测
3. 同时采用频谱增强和特征归一化来减少过拟合
4. 基于 CNN 的 baseline 优于 ASVspoof 中的最好单系统，结合本文的 2D DCT over log-Mel spectrogram 特征称为了最先进的欺诈检测系统。

## Introduction
1. 传统的检测使用信号处理相关技术选择各种专用的特征来进行欺诈检测，但是是针对于特定的任务的而无法真正找出虚假语音和真实语音的区别
2. [[Detection and evaluation of human and machine generated speech in spoofing attacks on automatic speaker verification systems 笔记]] 中，针对机器生成的语音的显著伪影。提出一种多语音特征的轻量级模型
3. 在 cv 中，GAN 常用图像生成，这个过程伴随 fingerprint 和 信号域的伪影。在语音领域，音频生成通常是帧级别的，没有跨帧的一致性，而这可能导致时间调制伪影；同时音频生成的训练过程通常在mel谱域，采用的是均方差损失，也没有考虑跨帧的一致性；同时在语音高频处也会产生伪影。
4. 本文提出采用基于对数mel谱的长范围频域分析，同时采用2D-DCT 特征计算，用于捕获音频的深度伪影。提出的特征本质是使 CNN 分类器从音频的长期或者全局调制模式中学习。
5. 贡献：
	1. 长期谱时特征 - 全局调制特征
	2. 实现 SpecAugment 和 特征归一化
	3. 优于baseline，效果最好


## 相关工作

### 音频深伪检测
1. 随着深度伪造技术的快速发展，开发一个不受训练数据约束的检测系统，并能够准确检测不同或未知深度伪造算法生成的音频，仍然是一个挑战。
2. FoR 和 RTVCspoof 音频深度伪造数据集

### 调制特征
1. 调制特征能够捕获信号中的长时模式，本文提出的特征是能分析出联合长时谱时调制信息的全局调制特征。
2. [[Modulation Dynamic Features for the Detection of Replay Attacks 笔记]] 通过在每个子带中进行 FFT 来分析时间调制，表明 temporal dynamics 在重放攻击中的有效性。
3. 先前的 2D-DCT 仅用于局部的谱时调制，本文的全局调制结合频谱和时间调制信息，能够时间长期的特征建模

## 实验

### Baseline 模型
基于 CNN，包括一个初始化卷积、三个 residual blocks，输出再通过 GRUs 和 自注意力池化层，最后通过 MLP 进行分类，如下图：
![[Pasted image 20221025151533.png]]

### 特征
本文提出的特征是一种谱时特征：基于对数-mel谱的2D-DCT，类似于 MFCC 计算，不同之处在于在对数Mel谱的时间维度和频率维度上全局应用二维（2D）离散余弦变换（DCT），具体步骤为：
1. 使用 FFT 计算信号 $x(n)$ 的谱 $X(w)$
2. 计算功率谱，应用 Mel 频率滤波器组获得 Mel谱 $M$
> 到这一步就提取了 Mel 谱
3. 在对数 Mel 谱中计算多维 DCT，得到 $dctn_M$
4. （可选）在 $dctn_M$ 中进行 $l1$ 归一化

> 其实就非常简单，MFCC 是进行一维的 DCT 变换，而 本文提出的所谓对数Mel谱2D-DCT变换，就是对 Mel 谱进行二维的 DCT 变换。。。这样就能发一篇论文了kkk，而且还是 CMU 发的。。。

修改 [[Deep Residual Neural Networks for Audio Spoofing Detection 笔记]] 中 residual net 的架构作为检测模型，为了评估特征，模型类似于 baseline 模型，但是没有 注意力层，因为时间信息已经包含在 二维 DCT 中了。

音频通过剪切填充固定到 4s，采样率 16k，fft 大小 1024，窗口长度 512，帧移 512，mel 滤波器长度 128。

此外，对输入特征进行频谱增强，对 2d dct特征进行归一化能够提高性能。即在 log mel 谱之后实现 SpecAugment，同时使用 sklearn 分别对 2d-dct 之后的结果分别进行两种归一化：l1-norm normalization 和 mean/std standardization normalization，不过效果差不多。

## 结果

### 单系统
![[Pasted image 20221025154813.png]]
最后一行是作者提出的系统，所提出的特征在EER和t-DCF得分方面都明显优于其他特征。

### 音频类型分析
在不同欺骗类型上的性能分析：
![[Pasted image 20221025155339.png]]

### 应用于说话人验证
![[Pasted image 20221025155512.png]]
ASVspoof2019 LA训练集中的20个说话者和6种攻击类型被合并为120个“欺骗身份”。对于真实的音频，我们有正对、负对以1:1的比例随机生成。结果表明，所提出的2D调制特征不仅在检测模型中更强大，而且在音频类型和说话人验证任务中也更有效。

## 总结
1. [[Learnable Spectro-temporal Receptive Fields for Robust Voice Type Discrimination 笔记]] 中，提出的谱-时感受野（STRF）是一个局部调制特征。在ASVspoof挑战的实验中，他们得出结论：“STRF能有效地抑制干扰噪声，但其本身不足以区分真实语音和合成语音”。相比之下，他们的结果为全局计算调制特征的重要性提供了另一个证据。
2. 上面的结果都是多次运行然后取最佳结果求平均得到的，这样可以确保结果的准确性。