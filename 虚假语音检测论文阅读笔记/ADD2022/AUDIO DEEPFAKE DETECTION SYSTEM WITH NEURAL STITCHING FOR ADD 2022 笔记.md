1. 用于赛道3.2的，使用标准的 34层 ResNet + multi-head attention pooling 来学习虚假语音的 embedding
2. 同时使用 neural stitching 提高模型的泛化性能
3. 最终在 3.2 的赛道中获得了 10.1% 的 EER，排名第一

## Introduction

1. 现在 SS 和 VC 技术很强，使得反欺诈特别重要，而且也很难
2. ADD 2022 的举办满足了这个需求
3. 本文提出能够更有效的检测虚假语音的系统，分析了不同的数据增强能够极大的提高总体性能

## 系统描述

### 方法

1. ResNet 可以捕获区分性的局部特征，然后聚合成 utterance level 的 embedding
2. 提出的方法包含两个网络：
	1. embedding 网络： ResNet-34 + attention pooling（聚合 frame level 的特征），同时计算输出的一阶和二阶矩，一起作为 utterance level 的特征
	2. 分类网络：两个全连接层+2d 的softmax 用于区分真假
	3. 如图：![[Pasted image 20230316173143.png]]
	4. embedding 网络中用 relu 激活，分类网络中用 mish（能够在分类任务中获得更好的性能），采用 focal loss 进行训练

### Neural Stitching

特征有三个类型：
+ 等变性，equivariance，Equivariance studies how transformations of the input image are encoded by the representationinformation or not
+ 不变性，invariance，invariance being a special case where a transformation has no effect
+ equivalence，Equivalence studies whether two representations, for example two different parametrisations of a CNN, capture the same visual information or not
为了研究 equivalence，训练两个不同的 CNN 模型，然后分成四个部分，$\phi=\phi_1 \circ \phi_2$ 和 $\phi^{\prime}=\phi_1^{\prime} \circ \phi_2^{\prime}$，目标是找到映射 $E_{\phi_1->\phi_1^{\prime}}$ 使得 $\phi_2^{\prime} \circ E_{\phi_1->\phi_1^{\prime}} \circ \phi_1$ 能够获得和原始的 $\phi_2^{\prime} \circ \phi_1^{\prime}$ 相同的性能。

这个映射表示为 stitching layer，可以通过一组滤波器或者变换层实现。stitching 操作之后模型的分类性能显著优于之前，表明 **在同一数据集上训练的不同网络之间进行一定程度的特征交换可能是有帮助的**。

受此想法启发，作者希望通过将模型层分解为不同的 stitching part 来了解stitching 在单个模型中是如何工作的。先前的一些工作表明，网络的浅层倾向于学习更一般的表征，而网络的深层更专注于特定的任务。因此，我们在推理阶段 cut off one deep layer ，并将分类层直接 stitch 到ResNet的输出，结果出乎意料地好。

上图右边显示了推理的过程。
> 这不就是直接丢掉一层吗？？？

## 实验和结果

### 数据集和特征

将训练集和验证集合在一起得到 55K 条语音 作为总的训练集，每个语音分成固定长度的段，重叠 50%。然后随机选 4000 个段作为验证集，剩下的作为测试集。把 赛道1 和赛道 3.2 的 adaption set 作为评估集。

选择了三个特征：
+ LFCC
+ DCT-DFT spec，和 LFCC 提取过程差不多，只不过没有 linear filterbank 那个步骤
+ Log-linear filterbank energy (LLFB)，和 LFCC 提取过程差不多，只不过没有 DCT 那个步骤

之前的研究证明，静态特征优于动态特征，因此这里只使用了静态特征（没有 delta），结果如下：![[Pasted image 20230316200528.png]]

80维的 ：LFCC，FFT=1024，效果在赛道1是最好的。

应用了两种数据增强：
+ 加入 distortion 和 noises，数据集来自 RIR 和 MUSAN，包含reverb,  noise, music 和 babble 四种，SNR 是0-20 之间的随机数；音量也有影响，把音量随机设置为 -10dB~20dB，最总获得了五倍的数据量（4+1），然后随机 sample 60K 条语音
+ 模拟音频压缩的影响：所有的clean音频都通过音频压缩算法进行模拟，压缩算法包括：MP3, OGG, AAC 和 OPUS。最后还模拟电话的传输，将音频先下采样到8k然后上采样到16k，又获得了五倍的数据（4+1），然后随机选 40k 条。
![[Pasted image 20230316201511.png]]
最终的效果如下：![[Pasted image 20230316201534.png]]
加入噪声训练后，EER 急速下降，说明性能很好！

### 训练

池化层对最终的结果影响很大，本文比较了五种类型的池化层，发现 learnable dictionary encoding(LDE)pooling 和 self multi-head attention(MH) 效果不错。

不使用 neural stitching 之前，有些得分太接近于1 或者 0（太极端了），可能是由于发生了过拟合，使用 stitching 之后得分就合理了很多。

测试集中，使用 neural stitching 能够提升 EER 的性能（EER值降低），之后还进行了一系列的 fine tune 来得到最终的结果：![[Pasted image 20230316202300.png]]

fine tune 的时候，采用了 spec augment 数据增强，同时改变不同的 chunk size，最终结果如下：![[Pasted image 20230316202535.png]]
最好的可以实现 8.83% 的 EER。

然后在上面最优的模型的基础上，将数据重叠从 50% 增加到 70%，将 lr 降低一百倍，继续在这个模型的基础上进行训练。

### 最终提交结果
![[Pasted image 20230316202730.png]]
赛道 3.2 的第一。

> 点评：工程味严重，创新性不足，但是效果确实好。