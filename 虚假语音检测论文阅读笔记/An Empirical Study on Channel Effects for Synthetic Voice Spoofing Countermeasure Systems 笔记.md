> interspeech 2021

1. 本文对几种最先进的 CM 系统进行跨数据集的研究，观察到与单数据集相比，性能发现显著下降，并认为数据集之间的信道不匹配是一个重要的原因
2. 通过在原始数据集训练但是在信道偏移数据集上评估 CM 系统，类似的退化验证了这一点
3. 针对 CM 系统提出了几种信道鲁棒性策略（数据增强、多任务学习、对抗学习等）

## Introduction

1. 作者认为，CM 系统在不同的数据集中进行训练和测试性能下降的原因，一方面是未知对抗更具有挑战性，另一方面，数据之间的信道变换也是很可能的原因
2. 本文所指的信道效果是指在记录和传输过程中施加在语音中的音频效应，如记录环境的混响、记录设备的频率响应、信号压缩算法等；如果不对这些影响进行适当的考虑和补偿，CM系统可能会过拟合训练集中的信道影响，而无法泛化到未知的信道变化。这些问题已经在重放攻击[[Cross-domain replay spoofing attack detection using domain adversarial training 笔记]]中得到了研究，但是在LA中不受关注。
3. 本文贡献：
	1. 首先对 ASVspoof2019LA、ASVspoif2015和VCC2020 三个数据集进行交叉数据集研究，发现 CM 性能都下降
	2. 然后比较三个数据集中真实语音的平均幅度谱，发现不匹配
	3. 在ASVspoof2019LA上训练三个CM系统，并在其信道增强版本ASVspoif2019LA-Sim的评估集上对其进行评估，来进行受控的跨信道实验，不出意外的性能退化验证了猜想
	4. 提出了几种利用信道增强数据提高信道鲁棒性的策略，改善了性能

## 跨数据集研究

### 三个数据集的信息（略）

### 实验设置

采用了在 ASVSpoof2019LA 数据库中SOTA的系统来进行实验，分别为：
+ LCNN （[[STC Antispoofing Systems for the ASVspoof2019 Challenge 笔记]]）
+ ResNet （[[Generalized end-to-end detection of spoofing attacks to automatic speaker recognizers 笔记]]）
+ ResNet-OC （[[One-class Learning Towards Synthetic Voice Spoofing Detection 笔记]]）

特征：60 维的 LFCC，帧长 20 ms，帧移 10 ms，音频长度固定为 750 帧（不够的进行重复填充，有多的随机选连续片段），lr 为 0.0003，每 10 epoch 减半，一共 100 epoch

每个 CM 的输出都是得分，评估指标为 EER。

### 结果、分析

三个模型在三个数据集中的EER如下（都是在 ASVSpoof2019LA数据集中训练的）：
![[Pasted image 20221030103845.png]]
可以看到，在不同的数据集中性能发生了大幅下降。
下图表示了 ResNet-OC在不同数据集中虚假和真实语音的得分分布：
![[Pasted image 20221030104411.png]]
可以看到，ResNet-OC 系统在虚假语音上的效果还不错，但是在真实语音中发生了大量的偏移（尤其是 VCC2020 数据集，把所有的真实语音都判断成虚假语音了）。
下图是三个数据的平均幅度谱，可以看到，差别很大：
![[Pasted image 20221030104651.png]]
这说明，信道不匹配是EER退化的重要原因。

### 信道仿真

通过开源模拟器，生成 ASVspoof2019LA-Sim 信道增强数据集，采用的是 12 种不同的信道增强方法（如通过不同的环境脉冲响应滤波器），此时平均幅度谱如下：
![[Pasted image 20221030105221.png]]
再次实验，最后的结果如下：
![[Pasted image 20221030105642.png]]
不出意外的性能下降。

## 信道鲁棒性策略

首先将在 ASVspoof2019LA 训练集中训练的模型称为 Vanilla 模型作为 baseline；

基于 ASVspoof2019LA-Sim 数据集，提出三种信道鲁棒策略：
+ 增强（AUG）：使用 Vanilla 模型 但是在 ASVspoof2019LA-Sim 数据集中进行训练
+ 多任务增强（MT-AUG）：在 Vanilla 模型 中添加一个信道分类器，如图右：![[Pasted image 20221030145422.png]] 中的信道分类器模型（没有 GRL），采用两个全连接层直接输出信道的标签，最后用交叉熵损失训练模型：$$\left(\hat{\theta}_e, \hat{\theta}_{c m}, \hat{\theta}_{c h}\right)=\underset{\theta_e, \theta_{c m}, \theta_{c h}}{\arg \min } \mathcal{L}_{c m}\left(\theta_e, \theta_{c m}\right)+\lambda \mathcal{L}_{c h}\left(\theta_e, \theta_{c h}\right)$$
+ 对抗增强（ADV-AUG）：在模型中加入一个 梯度反转层（Gradient Reversal Layer），信道分类器通过 GRL 进行反向传播。然后嵌入网络（也就是左边的那块）的目的是使得信道分类误差最小化，而信道分类器的目的是使得分类误差最大化，从而形成一个对抗训练：$$\begin{aligned}
\left(\hat{\theta}_e, \hat{\theta}_{c m}\right) &=\underset{\theta_e, \theta_{c m}}{\arg \min } \mathcal{L}_{c m}\left(\theta_e, \theta_{c m}\right)-\lambda \mathcal{L}_{c h}\left(\theta_e, \hat{\theta}_{c h}\right) \\
\left(\hat{\theta}_{c h}\right) &=\underset{\theta_{c h}}{\arg \min } \mathcal{L}_{c h}\left(\hat{\theta}_e, \theta_{c h}\right)
\end{aligned}$$
> GRL 其实已经用于重放攻击检测了

### 域内测试

训练集用的是 ASVspoof2019LA-Sim - train 中的数据，包含 CH01-10 的信道增强，在评估集进行测试（也就是包含已知信道 CH01-10 和 未知 CH11-12 信道）。

结果为：
![[Pasted image 20221030113632.png]]
相比于 Vanilla 模型获得了巨大的提升。

> 最后还分析了 DET 曲线，具体见论文。

### 域外测试

在域外数据集上的效果也很好：
![[Pasted image 20221030144417.png]]
这次的得分分布为：![[Pasted image 20221030144707.png]]
可以看出，真实语音的得分分布中，在 ASVspoof2015 数据集改善很大， 但是 VCC2020 还是有待改善，表明提出的信道鲁棒策略仍然有局限。