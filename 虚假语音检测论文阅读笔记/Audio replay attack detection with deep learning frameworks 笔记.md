
1. 研究基于深度学习方法的重放攻击检测
2. 效果优于 baseline

## Introduction

1. 本文研究使用卷积神经网络解决 RA 的问题
2. [[Spoofing speech detection using temporal convolutional neural network 笔记]] 提出将 temporal CNN架构用于VC和SS欺骗语音检测，取得了显著的效果；[[An Investigation of Deep-Learning Frameworks for Speaker Verification Antispoofing 笔记]] 使用 DNN 进行 VC 和 SS 欺骗检测，提出了一种 CNN+RNN 架构展示了最先进的性能
4. 本文使用 reduced LCNN 架构，  具有MFM的神经网络能够进行特征选择从而用于欺诈检测

## Baseline 系统

1. CQCC+GMM
2. LPCC+i-vector+SVM

## DNN 框架
> 使用高级特征提取和端到端方法

### 前端

使用 CQT 和 FFT 获得归一化 logspec。

两种方法获得固定的输入：
+ 沿时间轴截断具有固定大小的频谱
+ 固定窗口大小的滑动窗口

### CNN

基于 MFM 激活的 CNN 进行检测，MFM 定义为：$$\begin{aligned}
&y_{i j}^k=\max \left(x_{i j}^k, x_{i j}^{k+\frac{N}{2}}\right) \\
&\forall i=\overline{1, H}, j=\overline{1, W}, k=\overline{1, N / 2}
\end{aligned}$$
其中，$x$ 为 $H \times W \times N$，输出为 $H \times W \times \frac{N}{2}$ ，$i,j$ 为时频轴，$k$ 为通道轴，下图表明了  MFM 的原理：![[Pasted image 20221126103053.png]]

RELU 是通过阈值来抑制神经元的激活，而 MFM 是通过竞争关系来实现，从而相当于一个特征选择器。

使用的 LCNN 架构为：![[Pasted image 20221126103457.png]]

后端分类采用 GMM 模型。

### CNN+RNN

CNN 用于特征提取，RNN 建模长期依赖。两个模型通过反向传播联合优化，其架构为：![[Pasted image 20221126103701.png]]


## 实验和结果

数据集：ASVspool 2017

结果为：![[Pasted image 20221126103920.png]]
1. 在 baseline 用 MVN 可以提高准确性
2. LCNN+FFT 实现了最佳单系统结果
3. CQT 效果都不太好，可能是 CQT特征鲁棒性较差
4. 滑动窗口的固定输入法效果比截断的方法差，原因可以是使用整句话的 spectrogram 可以实现更准确的文本相关的深度模型
5. RNN+CNN 效果还不如 单个 LCNN，可能是因为这降低了频谱分辨率

同时，低频区（0-4k）的准确度 68% 远低于高频区（4k-8k）85%。