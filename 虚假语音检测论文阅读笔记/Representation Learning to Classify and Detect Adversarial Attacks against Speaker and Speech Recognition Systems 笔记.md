
1. 将表征学习用于对抗攻击的分类和检测


## Introduction
1. 最近有一些工作旨在从音频中去除对抗性噪声或者使用对抗训练来增强说话人识别网络。
2. 本文研究了如何检测和分类针对音频系统进行的对抗性攻击，在攻击分类、验证和未知攻击检测三个任务中进行实验，其中：
	1. 攻击分类的任务是确定测试话语是否真实的，或者是否属于一组已知的攻击
	2. 攻击验证是确定两个话语是否以相同的方式受到攻击
	3. 未知攻击检测任务是确定话语是否包含未知攻击


## 对抗攻击

### 攻击模型

设 $\mathbf{x} \in \mathbb{R}^T$ 是长为 $T$ 的真实语音，$y^{\text{benign}}$ 为对应的标签，通过在语音中添加一个扰动 $\delta$，攻击者可以生成对抗样本：$\mathbf{x}^{\prime}=\mathbf{x}+\boldsymbol{\delta}$，为了使得欺诈样本和原始语音尽可能相似，增强扰动的不可感知性，通常最小化某种度量 $D\left(\mathbf{x}, \mathbf{x}^{\prime}\right)<\varepsilon$，通常是扰动的 $p$ 范数。

### 攻击算法

PGD（projected gradient descent）算法的扰动通过最大化误分类误差来迭代计算：
$$\boldsymbol{\delta}_{i+1}=\mathcal{P}_{\varepsilon}\left(\boldsymbol{\delta}_i+\alpha \operatorname{sign}\left(\nabla_{\mathbf{x}_i^{\prime}} L\left(g\left(\mathbf{x}_i^{\prime}\right), y^{\text {benign }}\right)\right)\right)$$
其中，$g(\mathbf{x})$ 是说话人分类器，$L$ 为交叉熵损失。

此攻击的简化版本有 快速梯度符号法（FGSM）和 迭代FGSM（IterFGSM）。


CW（Carlini-Wagner）攻击通过寻找在确保不可感知性的同时欺骗分类器的最小扰动来计算的，即 $\delta$ 通过最小化以下损失来获得：
$$C(\boldsymbol{\delta}) \triangleq D(\mathbf{x}, \mathbf{x}+\boldsymbol{\delta})+c f(\mathbf{x}+\boldsymbol{\delta})$$
$f$ 定义为，当且仅当 $f(\mathbf{x}+\boldsymbol{\delta}) \leq 0$ 时系统失效。


## 攻击表征学习

### x-向量

 [[x-vector、i-vector]] 方法使用神经网络将每个语音话语中的身份/攻击信息编码为单个嵌入向量 [[x-vector、i-vector]]。

x-vector 网络分为三个部分：
+ encoder 从 MFCC 提取帧级别的表征
+ 时间维度的全局池化层：最终生成一个向量
+ 前馈网络计算后验概率

使用了 AAM-softmax 进行训练，不同的 x-vector 有不同的编码器架构和池化方法，本文使用的是Thin-ResNet34编码器，将从 x-vector 提取的 embedding 作为攻击嵌入。

### 攻击嵌入的应用

1. 攻击分类：将测试话语分为已知攻击类或未攻击类。我们可以根据不同的标准对攻击进行分类。如果用于训练特征提取网络的类与我们的目标类匹配，用网络的输出对测试样本进行分类，否则就训练另一个分类器。
2. 对抗验证（检测）：攻击验证的任务是确定测试话语是否包含与注册话语相同的攻击（可能存在不包括在嵌入提取网络中的攻击），使用概率线性判别分析（PLDA）来评估相同与不同攻击假设之间的对数似然比（假设检验的问题）。
3. 未知攻击检测：未知攻击检测是确定测试话语是否包含不在训练集中的攻击的任务。使用PLDA模型计算未知与已知攻击假设之间的似然比：$$\mathrm{LLR}=-\log \frac{1}{N} \sum_{i=1}^N \frac{P\left(\mathbf{x}_{\mathrm{test}}, \mathbf{X}_i \mid \text { same }\right)}{P\left(\mathbf{x}_{\mathrm{test}}, \mathbf{X}_i \mid \text { diff }\right)}$$ 其中， $x_{\text{test}}$ 为待测试的嵌入，$\mathbf{X}_i$ 为对应类别 $i$ 的已知攻击。


## 实验

1. 具体实验设置略
2. 对抗攻击样本的生成采用前面提到的 FGSM, Iter-FGSM, PGD 和 CW 等算法进行
3. x-vector 提取网络采用 Thin-ResNet34 x-Vector 中的架构，具体参数见论文
4. 其他略


## 总结

> 这篇文章其实和 欺诈检测 关系不大，主要讲的是音频的对抗攻击（类似于图像的对抗攻击，通过一些对抗手段来欺骗 ASV 系统使其判断成其他的 speaker 或者失效），而虚假语音检测是目的是给定一段语音判断他是由真人说的还是网络生成的。

