> Interspeech 2022

1. 语言失真会降低语言处理模型的性能
2. 提出使用 domain adversarial training 来提高模型的鲁棒性
3. 基于 SUPERB，在五个不同的语音处理任务上进行实验
4. 由于不知道失真的具体类型，分析了两个域（将所有的失真看成一个域）和多个域（不同的失真看成是不同的域）的情况，获得了较好的效果

## Introduction

1. 域不匹配会降低声学模型的性能，一些常见的域不匹配包括：
	1. 口音
	2. 多语言
	3. 失真（本文关注）
2. 过去已经有一些方法用从噪声语音中恢复干净的语音，但是在未知域中不能保证性能
3. 自监督模型在目标域中进行继续训练可以提高性能
4. 另一个方法是基于 DANN 进行域对抗训练（DAT），DANN 包括特征提取、下游模型和联合训练的域分类器。域分类器用于区分目标域和源域，从而使得特征提取器的输出可以是域不变的
5. DAT 已经被用于 带口音的 ASR、多语言情感识别、TTS、说话人识别和语音增强等
6. 本文将 DAT 的概念和自监督结合，提出使用 DAT 继续训练自监督模型（无标签）来提高失真场景下的性能，效果很好，有时候还超过了在有标签的失真数据下训练的模型

## 噪声鲁棒的自监督训练

一般来说，语音中的上游任务看成是一个特征提取器 $f(s;\theta_f)$，输入波形 $s$ 输出表征 $\hat{z}$，下游任务是一个标签预测 $y(\hat{z},\theta_y)$，域不匹配发生时，测试数据的分布和训练数据的分布不同，如语音失真。

本文的任务就是使模型 $\theta_f$ 可以适用于不同的域以提高泛化性。

给定源域数据 $\mathcal{S}=\{s_1,\dots,s_N\}$ 包含 $N$ 条语音，对应的标签 $\mathbf{y}=\{y_1,\dots,y_N\}$，如果测试集 $\mathcal{C}=\{c_1,\dots,c_N\}$ 和训练集是同一个分布下的，那么效果当然很好，但是如果是不同的分布 $\mathcal{C}^{\prime}=\left\{c_1^{\prime}, \cdots, c_L^{\prime}\right\}$，此时性能可能下降。

假设有无标签的数据集 $\mathcal{T}=\left\{\left\{\hat{s}_i^1\right\},\left\{\hat{s}_i^2\right\} \cdots,\left\{\hat{s}_i^K\right\}\right\}$ 包含 $K$ 种失真，$\left\{\hat{s}_i^k\right\}$ 是失真 $k$ 的音频集合，本文考虑两种情形，$\mathcal{C}^{\prime} \in \mathcal{T} \text { and } \mathcal{C}^{\prime} \notin \mathcal{T}$（也就是测试数据是否在失真集内）。用 $\mathcal{T}$ 来更新自监督模型 $\theta_f$，使其泛化到目标域 $\mathcal{C}^{\prime}$，有两个步骤：
1. 上游预训练的时候连续训练
2. 采用 DAT 进行 fine tune

### 上游连续训练

就是把无标签的数据 $\mathcal{T}$ 加入训练集进行无监督训练.....

### DAT

引入一个域判别器（也可以说域分类器）$d(\hat{z},\theta_d)$ 用来预测给定输入语音 $s$ 的域，参数通过以下公式联合优化：$$\begin{gathered}
\theta_y \leftarrow \theta_y-\alpha \frac{\partial L_y}{\partial \theta_y}, \quad \theta_d \leftarrow \theta_d-\beta \frac{\partial L_d}{\partial \theta_d} \\
\theta_f \leftarrow \theta_f-\eta\left(\frac{\partial L_y}{\partial \theta_f}-\lambda \frac{\partial L_d}{\partial \theta_f}\right)
\end{gathered}$$
其中，$L_f,L_y,L_d$ 分别为特征提取器、标签预测器和域分类器的损失。$\alpha,\beta,$ 为学习率，特征提取器有一个负的域分类器损失项从而实现对抗训练，梯度反转缩放因子 $\lambda$ 用于缩放负损失。

对于域分类器，有两种不同的设置。

A. 两个域
数据只包含两个域，clean 的语音 和 失真的语音 $\mathcal{T}$。此时域分类器 通过 BCE 损失进行优化：$$\mathcal{L}_{B C E}=-\frac{1}{N+M} \sum_{i=1}^{N+M} d_i \cdot \log p_i+\left(1-d_i\right) \cdot \log \left(1-p_i\right)$$
其中，$p_i$ 为域分类器经过 sigmoid 之后的输出，$d_i$ 为标签预测器输出的label，$N,M$ 代表样本数。

B. 多个域
语音被分成 $K+1$ 个域，$K$ 个失真的，一个 clean 的。这时有两个目标函数：
CE 损失：$$\mathcal{L}_{C E}=-\frac{1}{N+M} \sum_{i=1}^{N+M} \sum_{k=0}^K d_i^k \log p_i^k$$
和前面的差不多，就是改成了多分类的损失。但是在 DAT 时，最大化这个损失（前面有负号）会导致域分类器的错误分类。

熵损失 Entropy loss 定义为：$$\mathcal{L}_E=-\frac{1}{N+M} \sum_{i=1}^{N+M} \sum_{k=0}^K p_i^k \log p_i^k$$
通过在 DAT 阶段最大化这个函数，可以使得域分类器输出不同类别的均匀分布，也就是确保它能够对于每个域输出相似的概率。
> 以为域分类器本质是一个多分类问题，如果没有熵损失，对于失真的域，它可能每输出的都是 $K$ 个中确定的一个，从而导致 CE loss 非常大，但是这个时候模型学习的就不是区分 clean 域和 distortion 域，而是区分不同的 distortion 域，这显然不是我们需要的。


## 实验

在五个任务中做实验：
+ 意图分类：三类，action, object, and location
+ 情感识别：9个情感类
+ 关键词识别
+ 说话人识别
+ 语音识别

### 数据配置

噪声数据 $\mathcal{T}$ 包含三种失真：
+ musan 噪声
+ 高斯噪声
+ 混响
三个域的比例是 0.3,0.4,0.3，每个语音只有一种失真，加性噪声的 SNR 从 10 到 20 dB。

测试集三种配置：
+ 原始的测试集
+ 原始的测试集+失真（和目标域一致的）
+ 原始数据集+失真（新的）

### 上游模型

采用预训练的 HuBERT 作为基本模型，首先用前面的数据集 $\mathcal{T}$ 训练了 60 epoch（3+1=4，各类占 1/4）。

### 域分类器

mean pooling 层+线性层

## 结果

具体结果见原论文。

结论：
1. continual training 大部分情况下可以提高性能
2. DAT + ce loss 大部分情况下也都优于 baseline，但是在 ASR 上的改进没有 continual training 的效果大
3. DAT + continual training 在个别情况下比直接拿有标签的数据效果还好，说明潜力很大！
4. 当在训练时不知道具体的失真类型时，二域设置是有帮助的。
