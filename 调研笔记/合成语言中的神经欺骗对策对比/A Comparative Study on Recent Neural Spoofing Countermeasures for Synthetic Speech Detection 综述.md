
1. 本文对 2019年ASVspoof LA任务的检测对策进行比较，主要考虑了基于边缘的训练判别、广泛使用的前端提取、和一些用于处理可变长度语言的通用策略
2. 同时还通过实验表明了，仅改变随机初始化种子，同一模型的性能可能都会显著不同
3. 一些好的技术，如处理不同长度输入的平均池化、无参数损失函数等可以产生性能最佳的单系统

> 这篇综述主要考虑后端分类器。

## Introduction
基于神经网络的CM是语言反欺诈的热点。很多研究从图像迁移而来，但是图像的长度是固定的，语音的长度则是可变的。

本研究的目的：
1. 总结和比较了近年来语音反欺骗文献中报道的处理可变长度输入和一些损失函数的策略。
2. 引入了一个简单的超无参数损失函数。

作者发现，同一个模型多跑几次的效果差异很大，这表明未来CM模型需要进行统计分析或者多次运行来确定模型效果。

同时，结果表明，CM 可以通过注意力或者平均池化来处理不同长度的语音。同时简单的 sigmoid 函数就已经有良好的性能了。基于P2SGrad的新型损失函数大有可为，通过和 LCNN+平均池化进行组合可以获得最低的 EER。


## 基于神经网络的 CM 简述
用 $\boldsymbol{x}_{1: N^{(j)}} \equiv\left(\boldsymbol{x}_1, \boldsymbol{x}_2, \cdots, \boldsymbol{x}_{N^{(j)}}\right) \in \mathbb{R}^{N^{(j)} \times D}$  来表示输入的第 $j$ 个样本的声学特征，其中 $\boldsymbol{x}_n \in \mathbb{R}^D$ 是第 $n$ 帧的特征向量，$N^{j}$ 代表第 $j$ 个样本的总帧数，CM 系统（或模型）将 $\boldsymbol{x}_{1: N}(j)$ 转换成一个标量得分 $s_j \in \mathbb{R}$ 原来表明该样本的真实程度。

### 基于神经网络的后端分类
最大的问题就是，样本长度 $N^{j}$ 是可变的，有三种办法来处理，如下图：
![[Pasted image 20221018160201.png]]
1. 从固定长度计算得分：
	1. 通过pad 或者 trim 将输入变成 $\tilde{\boldsymbol{x}}_{1: K} \in \mathbb{R}^{K \times D}$ ，在使用 CNN   转到 $\boldsymbol{h}_{1: K / L}$，再进行 Flatten 最终得到标量得分值。
	2. 或者将输入分成多个 chunk，每个chunk 的长度固定，计算每个chunk的得分然后求平均。
2. 从可变长度计算得分：
	1. 固定长度的副作用是，trim 会丢失信息，pad 会带来一些人工痕迹。因此对于可变长度的 输入，流程为：$f: \mathbb{R}^{N^{(j)} \times D} \rightarrow \mathbb{R}^{N^{(j)} / L \times D_h} \rightarrow \mathbb{R}^{D_h} \rightarrow \mathbb{R}$，第一步将  $\boldsymbol{x}_{1: N^{(j)}}$ 转成 $\boldsymbol{h}_{1: N^{(j)} / L}$，当 $L$ 大于1时，用 CNN，当 $L=1$ 时，用RNN。 $\boldsymbol{h}_{1: N^{(j)} / L}$ 可以pooling 成 utterance-level 的向量 $\boldsymbol{o}_j=\sum_{m=1}^{N^{(j)} / L} w_m \boldsymbol{h}_m$（其实就是加权求和），最后从 $\boldsymbol{o}_j \in \mathbb{R}^{D_h}$ 得到最终的得分。


### 损失函数

1. 原始的交叉熵和 margin-based softmax函数：CM 模型可以通过最小化交叉熵进行训练，对于训练集的标签 $y_j \in\{1, \cdots, C\}$，数据集 $\mathcal{D}$ 上的损失函数定义为：
$$\mathcal{L}^{(\mathrm{ce})}=-\frac{1}{|\mathcal{D}|} \sum_{j=1}^{|\mathcal{D}|} \sum_{k=1}^C \mathbb{1}\left(y_j=k\right) \log P_{j, k}$$
	其中，$P_{j,k}$ 表示第 $j$ 个样本来自于第 $k$ 个类别的概率，他可以通过 softmax 函数进行计算：
$$P_{j,k}=\frac{\exp \left(\boldsymbol{c}_k^{\top} \boldsymbol{o}_j\right)}{\sum_{i=1}^C \exp \left(\boldsymbol{c}_i^{\top} \boldsymbol{o}_j\right)}$$
	当 $C=2$ 时，$P_{j, 1}=\frac{1}{\left.1+\exp \left(-\left(\boldsymbol{c}_1-\boldsymbol{c}_2\right)^{\top} \boldsymbol{o}_j\right)\right)}$，这被广泛应用于CM系统中用来判断语音的真假。
	而 margin-based softmax 函数的 $P_{j,k}$ 计算如下：
$$P_{j, k}=\frac{e^{\alpha\left[\cos \left(m_1 \theta_{j, k}+m_2\right)-m_3\right]}}{e^{\alpha\left[\cos \left(m_1 \theta_{j, k}+m_2\right)-m_3\right]}+\sum_{i=1, i \neq k}^C e^{\alpha \cos \theta_{j, i}}},$$
	其中，$\cos \theta_{j, k}=\widehat{\boldsymbol{c}}_k^{\top} \widehat{\boldsymbol{o}}_j$ 为归一化向量 $\widehat{\boldsymbol{c}}_k=\boldsymbol{c}_k /\left\|\boldsymbol{c}_k\right\| \text { and } \widehat{\boldsymbol{o}}_j=\boldsymbol{o}_j /\left\|\boldsymbol{c}_j\right\|$ 之间的余弦距离。$\left(\alpha, m_1, m_2, m_3\right)$ 为超参数，以来定义不同的margin，如 angular-softmax、additive-margin 等。最终的得分可以是 $s=P_{j, 1}$ 或者$s=\widehat{\boldsymbol{c}}_1^{\top} \widehat{\boldsymbol{o}}_j$ （前提是 $y_j=1$ 表示的是真实语音）。

2. 基于P2SGrad的新型MSE损失：margin-based softmax 函数对参数敏感，下面给出了一个没有超参数的损失函数：
$$\mathcal{L}^{(\mathrm{p} 2 \mathrm{~s})}=\frac{1}{|\mathcal{D}|} \sum_{j=1}^{|\mathcal{D}|} \sum_{k=1}^C\left(\cos \theta_{j, k}-\mathbb{1}\left(y_j=k\right)\right)^2$$
	其中，$\cos \theta_{j, k}=\widehat{\boldsymbol{c}}_k^{\top} \widehat{\boldsymbol{o}}_j$，损失是网络的输出和一个标量的 target value 之间的 MSE，只不过这里的网络输出的是一个 cos 距离。之所以叫 P2SGrad 是因为梯度 $\partial \mathcal{L}^{(\mathrm{p} 2 \mathrm{~s})} / \partial \boldsymbol{o}_j$ 和 probability-to-similarity gradient 相等，在推理阶段，通过 $k=1$ 表示真实语音，可以将 $s=\cos \theta_{j, 1}$ 设为输出分数。

