> 2018 年
1. 作者证明，通过对未标记的文本语料库进行语言模型的生成式预训练，然后对特定任务进行 fine tune ，可以在这些任务中实现巨大的收益

## Introduction

1. 使用预训练的 word embedding 可以提高性能
2. 但是其挑战性在于，不知道哪种类型的目标函数是最有效的文本表征学习；也没有将学习到的表征转移到目标任务的一致性的方法
3. 现有的技术大都是针对特定任务改变模型架构，然后添上一些辅助的学习目标函数，但是不确定性太大了
4. 本文探索使用无监督预训练和有监督 fine tune 相结合的半监督方法，学习通用的表征。只需要很少的 adaption 就可以迁移到很多任务中。包括两个阶段：
	1. 使用未标记的数据训练语言模型
	2. 使用特定的有监督目标函数在特定任务中进行训练
5. 采用 Transformer 架构，在 自然语言推理、问答、语义相似性和文本分类 四类12项任务中中进行了评估

## 先前工作（略）

## 框架
> 包括两个阶段：无监督预训练和 fine tune

### 无监督预训练

给定数据集 $\mathcal{U}=\left\{u_1, \ldots, u_n\right\}$，使用标准的语言模型最大化似然：$$L_1(\mathcal{U})=\sum_i \log P\left(u_i \mid u_{i-k}, \ldots, u_{i-1} ; \Theta\right)$$
其中，$k$ 为上下文窗口的大小，条件概率 $P$ 采用神经网络来建模，其参数为 $\Theta$ ，采用 SGD 进行训练。

实际采用的是 多层的 Transformer 的 decoder 作为语言模型，计算如下：$$\begin{aligned}
h_0 & =U W_e+W_p \\
h_l & =\text { transformer\_block }\left(h_{l-1}\right) \forall i \in[1, n] \\
P(u) & =\operatorname{softmax}\left(h_n W_e^T\right)
\end{aligned}$$
其中，$U=\left(u_{-k}, \ldots, u_{-1}\right)$ 为 token 的上下文向量，$n$ 为层数，$W_e$ 为 token 的 embedding matrix，$W_p$ 为位置的 embedding matrix。

### 有监督的 fine tune

完成模型预训练后，设有标记的数据集 $\mathcal{C}$ ，每个样本 都包含输入 token 序列 $x^1,\dots,x^m$ 和标签 $y$，输入通过预训练模型得到最后一层的 Transformer block 的激活 $h^m_l$ ，然后送到一个额外的线性层（参数为 $W_y$）进行预测：$$P\left(y \mid x^1, \ldots, x^m\right)=\operatorname{softmax}\left(h_l^m W_y\right)$$
然后最大化以下目标函数：$$L_2(\mathcal{C})=\sum_{(x, y)} \log P\left(y \mid x^1, \ldots, x^m\right)$$
同时为了改进泛化能力和收敛速度，将语言建模作为额外的目标函数，总体的优化目标变成：$$L_3(\mathcal{C})=L_2(\mathcal{C})+\lambda * L_1(\mathcal{C})$$

### 特定任务时的输入转换

对于某些任务，如文本分类，可以直接使用前面的模型进行 fine tune。
但是对于其他的任务，需要结构化的输入。使用 遍历式的方法，将结构化的输入转换为预训练的模型可以处理的有序序列，如图：![[Pasted image 20230302135245.png]]

## 实验（略）