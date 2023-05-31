> 论文 - BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models，Bar Ilan University，2022

1. 提出 BitFit，一种稀疏的  fine tune 方法，只修改模型的 bias 项
2. 对于中小型数据，BitFit+BERT 可以和 fine tune BERT 相竞争
3. 对于大型数据，可以和其他的稀疏 fine tune 方法相竞争

## Introduction

1. fine tune 成本很高
2. 提出一种简答的 fine tune 方法，优点有：
	1. 每人任务只需要 fine tune 很少的参数
	2. 每个任务改变的是相同的参数集
	3. 被改变的参数中是独立的和在整个参数空间的
	4. 对于中小型训练数据，仅更改这些参数就可以达到与完全微调相同的任务精度，有时甚至可以更好
3. 具体来说，就是 freeze 网络的大部分参数 而仅 fine tune bias term
4. 允许模型性能略微下降的条件下，可以只 fine tune query 和 middle-of-MLP 这两项

## 背景：fine tune 和 参数高效的 fine tune

fine tune 大模型在下游任务的性能取决于这样一个权衡：fine tune 诱导模型学习新知识的能力 和 模型在预训练过程中已经学习到的能力之间的权衡

现有的 方法 包括 Adapter 和 Diff-Pruning。

## BitFit

只训练 bias 项和 任务特定的分类层，包含三个关键属性：
+ 匹配 full fine tune 的结果
+ 允许任务流式到达，而访问所有的数据集
+ 只要 fine tune 模型参数的一小部分

具体来说，BERT encoder 包含 $L$ 层，每一层 $l$ 都有 $M$ 个 attention heads，每个 attention head $(m,l)$ 包含 key，query，value encoder，计算如下：$$\begin{aligned}
\mathbf{Q}^{m, \ell}(\mathbf{x}) & =\mathbf{W}_q^{m, \ell} \mathbf{x}+\mathbf{b}_q^{m, \ell} \\
\mathbf{K}^{m, \ell}(\mathbf{x}) & =\mathbf{W}_k^{m, \ell} \mathbf{x}+\mathbf{b}_k^{m, \ell} \\
\mathbf{V}^{m, \ell}(\mathbf{x}) & =\mathbf{W}_v^{m, \ell} \mathbf{x}+\mathbf{b}_v^{m, \ell}
\end{aligned}$$
其中，$\mathbf{x}$ 为前一层的输出，然后通过注意力机制计算得到：$$\mathbf{h}_1^{\ell}=\operatorname{att}\left(\mathbf{Q}^{1, \ell}, \mathbf{K}^{1, \ell}, \mathbf{V}^{1, \ell}, \ldots, \mathbf{Q}^{m, \ell}, \mathbf{K}^{m, \ell}, \mathbf{V}^{m, l}\right)$$
最后通过 MLP 和 Layer-Norm：$$\begin{aligned}
\mathbf{h}_2^{\ell} & =\operatorname{Dropout}\left(\mathbf{W}_{m_1}^{\ell} \cdot \mathbf{h}_1^{\ell}+\mathbf{b}_{m_1}^{\ell}\right) \\
\mathbf{h}_3^{\ell} & =\mathbf{g}_{L N_1}^{\ell} \odot \frac{\left(\mathbf{h}_2^{\ell}+\mathbf{x}\right)-\mu}{\sigma}+\mathbf{b}_{L N_1}^{\ell} \\
\mathbf{h}_4^{\ell} & =\operatorname{GELU}\left(\mathbf{W}_{m_2}^{\ell} \cdot \mathbf{h}_3^{\ell}+\mathbf{b}_{m_2}^{\ell}\right) \\
\mathbf{h}_5^{\ell} & =\operatorname{Dropout}\left(\mathbf{W}_{m_3}^{\ell} \cdot \mathbf{h}_4^{\ell}+\mathbf{b}_{m_3}^{\ell}\right) \\
\text { out }^{\ell} & =\mathbf{g}_{L N_2}^{\ell} \odot \frac{\left(\mathbf{h}_5^{\ell}+\mathbf{h}_3^{\ell}\right)-\mu}{\sigma}+\mathbf{b}_{L N_2}^{\ell}
\end{aligned}$$
上面公式中，所有的向量 $\mathbf{b}_{(\cdot)}^{\ell,(\cdot)}$ 都称为 bias 项。

由于 bias 项是加性的，对应于网络很小的一部分参数，对于 BERT-base，大概 0.09%，对于 BERT-large，大概 0.08%。

## 实验和结果

在 GLUE benchmark 上进行实验，采用预训练的 BERT-base，BERT-large 和 RoBERTa-base 模型，HuggingFace 实现。

和其他方法对比：![](image/Pasted%20image%2020230508111131.png)

在不同模型上的效果：![](image/Pasted%20image%2020230508111159.png)

只 fine tune bias 的一部分的结果：![](image/Pasted%20image%2020230508111229.png)
有一些层的 bias 项影响更大。

还有一些其他的优点：
1. 泛化差距更小（泛化性更强）
2. BitFit 达到 full fine tune 的能力 与 训练数据成反相关

