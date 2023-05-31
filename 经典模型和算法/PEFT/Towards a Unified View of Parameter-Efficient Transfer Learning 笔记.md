> CMU，2022，ICLR

1. 分解了一些 SOTA 的参数高效迁移学习方法，提出了一个统一的框架以建立其之间的联系
2. 将这些方法看成是对预训练模型中的特定的 **隐藏状态** 的修改
3. 框架能够在不同的方法之间传递设计方法，从而可以实现新的方法

## Introduction

1. 已经提出了很多的参数高效 fine tune 方法，可以在没有灾难性遗忘的情况下快速适应新任务，并且在 OOD 中鲁棒性很强
2. 但是对这些方法成功的原理知之甚少，之间的联系也不清楚，本文回答一下三个问题：
	1. 方法之间的联系是什么
	2. 方法之间是否存在一些共同的设计要点，这些要点是什么
	3. 方法之间的设计要点是否可以迁移到其他的方法中实现更好的变体
3. 不过实验首先表明，现有的参数高效微调方法在更高资源和具有挑战性的任务上仍然落后于完全微调

## 预备知识

Transformer 一共包含 $L$ 层，其中的一层如图：![](image/Pasted%20image%2020230509153557.png)
传统的 attention 计算如下：$$\operatorname{Attn}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V})=\operatorname{softmax}\left(\frac{\boldsymbol{Q} \boldsymbol{K}^T}{\sqrt{d_k}}\right) \boldsymbol{V}$$
其中，$\boldsymbol{Q} \in \mathbb{R}^{n \times d_k},\boldsymbol{K} \in \mathbb{R}^{m \times d_k},\boldsymbol{V} \in \mathbb{R}^{m \times d_v}$，MHA 在 $N_h$ 个 head 上并行计算 attention，每个 head 都被权重矩阵 $\boldsymbol{W}_q^{(i)}, \boldsymbol{W}_k^{(i)}, \boldsymbol{W}_v^{(i)} \in \mathbb{R}^{d \times d_h}$ 投影到对于的 QKV 中，给定 $m$ 个 vector 的序列 $\boldsymbol{C} \in \mathbb{R}^{m \times d}$ 和 query 向量 $\boldsymbol{x} \in \mathbb{R}^{d}$，MHA 计算如下：$$\operatorname{MHA}(\boldsymbol{C}, \boldsymbol{x})=\operatorname{Concat}\left(\operatorname{head}_1, \cdots, \operatorname{head}_{\mathrm{h}}\right) \boldsymbol{W}_{{o}}, \operatorname{head}_{\mathrm{i}}=\operatorname{Attn}\left(\boldsymbol{x} \boldsymbol{W}_q^{(i)}, \boldsymbol{C} \boldsymbol{W}_k^{(i)}, \boldsymbol{C} \boldsymbol{W}_v^{(i)}\right)$$
其中，$\boldsymbol{W}_{{o}}\in\mathbb{R}^{d\times d}$，$d$ 为模型的维度，$d_h=d/H_h$。

而针对于全连接层，计算如下：$$\operatorname{FFN}(\boldsymbol{x})=\operatorname{ReLU}\left(\boldsymbol{x} \boldsymbol{W}_1+\boldsymbol{b}_1\right) \boldsymbol{W}_2+\boldsymbol{b}_2$$
其中，$\boldsymbol{W}_1\in\mathbb{R}^{d\times d_m},\boldsymbol{W}_2\in\mathbb{R}^{d_m\times d}$，通常，在 Transformer 中，$d_m=4d$。

从数学角度看一些现有的 PEFT 方法：

Adapter 通常采用下投影矩阵 $\boldsymbol{W}_{down}\in\mathbb{R}^{d\times r}$ 将输入 $\boldsymbol{h}$ 投影到 bottleneck 维度 $r$，然后接一个非线性激活，然后是上投影 $\boldsymbol{W}_{up}\in\mathbb{R}^{r\times d}$ ，最后还有一个残差连接：$$\boldsymbol{h} \leftarrow \boldsymbol{h}+f\left(\boldsymbol{h} \boldsymbol{W}_{\text {down }}\right) \boldsymbol{W}_{\text {up }}$$
Prefix Tuning 在 MHA 的 keys 和 values 之前放置 $l$ 个 prefix vector 组成新的  keys 和 values ，此时 attention head 的计算公式变为：$$\operatorname{head}_i=\operatorname{Attn}\left(\boldsymbol{x} \boldsymbol{W}_q^{(i)}, \operatorname{concat}\left(\boldsymbol{P}_k^{(i)}, \boldsymbol{C} \boldsymbol{W}_k^{(i)}\right), \operatorname{concat}\left(\boldsymbol{P}_v^{(i)}, \boldsymbol{C} \boldsymbol{W}_v^{(i)}\right)\right)$$

Prompt Tuning 简化了 Prefix Tuning，只在输入的 word embedding 上添加，类似于之前的 P-tuning。

LoRA 在 Transformer 层中添加一个可以训练的低秩矩阵。对于预训练的权重矩阵 $\boldsymbol{W}\in\mathbb{R}^{d\times k}$ ，采用低秩分解进行权重更新 $\boldsymbol{W}+\Delta W=\boldsymbol{W}+\boldsymbol{W}_{\text {down }} \boldsymbol{W}_{\text {up }}$，其中 $\boldsymbol{W}_{\text {down }} \in \mathbb{R}^{d \times r}, \boldsymbol{W}_{\text {up }} \in \mathbb{R}^{r \times k}$ 是可训练的参数，LoRA 把这种更新应用到 query 和 value 的投影矩阵 $\boldsymbol{W}_q, \boldsymbol{W}_v$ 上，然后更新如下：$$\boldsymbol{h} \leftarrow \boldsymbol{h}+s \cdot \boldsymbol{x} \boldsymbol{W}_{\text {down }} \boldsymbol{W}_{\text {up }}$$
其中，$s\ge1$ 为可训练的缩放因子。

## 统一视角

首先推到了一种 prefix tuning 的等价形式，使其和 adapter 建立联系，然后提出了一种统一的框架。

### Prefix Tuning

Prefix Tuning 中，通过在原始的 attention 的 key 和 value 的前面添加 $l$ 个可学习的向量来修改注意力模块：$$\begin{aligned}
& \text { head }=\operatorname{Attn}\left(\boldsymbol{x} \boldsymbol{W}_q, \operatorname{concat}\left(\boldsymbol{P}_k, \boldsymbol{C} \boldsymbol{W}_k\right), \operatorname{concat}\left(\boldsymbol{P}_v, \boldsymbol{C} \boldsymbol{W}_v\right)\right) \\
& =\operatorname{softmax}\left(\boldsymbol{x} \boldsymbol{W}_q \operatorname{concat}\left(\boldsymbol{P}_k, \boldsymbol{C} \boldsymbol{W}_k\right)^{\top}\right)\left[\begin{array}{c}
\boldsymbol{P}_v \\
\boldsymbol{C} \boldsymbol{W}_v
\end{array}\right] \\
& =(1-\lambda(\boldsymbol{x})) \operatorname{softmax}\left(\boldsymbol{x} \boldsymbol{W}_q \boldsymbol{W}_k^{\top} \boldsymbol{C}^{\top}\right) \boldsymbol{C} \boldsymbol{W}_v+\lambda(\boldsymbol{x}) \operatorname{softmax}\left(x \boldsymbol{W}_q \boldsymbol{P}_k^{\top}\right) \boldsymbol{P}_v \\
& =(1-\lambda(\boldsymbol{x})) \underbrace{\operatorname{Attn}\left(\boldsymbol{x} \boldsymbol{W}_q, \boldsymbol{C} \boldsymbol{W}_k, \boldsymbol{C} \boldsymbol{W}_v\right)}_{\text {standard attention }}+\lambda(\boldsymbol{x}) \underbrace{\operatorname{Attn}\left(\boldsymbol{x} \boldsymbol{W}_q, \boldsymbol{P}_k, \boldsymbol{P}_v\right)}_{\text {independent of } \boldsymbol{C}},
\end{aligned}$$
其中，$\lambda(\boldsymbol{x})=\frac{\sum_i \exp \left(\boldsymbol{x} \boldsymbol{W}_q \boldsymbol{P}_k^{\top}\right)_i}{\sum_i \exp \left(\boldsymbol{x} \boldsymbol{W}_q \boldsymbol{P}_k^{\top}\right)_i+\sum_j \exp \left(\boldsymbol{x} \boldsymbol{W}_q \boldsymbol{W}_k^{\top} \boldsymbol{C}^{\top}\right)_j}$ 是标量缩放因子，表示在 prefix 上归一化的注意力权重。

上式中，第一项和 prefix 无关，第二项仅和 prefix 有关，进一步写成：$$\boldsymbol{h} \leftarrow(1-\lambda(\boldsymbol{x})) \boldsymbol{h}+\lambda(\boldsymbol{x}) \Delta \boldsymbol{h}, \quad \Delta \boldsymbol{h}:=\operatorname{softmax}\left(\boldsymbol{x} \boldsymbol{W}_q \boldsymbol{P}_k^{\top}\right) \boldsymbol{P}_v$$
定义 $\boldsymbol{W}_1=\boldsymbol{W}_q \boldsymbol{P}_k^{\top}, \boldsymbol{W}_2=\boldsymbol{P}_v, f=\text { softmax }$，则：$$\boldsymbol{h} \leftarrow(1-\lambda(\boldsymbol{x})) \boldsymbol{h}+\lambda(\boldsymbol{x}) f\left(\boldsymbol{x} \boldsymbol{W}_1\right) \boldsymbol{W}_2$$
这个公式和 adapter 非常相似，区别在于 prefix tuning 是做了加权的。

### 统一框架

所以的方法都可以看成是对改变量 $\Delta\boldsymbol{h}$ 的一种学习，记要被修改的原始的隐表征为 $\boldsymbol{h}$ ，计算 $\boldsymbol{h}$ 对应的子模块的输入记为 $\boldsymbol{x}$（例如，两者可以分别为 attention的输出和输入），然后三个方法就可以统一写为：![](image/Pasted%20image%2020230509214245.png)
> 这里的 head attention 表示 multi-head attention 中的每个 head 都有一个 attention 的操作，表明每个 head 的输出表征都会更新，而且每个 head 都有 prefix

表中，Functional Form 表示计算 $\Delta\boldsymbol{h}$ 的函数形式。Modified Representation 表明修改的是模型的哪一部分。Insertion Form 表明模型插入模型的方式，如图：![](image/Pasted%20image%2020230509214730.png)
+ adapter 是顺序方式插入，输入输出都是 $\boldsymbol{h}$
+ prefix tuning 和 LoRA 相当于是并行插入，其输入是 $\boldsymbol{x}$。
最后，Composition Function 表明修改后的 $\Delta\boldsymbol{h}$ 和 原始的 $\boldsymbol{h}$ 是如何组合的：
+ adapter 是执行简单的 additive
+ prefix tuning 用的是 gated additive
+ LoRA 通过一个常数因子缩放变化量之后加到原来的表征中

其实，prompt tuning 类似于 prefix tuning，但是prompt 只修改第一层中的 head attention，而对于不同的 adapter 的变体则类似于 adapter 中的表示方法。

通过这样一种统一的表征，我们可以沿着这些设计维度确定哪些是关键设计因子，从而在不同的方法中传递不同的因子。

### 变体

例如：
+ Parallel Adapter 通过将 prefix tuning 的并行思想引入到 Adapter 中，神奇的是已经有研究是关于这个了
+ Multi-head Parallel Adapter 又进一步，像 prefix tuning 一样，采用 parallel adapters 修改 head attention 的输出
+ Scaled Parallel Adapter 将LoRA 中的 composition 和 insertion form 引入 Adapter

## 实验和结果

四个任务：
+ 摘要
+ 翻译
+ 推理
+ 情感分类
结果：![](image/Pasted%20image%2020230509221020.png)
不同 PEFT 方法的对比，左边图是摘要和翻译任务，右边是推理和分类任务。

不同插入格式的对比：![](image/Pasted%20image%2020230509221255.png)
并行效果优于顺序。

修改不同表征：![](image/Pasted%20image%2020230509221509.png)
修改 FFN 层效果优于修改 attention 模块。

不同变体的结果对比：![](image/Pasted%20image%2020230509221801.png)
MAM Adapter 效果最好。

这里的 MAM Adapter 是作者提出的缝合怪，采用 Scaled parallel adapter 修改 FFN，同时像 prefix tuning 一样修改 head attention，得到所谓的  Mix-And-Match adapter (MAM Adapter)。