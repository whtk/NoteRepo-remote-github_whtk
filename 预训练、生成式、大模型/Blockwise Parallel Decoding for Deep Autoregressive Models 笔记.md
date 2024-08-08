> NIPS 2018，UC Berkeley，Google Brain

1. 自回归的 sequence-to-sequence 模型的生成过程是 sequential 的
2. 提出 blockwise parallel decoding，并行预测多个 time steps，然后通过 score 模型进行验证，回退到最长的 prefix，从而提高生成速度
3. 在 state-of-the-art self-attention models 上进行实验，相比于 baseline greedy decoder，迭代次数减少 2 倍，最多 7 倍

## Introduction

1. 自回归的 sequence-to-sequence 模型在推理时一个 token 一个 token 生成，很难用于实际应用
2. 一些方法用于加速自回归模型的生成，如概率密度蒸馏、子缩放、分解问题为短序列的自回归生成离散潜变量，然后并行生成
3. 本文提出利用 Transformer 的并行评分能力，训练自回归模型的变体，预测多个未来位置；测试阶段，使用这些 proposal 模型并行预测下几个位置，然后通过基础模型并行评分，确定这些预测的最长前缀，从而跳过贪婪解码循环的一次或多次迭代；
4. 相比于自回归模型的贪婪解码，可以将生成速度提高一倍，同时保持质量不变

## Greedy Decoding

在 sequence-to-sequence 问题中，给定输入序列 $x = (x_1, \ldots, x_n)$，预测对应的输出序列 $y = (y_1, \ldots, y_m)$。学习自回归 scoring model $p(y | x)$ 进行从左到右合成：
$$\log p(y\mid x)=\sum_{j=0}^{m-1}\log p(y_{j+1}\mid y_{\leq j},x).$$

推理的目标是找到 $y^* = \arg\max_y p(y | x)$。

由于输出空间很大，精确搜索不可行。一般用贪婪解码得到预测的 $\hat{y}$：
+ 从空序列 $\hat{y}$ 和 $j = 0$ 开始，重复地用最高分数的 token 扩展预测：$\hat{y}_{j+1} = \arg\max_{y_{j+1}} p(y_{j+1} | \hat{y}_{\leq j}, x)$，$j \leftarrow j + 1$，直到满足终止条件

## Blockwise Parallel Decoding

标准的贪婪解码需要 $m$ 步生成长度为 $m$ 的输出。当词表很大时，可以通过训练一组辅助模型来得到 candidate extensions。

![](image/Pasted%20image%2020240807174619.png)

假设原始模型为 $p_1 = p$，还有一组辅助模型 $p_2, \ldots, p_k$，其中 $p_i(y_{j+i} | y_{\leq j}, x)$ 是给定前 $j$ 个 token 时第 $(j + i)$ 个 token 为 $y_{j+i}$ 的概率。提出 blockwise parallel decoding 算法，保证生成相同的预测 $\hat{y}$，但只用 $m/k$ 步：
+ 预测：对于 $i = 1, \ldots, k$，得到 block 预测：$y_{j+i} = \arg\max_{y_{j+i}} p_i(y_{j+i} | \hat{y}_{\leq j}, x)$
+ 验证：找到最大的 $\hat{k}$，使得对于所有 $1 \leq i \leq \hat{k}$，$\hat{y}_{j+i} = \arg\max_{y_{j+i}} p_1(y_{j+i} | \hat{y}_{\leq j+i-1}, x)$。注意 $\hat{k} \geq 1$。
+ 接受：将 $\hat{y}$ 拓展为 $\hat{y}_{j+1}, \ldots, \hat{y}_{j+\hat{k}}$，$j \leftarrow j + \hat{k}$

在预测阶段中，找到 $p_1$ 和 $p_2, \ldots, p_k$ 的局部贪婪预测。
> 由于这些是不相交的模型，每个预测都可以并行计算，因此与单个贪婪预测相比，几乎没有时间损失。

在验证阶段，找到长度为 $k$ 的拓展的最长前缀，这个前缀由 $p_1$ 生成。如果 scoring model 可以在少于 $k$ 步内处理这个长度为 $k$ 的 token 序列，这个步骤将有助于节省时间，前提是多于一个 token 是正确的。

在接受阶段，用验证的前缀进行拓展。如果 base 模型和 proposal 模型在预测上出现分歧则会提前停止，确保结果与使用 $p_1$ 进行贪婪解码的结果相同。
> $p_1$ 是 base model，其他的是 proposal model。

提高解码的关键在于，base model $p_1$ 在验证阶段中并行进行验证。
> 虽然解码期间的操作总数与预测数量的平方成正比，但顺序操作数是常数，不受输出长度影响。从而可以并行执行验证阶段的多个位置，而不需要额外的 wall-clock 时间。

## Combined Scoring and Proposal Model

使用 Transformer 进行 scoring 时，算法需要每步两次模型调用：预测阶段并行调用 $p_1, \ldots, p_k$，验证阶段调用 $p_1$。即使模型完美，调用次数也只能从 $m$ 减少到 $2m/k$，而非 $m/k$。

如果假设 scoring 和 proposal 模型合并，可以将模型调用次数从 $2m/k$ 减少到 $m/k + 1$。

具体来说，假设有一个 Transformer 模型，在验证阶段对所有 $i = 1, \ldots, k$ 和 $i' = 1, \ldots, k$ 计算 $p_i(y_{j+i'+i} | \hat{y}_{\leq j+i'}, x)$，这可以通过增加最终投影层的维度 $k$ 倍，每个位置计算 $k$ 个独立的 softmax 实现。在预测阶段得到 $k$ 个未来预测后，调用模型得到输出。

此时，在验证阶段计算 $\hat{k}$ 后，已经计算了 $p_i(y_{j+\hat{k}+i} | y_{\leq j+\hat{k}}, x)$，这正是下一次解码迭代的预测阶段所需的。因此这两个阶段可以合并，除了第一次迭代外，模型调用次数减少一半。
> 将第 $n$ 的预测和第 $n+1$ 步的验证合并到一次计算中。

如下图：
![](image/Pasted%20image%2020240807182335.png)

即使在验证阶段需要为每个位置计算 proposal，所有预测仍然可以并行进行。

## Approximate Inference

到目前为止，block parallel decoding 产生与标准贪婪解码相同的输出。通过放宽验证阶段的标准，可以在可能偏离贪婪输出的代价下获得额外的加速。

### Top-k Selection

不要求预测与 scoring model 的预测完全匹配，而是要求在 top $k$ 项中。为此，将验证标准替换为：
$$\hat{y}_{j+i}\in\text{top-}k_{y_{j+i}}p_1(y_{j+i}\mid\hat{y}_{\leq j+i-1},x).$$

### Distance-Based Selection

在输出空间具有自然距离度量 $d$ 的问题中，可以用近似匹配替换最高分数元素：
$$d\left(\hat{y}_{j+i},\operatorname*{argmax}_{y_{j+i}}p_1(y_{j+i}\mid\hat{y}_{\leq j+i-1},x)\right)\leq\epsilon.$$

在图像生成的情况下，$d(u, v) = |u-v|$ 是给定颜色通道内强度 $u$ 和 $v$ 之间的绝对差异。

### Minimum Block Size

可能第一个非贪婪预测是错误的，此时只有一个 token 被添加到假设中。为了确保最小加速，可以要求每个解码步骤至少添加 $1 < l \leq k$ 个 token。设置 $l = k$ 对应于固定大小 $k$ 的 block 并行解码。

### 实现和训练

给定一个预训练的 base Transformer，在 decoder 输出和最终投影层之间插入一个隐藏大小为 $k \times d_{\text{hidden}}$ 和输出大小为 $k \times d_{\text{model}}$ 的 feedforward 层，其中 $d_{\text{hidden}}$ 和 $d_{\text{model}}$ 是网络中其他部分的维度。每个输出和输入之间包含一个残差连接。在每个输出上用上原始的投影层，得到 $p_1, \ldots, p_k$ 的 logits，如图：
![](image/Pasted%20image%2020240808145242.png)


由于训练时的内存限制，无法使用 $p_1, \ldots, p_k$ 对应的 k 个交叉熵损失的均值作为整体损失。而是每个 minibatch 均匀随机选择一个子损失，得到整体损失的无偏估计。在推理时，所有 logits 可以并行计算，相对于 base model 成本较小。

## 实验（略）
