
> Neurocomputing 2019

1. 提出具有全局和局部特征的 Siamese capsule network
2. Siamese 网络用于收集关于类别之间的全局语义差异的信息，这可以更准确地表示不同类别之间的语义距离。建立了一种全局记忆机制来存储全局语义特征，然后将其纳入文本分类模型
3. capsule vector 获得局部特征的空间位置关系，从而提高特征的表示能力

## 模型

提出的模型如图：![](./image/Pasted%20image%2020230130191712.png)
可以分为两个部分：
+ 基本的 capsule 网络，用于提取文本的局部特征
+ Siamese 网络用于学习类别之间的差异信息
+ global memory mechanism 用于存储全局特征

### 公式化

设数据集包含 $c$ 个类别的文档，capsule network 的输入是文档中的某段话，每个类别的文档 $D=\left\{T_1, T_2, \ldots, T_n\right\}$ 都包含 $n$ 条句子，每个句子 $T_d=\left\{i_1, i_2, \ldots, i_k\right\}$ 有 $k$ 个单词，且每个词都是 $m$ 维的词向量。每个 capsule 为 $8$ 或 $16$ 维的向量，$\omega$ 表示使用局部特征进行分类的 capsule vector 的比例的分类权重，模型输入一个句子，输出分类概率。

分三步训练：
1. 随机初始化两个 capsule network 来构成 Siamese network，学习类别之间的差异信息
2. $\omega=1$，以监督学习方式获得局部特征，将获得的局部特征存储在 global memory mechanism 中
3. 对 global memory mechanism 的所有特征进行 SVD，获得每个类的中心 capsule，即每个类的全局特征

测试时：
1. $\omega=0.5$，测试样本输入 capsule network 获得分类概率
2. 测试样本输入另一个共享权重的 capsule network，得到一组capsule，然后进行线性变换得到和全局 capsule 大小相同的 capsule，计算两者之间的距离得到分类概率
3. 概率相加得到最终的分类概率（因为 $\omega=0.5$）

### 基本 capsule network 的原理

### Siamese capsule networks

### Global memory mechanism