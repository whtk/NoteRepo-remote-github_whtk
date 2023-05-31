
> ICASSP 2021

1. 提出一种自然环境下端到端的说话人验证模型
2. 使用 thin-ResNet 提取说话人 embedding，使用 Siamese capsule network 和动态路由作为后端来计算相似性得分

## Introduction

1. 说话人验证包含：
	1. 前端：将语言转换为固定的 embedding
	2. 后端：计算两个向量的 相似度得分
2. 过去几年前端：使用 DNN 替换 i-vector/PLDA，也有使用 TDNN、RNN、CNN 等架构
3. 后端最常用的是余弦距离，很少有研究来考虑替代这个，高级的模型如 siamese network 可以用于测量两个特征之间的相似度，且已经被证明在不同的任务中是成功的
4. MLP+Siamese network 可以实现输入输出的转换，但是没有考虑特征向量内的部分和整体关系，capsule network 可以用于检测部分和整体的关系
5. 本文提出具有Siamese+capsule network后端的说话人识别模型，进行文本无关的说话人验证，贡献有：
	1. 提出新的 Siamese+capsule 网络
	2. 使用 Voxceleb1 进行端到端训练
	3. 尽管使用小的数据集，但是仍然取得了优异的结果

## 相关工作

### 用于说话人识别的孪生网络

最常用的相似性得分计算是使用余弦相似度：$$\operatorname{Score}\left(V_1, V_2\right)=\frac{V_1^T \times V_2}{\left|V_1\right|\left|V_2\right|}$$
而使用 DNN 模型来计算相似度得分反而性能不如现有的技术。

### capsule network

[[CapsuleNet]] 里面介绍了一种 Siamese Capsule Networks，使用初级 capsule 提取特征，高级的capsule 通过动态路由自主构建部分和整体的关系，然后再转换为次级潜在空间，使用这些表征的非线性组合计算相似性得分。

然后就有论文使用这个模型+BGRU进行文本分类，本文则把它引入说话人验证。

## 网络结构
![](./image/Pasted%20image%2020230130122622.png)

### 前端

使用 ResNet 作为端到端 DNN 模型的前端，具体来说是 thin-ResNet34，细节不重要，原始模型最终得到 512 维的向量，但是本文实际上移除了最后一个 FC 层。

模型对 test utterance 和 enrollment utterance 时使用相同的权重，最终得到的是两个维度为 4096 的向量。

### 后端

后端仅有高级的 capsule 组成，没用初级的 capsule。目标是在相同的 index 下比较两个 embedding，但是 初级的 capsule 有 卷积和squash操作会对这种比较造成影响，所以移除了 primary capsule。

每个 tuple $\left(v_{1_i}, v_{2_i}\right)$ ，其中 $v_{1_i}$ 表示 enrollment 语音embedding中的第 $i$ 个 index，同理下标 $2$ 表示的是 test 语音embedding。这样一对 tuple 被认为是一个 part（也就是一个 capsule），目标是提取 part 相对于 whole 的关系。

本文使用了 4 个capsule（大小为 128），数量选择是靠经验的，使用 3 次迭代，embedding 是通过了 $L2$ normalization 的。通过二分类损失进行训练。最后一层是 sigmoid 。得分是来自于 Sigmoid 计算后的相似度计算。（？？？这不还是余弦相似度吗）

### 实现

训练时，选择三个语音，先随机选两个来自同一个说话人的语音，再随机选一个来自不同说话人的语音。

使用 Adam 优化器训练，freeze 前端的参数，lr 开始为 0.01，采用 cyclical learning rate 的 scheduler，batch 为 64。

## 结果和分析

数据集：VoxCeleb1

结果：![](./image/Pasted%20image%2020230130124825.png)
效果最好，而且只用 VoxCeleb1 训练。

不同配置的结果：![](./image/Pasted%20image%2020230130125007.png)
1. 直接从GhostVlad聚合模块获得的 embedding 效果最好
2. 使用 primary capsules 会导致性能降低
