> interspeech 2019，Numediart Institute, University of Mons

1. TTS 中用于控制风格的参数通常包含 latent variables，且很难解释
2. 本文分析和比较了不同的 latent spaces，且对其表达性做出解释

## Introduction

1. 语音中 variability 的控制的一个问题就是缺乏 label，其中一个解决方案是用 categorical 的表征（例如对情感分6类），缺点在于无法提供一个连续的表征；也有论文在 multidimensional 的连续空间进行表征
2. 之前的工作都没有讲清楚 latent space 和 要控制的 audio characteristics 之间的关系
3. 本文采用特征选择技术比较不同的 embedding，然后学习了每个 latent space 和 audio features 之间的关系，具体来说，在三个不同的任务中比较了三个 latent spaces：
	1. 风格分类
	2. 说话人分类
	3. 基于 VAE 的 TTS

## 数据

说话人被要求以不同的意愿表述不同的风格：
+ 中立
+ 高兴
+ 伤心
+ bad guy
+ rom afar
+ proxy
+ old man
+ little creature

具体信息如下：
![](image/Pasted%20image%2020231021224607.png)

## embedding 计算

分别用三个任务  Style Classification, Speaker Classification and VAE-TTS 来产生 embedding，如图：
![](image/Pasted%20image%2020231021224702.png)
只有第一个系统训练的时候用了 style label，但是可以发现后面两个任务的点也可以分开。

模型细节略。

## latent space 分析

### 风格分类得分

显然风格分类模型得到的 embedding 会有更高的 分类得分，把其他两个任务的 embedding 也拿来做风格分类，计算其互信息，如图：
![](image/Pasted%20image%2020231021225424.png)

### embedding space 和 audio feature 之间的关系

分析了 embedding space 和 eGeMAPS feature set 之间的关系，过程如下：
+ 采用最小二乘线性回归 每个 latent space 和 audio feature space 之间的一个线性函数（即超平面）
+ 基于 latent embeddings 计算  audio feature 的线性函数近似
+ 目标是分析近似的估计和 ground truth 之间的相似性
+ 于是计算 Absolute Pearson Correlation Coefficient (APCC) 

其实就是计算 audio feature 和每个超平面之间的 APCC 值，结果如下：
![](image/Pasted%20image%2020231022095158.png)
> 这是不是意味着 F0 和韵律的关系最大？


### latent spaces 降维

降维后的 APCC 均值如下：
![](image/Pasted%20image%2020231022095354.png)

降维之后的数据分布和梯度方向如下：
![](image/Pasted%20image%2020231022095441.png)