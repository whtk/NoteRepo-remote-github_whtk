> 来自论文 - Recent Advances in Open Set Recognition: A Survey

1. 开集识别：在训练的时候存在对世界的不完整只是，且测试的时候可以输入未知的类别；这要求分类器不仅可以准确分类已知的类别，而且可以有效的处理未知的类
2. 本文综述了开集识别技术，包括定义、模型、数据集、评估指标和算法比较等
3. 分析了 OSR 和其他任务之间的关系，如 zero shot、few shot 等

## Introduction

1. 基于所谓的 There are known knowns 论述，基本的识别问题可以分为以下几种：
	1. known known classes (KKCs)，有明确标记的正训练样本，甚至有 side-information
	2. known unknown classes (KUCs)，有标记的负样本，但是不必属于有意义的类
	3. unknown known classes∗2 (UKCs)，训练的时候没有可用的类样本，但是训练期间有可用的 side-information，如 语义或属性信息
	4. unknown unknown classes (UUCs)，训练时没有任何可用的类样本，而且也没有任何的 side-information
2. 