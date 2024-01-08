> NIPS 2020，Seoul National University

1. Normalizing ﬂows（NF）在参数复杂度上很低效
2. 提出 NanoFlow，采用单个 neural density estimator 来建模多个 stage 下的 变换
3. 提出了 一种有效的减参方法 和 ﬂow indication embedding 的概念

## Introduction

1. flow 模型存在一个 question：到底是 NF 需要很大的网络来实现 expressive bijection，还是神经网络的表征能力被低效利用？
	1. 作者认为应该考虑模型参数复杂度，模型的 expressiveness 不应该随着参数线性增长
2. 提出 NanoFlow，
