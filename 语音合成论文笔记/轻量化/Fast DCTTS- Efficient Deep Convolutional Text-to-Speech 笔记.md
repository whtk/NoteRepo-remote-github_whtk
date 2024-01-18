> ICASSP 2021，Handong Global University，韩国

1.  提出 Fast DCTTS，可以在单个 CPU 线程上实时合成音频，包含了精心设计的轻量化网络
2. 还提出一种新的 group highway activation，可以在计算效率和门控机制的 regularization effect 之间权衡
3. 还引入了一个新的称为 elastic mel-cepstral distortion (EMCD) 来测量输出 mel 谱的保真度
4. 相比于 baseline，MOS 从 2.62 提高到 2.74，且只需 1.76% 计算量和 2.75% 的参数

## Introduction

1. 提出 Fast DCTTS
2. 实验发现，depthwise separable convolution 并不能提高 TTS 的速度，尽管理论上可以减少计算量
3. 同时观察到，将 DCTTS 中的 highway activation 替换为 residual connection 会降低性能

## 相关工作（略）

选择 [Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention 笔记](Efficiently%20Trainable%20Text-to-Speech%20System%20Based%20on%20Deep%20Convolutional%20Networks%20with%20Guided%20Attention%20笔记.md) 作为 baseline。

