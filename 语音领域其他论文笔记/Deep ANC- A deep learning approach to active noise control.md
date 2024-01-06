> neural network 2021，Ohio State University

1. 传统的 ANC 基于自适应的信号处理（用 LMS 算法），但是这些方法属于线性系统，从而不适用于非线性的失真
2. 提出 deep ANC，采用深度学习方法来编码最优的控制参数
3. 采用 卷积循环网络来从参考信号中估计 消除信号的 spectrogram 的实部和虚部

## Introduction

1. ANC 需要预测任何时刻的信号的幅度和相位
2.  常见的两种 ANC：
	1. 前馈
	2. 反馈
3. 