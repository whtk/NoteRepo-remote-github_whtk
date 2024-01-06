> RDAS: Raw Differentiable Architecture Search

1. 本文自动学习用于语音深度伪造和欺骗检测的网络架构，同时联合优化其他网络组件和参数
2. 产生的架构的结果是最佳的单系统

## Introduction

1. 有文表明，使用专门设计的网络和损失函数通常会导致更好的性能
2. 但是其性能上限取决于输入的特征，不同的特征可能导致性能发生显著变换
3. Rawnet2 架构使用一组滤波器，通过时域卷积直接作用于原始音频波形
4. 目前很多E2E的方法的输入特征都是手工设计提取的。
5. 作者之前的工作探索了学习网络架构的自动方法。[[Partially-Connected Differentiable Architecture Search for Deepfake and Spoofing Detection 笔记]] 使用一对称为 cell 的核心网络组件执行架构搜索。但是输入仍为 hand-crafted features
6. 在本文中，我们引入 Raw PC-DARTS，第一个E2E语音深度伪造和欺骗检测解决方案，它直接对原始波形进行操作，同时允许网络架构和网络参数的联合优化

## 相关工作

## Raw PC-DARTS
