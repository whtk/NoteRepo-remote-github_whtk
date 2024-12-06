> ICLR 2022，Apple 公司，https://github.com/apple/ml-cvnets

1. 本文将 CNN 和 ViT 结合起来构建一个用于 mobile vision task 的轻量化、低延迟的网络
2. 提出 MobileViT，用于移动设备的轻量化的通用 ViT

## Introduction

1. 现有的将卷积+transformer 混合起来的方法仍然 heavy-weight，且对数据增强很敏感
2. 优化 FLOPs 并不足以实现 low latency
3. 本文观关注于设计轻量化、通用、低延迟的网络，将 CNN 和 ViT 结合起来，引入 MobileViT 模块在一个 tensor 中编码 local 和 global 信息
4. 在 5-6M 的参数下，可以在 ImageNet-1k 数据集中实现 top-1 的性能

## 相关工作（略）

## 