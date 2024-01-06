> DeepMind，2021，NIPS

1. 使用  VQ-VAE 进行 large scale 的图像生成
2. 采用 feed-forward encoder 和 decoder 网络
3. 表明，multi-scale hierarchical organization of VQ-VAE 可以生成高质量的样本

## Introduction（略）

## 背景（略）

## 方法

![](image/Pasted%20image%2020230925152228.png)

![](image/Pasted%20image%2020230925152254.png)

分为两个步骤：
+ 训练一个 hierarchical VQ-VAE 把图像编码到离散的 latent space
+ 拟合 PixelCNN 先验

### 步骤1：学习 Hierarchical Latent Codes

目的是把 local information 和 global information 分开建模。

top latent code 建模 global information，bottom latent code 以 top latent 为条件，用于表征 local details。

对于 256x256 的图片，有两个 level。encoder 先将图片转换到 64x64 的表征，然后量化得到 bottom level latent map。另一个 residual blocks 继续将表征转化为 32x32，然后量化得到 top level latent map。decoder 是一个 feed-forward network，输入为所有 level 的量化表征，通过上采样得到原始图像大小。

### 步骤2：学习 Latent Codes 的先验

（略）