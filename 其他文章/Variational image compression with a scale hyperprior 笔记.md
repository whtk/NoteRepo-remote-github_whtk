> Google，ICLR，2018

1. 基于 VAE 提出一个端到端的图像压缩模型
2. 采用 超先验（hyperprior）来捕获空间依赖，和 side information 有关

## Introduction

1. 有损压缩减少了传输的信息但是也引入了误差
2. 对于离散值，可以用熵编码进行压缩，其依赖于量化表征的先验概率
3. 提高压缩性能的一个方法是，传输 side information，也就是引入一些额外的比特信息


## 基于变分模型的压缩

transform coding 的方法中，encoder 讲图像 $\boldsymbol{x}$ 转为 latent representation $\boldsymbol{y}$，然后量化得到 $\hat{\boldsymbol{y}}$，此时就是一个一个的离散值，然后就可以用熵编码的方式进行编码传输。

