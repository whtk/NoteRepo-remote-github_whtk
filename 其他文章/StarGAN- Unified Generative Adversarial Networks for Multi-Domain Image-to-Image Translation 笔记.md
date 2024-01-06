1. 提出StarGAN，用于图像转换（image to image），可以在一个模型下用多种数据集训练不同域
2. 在面部特征转换、表情合成方面做的实验

## Introduction

1. domain：a set of images sharing the same attribute value，其中 attribute value 为一个 attribute 的特定值（如：头发棕色、褐色、黑色等，性别男、女等）
2. 所谓的multi-domain image-to-image translation：change images according to attributes from multiple domains，在多个domain上同时改变图像的特征
3. 要实现 multiple domain 的图像转化（$k$ 个 domain），需要训练 $k(k−1)$ 个生成器（两两之间就需要两个），而且训练的时候只能取一部分相关的图像（也就是只取这两个 domain 的数据），忽略了图像的全局特征
4. 于是提出 StarGAN：
   1. 用一个生成器（判别器自然也是一个）训练所有域之间的映射
   2. 将图像和域信息（label，可以是 binary 或 one-hot 向量，训练时随机生成）同时作为输入，根据域信息学习转换
   3. mask label 技术：忽略未知 label，专注于特定数据集的 label

>普通的方法和 StarGAN 对比：
>![](image/Pasted%20image%2020230916212214.png)

## 相关工作（略）

## StarGAN

目的是，训练 $G$，将 输入图像 $x$ 基于条件 $c$ 转换成目标图像 $y$，$G(x,c) \rightarrow y$，$c$ 是随机产生的某个域。

同时引入一个辅助分类器，使得一个判别器就可以控制多个域，判别器可以同时得到 source image 和 domain label 的分布： $D: x \rightarrow\left\{D_{s r c}(x), D_{c l s}(x)\right\}$，下标 $cls$ 代表分类器。
> 也就是既要判断图片是 real or fake，也要判断图片是来自哪个 domain。

如图：
![](image/Pasted%20image%2020230915221701.png)

对抗损失：

$$
\begin{aligned}
\mathcal{L}_{a d v}=& \mathbb{E}_{x}\left[\log D_{s r c}(x)\right]+\\
& \mathbb{E}_{x, c}\left[\log \left(1-D_{s r c}(G(x, c))\right)\right]
\end{aligned}
$$
> 这个就是 GAN 中最原始的损失。

域分类损失：
给定输入图片 $x$ 和目标域 $c$，目标是生成 $y$。引入的域分类器对应的损失可以分为两部分：真实图像的域分类损失，用于优化 $D$，虚假图片的域分类损失，用于优化 $G$。
记真实图片的域分类损失 $\mathcal{L}_{c l s}^{r}$ 和 虚假图片的域分类损失 $\mathcal{L}_{c l s}^{f}$ ，则（ $c^\prime$ 是原始图片对应的域）：
$$
\begin{align}
\mathcal{L}_{c l s}^{r}=\mathbb{E}_{x, c^{\prime}}\left[-\log D_{c l s}\left(c^{\prime} \mid x\right)\right] \\
\mathcal{L}_{c l s}^{f}=\mathbb{E}_{x, c}\left[-\log D_{c l s}(c \mid G(x, c))\right]
\end{align}
$$
优化第一项，使得 $D$ 学习区分真实图片对应的域。优化第二项，使得 $G$ 能够生成被正确分类的图片（也就是可以生成需要的 domain）。

重构损失：
优化上面几个损失并**不能确保生成的图片可以保留输入图片的内容而只是改变其 style**，于是引入重构损失，简单来说，就是把图片和目标域 $c$ 输入生成器 $G$，输出的 fake 图片再和原始域 $c^\prime$ 一起输入 $G$ ，要让前后两张图片尽可能相同：

$$
\mathcal{L}_{r e c}=\mathbb{E}_{x, c, c^{\prime}}\left[\left\|x-G\left(G(x, c), c^{\prime}\right)\right\|_{1}\right]
$$

最终，生成器和判别器的损失函数分别为：

$$
\begin{gathered}
\mathcal{L}_{D}=-\mathcal{L}_{a d v}+\lambda_{c l s} \mathcal{L}_{c l s}^{r} \\
\mathcal{L}_{G}=\mathcal{L}_{a d v}+\lambda_{c l s} \mathcal{L}_{c l s}^{f}+\lambda_{r e c} \mathcal{L}_{r e c},
\end{gathered}
$$

>上图碎碎念：
>a: 对判别器，输入为real 图片和fake 图片，输出判断图片是fake还是real，同时输出图片对应的域
>bc: 对分类器，输入为real图片和待转换的域 $c$，输出fake 图片，然后会把fake 图片和real 图片的域 $c^\prime$ 输入生成器，获得重构图片用来计算重构损失。
>d: 对判别器，根据分类器输出的fake 图片来判断图片是fake 还是real，同时会输出这种图片的域分类结果，用来进行分类损失计算

### 多个数据集下的训练（略）



## 实现