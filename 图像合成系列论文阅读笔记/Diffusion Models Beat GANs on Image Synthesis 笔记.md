> nips, 2022, openai

1. 表明，diffusion 模型可以实现图像合成质量超过 sota 生成模型
2. 对于 conditional 的图像合成，采用 classifier guidance 来提高生成质量

> 总的来说，本文的贡献就是提出了 classifier guidance 的 diffusion 模型，而且通过改进 diffusion 中的 Unet 结构表明了 diffusion 的效果可以做到很好！

## Introduction

1. GAN 的合成质量在 FID、Inception score 和 precision 上很不错，但是这些指标并不能体现多样性，GAN 生成的图像也缺乏多样性，而且 GAN 也不好 train
2. diffusion 在分辨率的图像生成上还是有困难
3. 作者认为，diffusion 和 GAN 之间的 gap 来自两点：
	1. GAN 使用的模型架构是 refine 过的，太多研究在做这个了
	2. GAN 在多样性和保真度方面进行 trade off，可以生成高质量的图片但是不能 cover 整个分布
4. 那么就可以通过：
	1. 修改 diffusion 模型架构
	2. 设计方案来更好地 trade off 多样性和保真度
	3. 从而可以实现超过 sota GAN 的效果


## 背景

diffusion 模型的介绍（略）。

采用了 IDDPM 中的混合目标函数和参数配置。

采用了 DDIM 中的 non-Markovian 的 noising process 来减少采样。

采用 Inception score、FID、Improved Precision and Recall 来评估质量。

## 改进架构

DDPM 用 UNet 架构 + skip connection + attention + timestep embedding，本文作者做了以下探索：
+ 在模型大小相对不变的情况下，增加深度和宽度
+ 增加 attention head 的数量
+ 在 32 16 8 的 feature map 中使用 attention 而不只是在 16 的
+ 采用 BigGAN 中的 residual block
+ 对 residual connection 做 $\frac{1}{\sqrt{2}}$  的缩放

上面所有的配置的模型都在 ImageNet 128 上训练，batch size 为 256，采用 250 个 sample steps：
![](image/Pasted%20image%2020230818153121.png)
可以发现，除了缩放，所有的修改都可以提升性能，而且一起用的改进更大。

还比较了不同的 head 数的影响，发现 head 数越多或者 channel 数越少都可以提升 FID，发现每个 head 有 64 个 channel 是权衡之下最好的。

### 自适应 group normalization

提出 adaptive group normalization，记为 $\operatorname{AdaGN}(h, y)=y_s \operatorname{GroupNorm}(h)+y_b$ ，其中 $s$ 为 residual block 的激活输出，$y=[y_s,y_b]$ 分别是 time step 和 class 的 embedding。

消融实验如下：
![](image/Pasted%20image%2020230818154301.png)

本文剩下的部分都使用下述配置：
+ variable width with 2 residual blocks per resolution
+ multiple heads with 64 channels per head
+ attention at 32, 16 and 8 resolutions
+ BigGAN residual blocks for up and downsampling
+ adaptive group normalization for injecting timestep and class embeddings into residual blocks.

## Classifier Guidance

前面的 自适应 group normalization 已经将类别信息引入了 normalization layer，下面探索另一种方法：利用分类器 $p(y\mid x)$ 来改进 diffusion generator。

NCSN 给出一种实现方法，将分类器的梯度作为 diffusion 模型的梯度。具体来说，基于噪声图像 $x_t$ 训练分类器 $p_\phi\left(y \mid x_t, t\right)$，然后用这个模型的梯度 $\nabla_{x_t} \log p_\phi\left(y \mid x_t, t\right)$ 来引导 diffusion 的采样过程。

下面给出利用分类器推导 conditional 采样过程的两种方法。

### Conditional Reverse Noising Process

首先，正常的 reverse noising 过程为 $p_\theta\left(x_t \mid x_{t+1}\right)$，为了让他以 label $y$ 为条件，需要根据下式进行采样：
$$p_{\theta, \phi}\left(x_t \mid x_{t+1}, y\right)=Z p_\theta\left(x_t \mid x_{t+1}\right) p_\phi\left(y \mid x_t\right)$$
其中，$Z$ 为归一化常数，但是直接从此分布中进行采样很困难，不过可以将其近似为 perturbed Gaussian distribution。

在 diffusion 中，有：
$$\begin{aligned}
p_\theta\left(x_t \mid x_{t+1}\right) & =\mathcal{N}(\mu, \Sigma) \\
\log p_\theta\left(x_t \mid x_{t+1}\right) & =-\frac{1}{2}\left(x_t-\mu\right)^T \Sigma^{-1}\left(x_t-\mu\right)+C
\end{aligned}$$
相比于 $\Sigma^{-1}$ ，我们可以假设 $p_\phi\left(y \mid x_t\right)$ 的曲率更低？

从而可以在 $x_t=\mu$ 附近进行泰勒展开：
$$\begin{aligned}
\log p_\phi\left(y \mid x_t\right) & \left.\approx \log p_\phi\left(y \mid x_t\right)\right|_{x_t=\mu}+\left.\left(x_t-\mu\right) \nabla_{x_t} \log p_\phi\left(y \mid x_t\right)\right|_{x_t=\mu} \\
& =\left(x_t-\mu\right) g+C_1
\end{aligned}$$
其中，$g=\left.\nabla_{x_t} \log p_\phi\left(y \mid x_t\right)\right|_{x_t=\mu}$ 且 $C_1$ 为常数。从而：
$$\begin{aligned}
\log \left(p_\theta\left(x_t \mid x_{t+1}\right) p_\phi\left(y \mid x_t\right)\right) & \approx-\frac{1}{2}\left(x_t-\mu\right)^T \Sigma^{-1}\left(x_t-\mu\right)+\left(x_t-\mu\right) g+C_2 \\
& =-\frac{1}{2}\left(x_t-\mu-\Sigma g\right)^T \Sigma^{-1}\left(x_t-\mu-\Sigma g\right)+\frac{1}{2} g^T \Sigma g+C_2 \\
& =-\frac{1}{2}\left(x_t-\mu-\Sigma g\right)^T \Sigma^{-1}\left(x_t-\mu-\Sigma g\right)+C_3 \\
& =\log p(z)+C_4, z \sim \mathcal{N}(\mu+\Sigma g, \Sigma)
\end{aligned}$$
其中的常数 $C_4$ 可以忽略（对应前面的 $Z$），上式的意义在于，conditional 的情况下进行采样也可以写成类似于高斯采样的形式（和 unconditional 和相似），只不过此时的均值变成了 $\mu+\Sigma g$ 。此时的采样算法为：
![](image/Pasted%20image%2020230818163532.png)

### Conditional Sampling for DDIM

上面的算法只能用于随机 diffusion 采样，而不能用在 DDIM 中。于是采用 score-based conditioning 的技巧，利用 diffusion 和 score-based 模型之间的联系。

具体来说，给定模型 $\epsilon_\theta(x_t)$ 用于预测噪声，用它来获得 score function：
$$\nabla_{x_t} \log p_\theta\left(x_t\right)=-\frac{1}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta\left(x_t\right)$$
那么：
$$\begin{aligned}
\nabla_{x_t} \log \left(p_\theta\left(x_t\right) p_\phi\left(y \mid x_t\right)\right) & =\nabla_{x_t} \log p_\theta\left(x_t\right)+\nabla_{x_t} \log p_\phi\left(y \mid x_t\right) \\
& =-\frac{1}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta\left(x_t\right)+\nabla_{x_t} \log p_\phi\left(y \mid x_t\right)
\end{aligned}$$
定义一个新的噪声预测 $\hat{\epsilon}\left(x_t\right)$ 对应上述联合分布的 score：
$$\hat{\epsilon}\left(x_t\right):=\epsilon_\theta\left(x_t\right)-\sqrt{1-\bar{\alpha}_t} \nabla_{x_t} \log p_\phi\left(y \mid x_t\right)$$

然后就可以用和 DDIM 中相同的采样步骤了，算法如下：
![](image/Pasted%20image%2020230818165119.png)

### Scaling Classifier Gradients

对于大规模的生成任务，在 ImageNet 上训练。分类器模型是 UNet 的下采样部分+ 在 8x8 的 layer 上的 attention pool 来得到最终的输出。训练完成之后，通过算法1进行采样。

但是，需要对分类器的梯度进行 scale，也就是乘于一个大于 1 的常数。如果是 1 的话，只有 50% 的概率输出类别是看上去想要的类别，而大于 1 这个概率则接近 100%。

注意到，$s \cdot \nabla_x \log p(y \mid x)=\nabla_x \log \frac{1}{Z} p(y \mid x)^s$ ，其中 $Z$ 为任意常数。当 $s>1$ 时，分布比 $p(y\mid x)$ 更 sharp，从而使得分类器能够更关注类别，也就更有潜力生成高质量的样本（但是相对的，多样性更少了）。

![](image/Pasted%20image%2020230818195249.png)

这其实就是一个 fidelity 和 diversity 的 trade off。大于 1 的因子导致 recall 和  IS 之间存在 trade off。而且这个 trade off 比在 GAN 中的效果要好。

## 结果
（略）
