> preprint，2022，清华、中科大、微软

1. VQ-Diffusion 有时候会生成低质量的样本或者和文本弱相关的
2. 原因可能是  flawed sampling strategy，提出两种方法来提高 VQ-Diffusion 的样本质量：
	1. 采用 classifier-free guidance sampling，且提出了一个更通用且高效的实现
	2. 提出一种高质量的推理策略来减轻 VQ-Diffusion 中存在的 联合分布问题
3. 实验表明，相比于原始的 VQ-Diffusion 有很大改进

## Introduction

1. 本文主要改进 VQ-Diffusion：
	1. Discrete classifier-free guidance：假设条件信息为 $y$，生成的图片为 $x$，diffusion 模型最大化条件概率 $p(x|y)$，假设生成的样本满足后验分布  $p(y|x)$ ，但是这种假设可能会有问题，大多数情况下模型会忽略这种假设。于是提出同时考虑先验和后验，这种方法和之前的 classifier-free technique 很相似，但是提出的方法更准确，因为模型估计的是概率而非噪声；而且除了把 condition 设为 0，引入了一个更 general 和 effective 方法
	2. High-quality inference strategy：在每个 denoising step，通常是同时采样多个 token，每个 token 都根据其概率独立采样；但是不同的位置通常是相关的，假设一个只有两个样本的数据集 AA 和 BB，每个样本 50% 概率出现，如果是独立采样，就会出现 AB、BA不正确的输出，于是引入 high-quality inference strategy，先减少每个 step 采样的 token 数，然后找到那些有着 high confidence 的更正确的 token，引入 purity prior 去采样这些 high confidence 的 token


## 背景：VQ-Diffusion

见 [Vector Quantized Diffusion Model for Text-to-Image Synthesis 笔记](Vector%20Quantized%20Diffusion%20Model%20for%20Text-to-Image%20Synthesis%20笔记.md) 。

 VQ-Diffusion 可能存在以下两个问题：
 1. 条件信息是直接注入到 $p_\theta(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t,y)$，我们希望网络根据 $\boldsymbol{x}_t,y$ 来恢复 $\boldsymbol{x}_{t-1}$，但是网络可能忽略 $y$ 因为 $\boldsymbol{x}_t$ 已经包含了足够的信息，从而导致生成的图片和 $y$ 相关性不大
 2. 对于第 $t$ 个 time step，$\boldsymbol{x}_{t-1}$ 中的每个点都从 $\boldsymbol{x}_{t}$ 中独立采样，从而无法建模不同位置之间的相关性

## 方法

### Discrete Classifier-free Guidance

因为文本合成图像任务中，生成的图像是理所当然要匹配文本条件的，VQ-Diffusion 只是简单地将这种条件信息注入到 denoising network，但是网络本身可能会忽略引入的条件而直接用带噪输入来预测原始图像，从而导致生成的图像和文本相关性不高。

从优化的角度看，diffusion 模型优化的目标是最大化 $p(x|y)$，但是这并不能提高分类概率 $p(y|x)$，于是何不一起优化 $\log p(x|y)+s\log p(y|x)$，采用 $s$ 控制权重，此时最大化此式可以写为：
$$\begin{aligned}
&\operatorname*{argmax}_x[\log p(x|y)+s\log p(y|x)] \\
&=\underset{x}{\operatorname*{argmax}}[(s+1)\log p(x|y)-s\log\frac{p(x|y)}{p(y|x)}] \\
&=\underset{x}{\operatorname*{argmax}}[(s+1)\log p(x|y)-s\log\frac{p(x|y)p(y)}{p(y|x)}] \\
&=\underset{x}{\operatorname*{argmax}}[(s+1)\log p(x|y)-s\log p(x)] \\
&=\operatorname{argmax}[\log p(x)+(s+1)(\log p(x|y)-\log p(x))]
\end{aligned}$$
> 如果只看到倒数第二行，其实和 classifier free 的方法完全一致，这里的 $s$ 就是原论文的 $w$。

在 classifier-free 中，通常使用 null 作为条件来计算 $p(x)$，但是这种固定的 text embedding 效果不太好，于是提出使用  learnable vector，然后推理的时候，使用的 denoising 方法为：
$$\log p_\theta\left(x_{t-1} \mid x_t, y\right)=\log p_\theta\left(x_{t-1} \mid x_t\right)+(s+1)\left(\log p_\theta\left(x_{t-1} \mid x_t, y\right)-\log p_\theta\left(x_{t-1} \mid x_t\right)\right)$$
虽然公式相同，但是和连续的情况下的 diffusion 还是有点不同的：
+ （没看懂）
+ 连续的 diffusion 不是直接预测概率 $p(x|y)$ 而是采用梯度来近似，而离散情况可以直接估计这个概率分布
+ 连续的 diffusion 的用的是 null，这里用的是一个可学习的 vector

### High-quality Inference Strategy

VQ-Difffusion 不同位置的 token 的采样都是独立的，提出两个方法：

第一是每个 step 都采样更少的 token，这样就主要让 diffusion 过程来学习 token 之间的相关性而非通过采样实现。

第二是，当 image token 的位置 $i$ 的 token 是 $[MASK]$ 即 $x_t^i=[MASK]$ 时，下式成立：
$$\begin{aligned}q(x_{t-1}^i=[\text{MASK}]|x_t^i=[\text{MASK}],x_0^i)=\bar{\gamma}_t-1/\bar{\gamma}_t\end{aligned}$$
这表明，每个位置都有相同的概率离开 $[MASK]$ 状态，也就是从 $[MASK]$ 状态转移到非 $[MASK]$ 状态是位置无关的，但是作者发现不同的位置有不同的  confidence，而且是 purity 越高， confidence 越高。于是可以基于 purity score 进行重要性采样而非是之前的随机采样，而位置 $t$ time step $t$ 的 purity 定义为：
$$purity(i,t)=\max_{j=1...K}p(x_0^i=j|x_t^i)$$
从而可以提升采样质量。

