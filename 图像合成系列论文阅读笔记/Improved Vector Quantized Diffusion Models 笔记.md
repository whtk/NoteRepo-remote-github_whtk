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

略
