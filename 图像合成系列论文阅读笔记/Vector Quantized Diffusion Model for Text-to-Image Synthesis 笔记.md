> CVPR 2022，中科大、微软

1. 提出 VQ-Diffusion，基于 VQ-VAE，但是 latent space 是用 diffusion model 建模的
2. 发现在 text-to-image 上的效果特别好，而且生成效率也很高


## Introduction

1. 自回归模型如 DALL-E 可以实现非常好的 text-to-image 生成
2. 但是会存在一些问题：
	1. unidirectional bias，现有的方式都是以 reading order 来预测 pixel 或 token，这种固定的顺序就会引入 bias，因为上下文信息不仅仅来自于之前的 token
	2. 预测误差累计，推理时每个 step 都是基于前一个 token 的，但是训练时却不是这样的（也就是所谓的 teacher-forcing ），从而可能导致误差传播
3. 提出 VQ-Diffusion，采用 DDPM 的变体来建模 latent space，forward 过程逐渐在 latent variables 上加噪
4. VQ-Diffusion 可以减轻 unidirectional bias，其包含一个 独立的 text encoder 和一个 diffusion image decoder，推理时，所有的 image token 要么 mask 要么随机，然后 diffusion model 根据输入文本逐步估计 image token 的分布，每个 step 都可以看到所有的上下文信息
5. 同时由于采用了 mask-and-replace 的 diffusion 策略，也可以避免误差累计，因为训练的时候没用 teacher forcing，而是 mask token 或者 random token，推理的时候每个 step 会重新采样所有的 token，从而可以修正错误的 token 来避免误差的累积

## 相关工作（略）

## 背景：使用 VQ-VAE 学习 discrete latent space

VQ-VAE 包含 encoder $E$，decoder $G$ 和 codebook $\mathcal{Z}=\{z_k\}_{k=1}^K\in\mathbb{R}^{K\times d}$，$K$ 为向量的数量，$d$ 为编码的维度。给定图像 $\boldsymbol{x}\in\mathbb{R}H\times W\times3$，使用 encoder 得到一些列 image token $\boldsymbol{z}_q$，即首先 $\boldsymbol{z}=E(\boldsymbol{x})\in\mathbb{R}^{h\times w\times d}$，然后使用量化器 $Q(\cdot)$ 进行量化，也就是将每个 $\boldsymbol{z}_{i,j}$ 映射到最 codebook 中最邻近的编码 $\boldsymbol{z}_k$，从而：
$$\boldsymbol{z}_q=Q(\boldsymbol{z})=\left(\operatorname*{argmin}_{\boldsymbol{z}_k\in\mathcal{Z}}\|\boldsymbol{z}_{ij}-\boldsymbol{z}_k\|_2^2\right)\in\mathbb{R}^{h\times w\times d}$$
其中 $h\times w$ 为 encoder 之后的长度。然后通过 decoder 重构图像 $\tilde{\boldsymbol{x}}=G\left(\boldsymbol{z}_q\right)$。此时，图像生成就变成，从 latent distribution 中采样 image token。encoder $E$，decoder $G$ 和 codebook $\mathcal{Z}$ 可以通过一下损失端到端地训练：
$$\begin{array}{r}
\mathcal{L}_{\mathrm{VQVAE}}=\|\boldsymbol{x}-\tilde{\boldsymbol{x}}\|_1+\left\|\operatorname{sg}[E(\boldsymbol{x})]-z_q\right\|_2^2 \\
+\beta\left\|\operatorname{sg}\left[z_q\right]-E(\boldsymbol{x})\right\|_2^2
\end{array}$$
其中的 $\operatorname{sg}$ 为  stop-gradient。将上式中的第二项替换为 exponential moving averages (EMA) 来更新 codebook。

## VQ-Diffusion

给定文本图像对，采用预训练的 VQ-VAE 来得到离散的 image token $\boldsymbol{x} \in \mathbb{Z}^N$，其中 $N=h\times w$ 表示序列长度。位置 $i$ 的 token $x^i$ 对应 codebook 中的 index，即 $x^i \in\{1,2, \ldots, K\}$，同时 text token $\boldsymbol{y} \in \mathbb{Z}^M$ 通过 BPE-encoding 获得。此时 text-to-image 可以被看作是最大化条件分布 $q(\boldsymbol{x} \mid \boldsymbol{y})$。

自回归模型是这么建模的：
$$q(\boldsymbol{x} \mid \boldsymbol{y})=\prod_{i=1}^N q\left(x^i \mid x^1, \cdots, x^{i-1}, \boldsymbol{y}\right)$$
由于 token 是一个一个预测的，忽略了数据的 2D 结构。

提出的 VQ-Diffusion 则采用 diffusion 来最大化这个分布。

### Discrete diffusion process

forward diffusion 通过马尔可夫链 $q\left(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1}\right)$ 逐渐引入噪声，即随机替换 $\boldsymbol{x}_{t-1}$ 中的部分 token，从而得到一系列的带有噪声的 latent variables，$\boldsymbol{z}_1, \ldots, \boldsymbol{z}_T$，其中 $\boldsymbol{z}_T$ 为纯噪声。

reverse diffusion 从 $\boldsymbol{z}_T$ 开始，从 latent variables 中逐步降噪，通过从 $q\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{x}_0\right)$ 中进行采样来恢复数据。但是由于推理时 $\boldsymbol{x}_0$ 是未知的，于是训练一个 transformer  网络来近似 $p_\theta\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{y}\right)$。

给定 image token 记为 $x_0^i \in\{1,2, \ldots, K\}$，忽略上标的位置，简写为 $x_0$。

定义转移概率 $\left[\boldsymbol{Q}_t\right]_{m n}=q\left(x_t=m \mid x_{t-1}=n\right) \in\mathbb{R}^{K\times K}$  ，此时 forward Markov diffusion 可以写为：
$$q\left(x_t \mid x_{t-1}\right)=\boldsymbol{v}^{\top}\left(x_t\right) \boldsymbol{Q}_t \boldsymbol{v}\left(x_{t-1}\right)$$
> 分析可知，$q\left(x_t \mid x_{t-1}\right)$ 其实是一个矩阵，第 $m,n$ 的位置表示 转换概率 $\left[\boldsymbol{Q}_t\right]_{m n}$。

其中 $\boldsymbol{v}(x)$ 是长为 $K$ 的 one-hot 向量（只在位置 $x$ 为 1）。同时根据马尔可夫链的性质，有：
$$q\left(x_t \mid x_0\right)=\boldsymbol{v}^{\top}\left(x_t\right) \overline{\boldsymbol{Q}}_t \boldsymbol{v}\left(x_0\right), \text { with } \overline{\boldsymbol{Q}}_t=\boldsymbol{Q}_t \cdots \boldsymbol{Q}_1$$
还有：
$$\begin{aligned}
q(x_{t-1}|x_t,x_0)=\frac{q(x_t|x_{t-1},x_0)q(x_{t-1}|x_0)}{q(x_t|x_0)} \\
\begin{aligned}=\frac{\left(\boldsymbol{v}^\top(x_t)\boldsymbol{Q}_t\boldsymbol{v}(x_{t-1})\right)\left(\boldsymbol{v}^\top(x_{t-1})\overline{\boldsymbol{Q}}_{t-1}\boldsymbol{v}(x_0)\right)}{\boldsymbol{v}^\top(x_t)\overline{\boldsymbol{Q}}_t\boldsymbol{v}(x_0)}.\end{aligned}
\end{aligned}$$
之前的工作提出，在 categorical distribution 中引入一个小的均匀分布的噪声，此时转移矩阵 $\boldsymbol{Q}_t$ 定义为：
$$\boldsymbol{Q}_t=\left[\begin{array}{cccc}
\alpha_t+\beta_t & \beta_t & \cdots & \beta_t \\
\beta_t & \alpha_t+\beta_t & \cdots & \beta_t \\
\vdots & \vdots & \ddots & \vdots \\
\beta_t & \beta_t & \cdots & \alpha_t+\beta_t
\end{array}\right]$$
其中，$\alpha_t \in[0,1] , \beta_t=\left(1-\alpha_t\right) / K$，这说明每个 token 都有 $(\alpha_t+\beta_{t)}$ 的概率保持不变，有 $K\beta_t$ 的概率从 $K$ 个 token 中均匀采样。

但是，采用均匀分布进行加噪可能有一些问题，因为 token 是有语义信息的，从一个 token 转移到另一个 token 并不可能的完全等概的。
> 这相当于是给了模型一个错误的先验，导致学习的时候会存在某种冲突？

于是提出 Mask-and-replace diffusion strategy，随机 mask 一部分 token，引入一个额外的 token，称为 $[MASK]$，此时就有 $K+1$ 个转移状态。定义 mask diffusion 如下：每个普通的 token（$K$个）都有概率 $\gamma_t$ 被替换为 $[MASK]$，有概率 $K\beta_t$ 均匀采样，有概率 $\alpha_t=1-K\beta_t-\gamma_t$ 保持不变，而如果一个 token 是 $[MASK]$ 则永远保持不变。此时的转移矩阵定义为：
$$\boldsymbol{Q}_t=\begin{bmatrix}\alpha_t+\beta_t&\beta_t&\beta_t&\cdots&0\\\beta_t&\alpha_t+\beta_t&\beta_t&\cdots&0\\\beta_t&\beta_t&\alpha_t+\beta_t&\cdots&0\\\vdots&\vdots&\vdots&\ddots&\vdots\\\gamma_t&\gamma_t&\gamma_t&\cdots&1\end{bmatrix}$$
这种策略的好处是：
+ 改变后的 token 更容易被网络区分，从而使得 reverse 过程更简单
+ 相比于仅做 mask，需要一个均匀噪声，否则会有问题
+ 替换操作使得模型需要理解上下文而非仅关注于 mask token
+ cumulative 转移矩阵可以计算了：$\overline{\boldsymbol{Q}}_t\boldsymbol{v}(x_0)=\overline{\alpha}_t\boldsymbol{v}(x_0)+(\overline{\gamma}_t-\overline{\beta}_t)\boldsymbol{v}(K+1)+\overline{\beta}_t$
+ 计算复杂度也降低了

### reverse 过程

训练一个 denoising 网络 $p_\theta(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t,\boldsymbol{y})$，通过最小化变分下界来训练：
$$\begin{aligned}
\mathcal{L}_{vlb}& =\mathcal{L}_0+\mathcal{L}_1+\cdots+\mathcal{L}_{T-1}+\mathcal{L}_T,  \\
\mathcal{L}_0& =-\log p_\theta(\boldsymbol{x}_0|\boldsymbol{x}_1,\boldsymbol{y}),  \\
\mathcal{L}_{t-1}& =D_{KL}(q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t,\boldsymbol{x}_0)\mid\mid p_\theta(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t,\boldsymbol{y})),  \\
\mathcal{L}_T& =D_{KL}(q(\boldsymbol{x}_T|\boldsymbol{x}_0)\mid\mid p(\boldsymbol{x}_T)). 
\end{aligned}$$
其中 $p(\boldsymbol{x}_T)$ 为先验分布，在 mask-and-replace 策略下，为：
$$p(\boldsymbol{x}_T)=\begin{bmatrix}\overline{\beta}_T,\overline{\beta}_T,\cdots,\overline{\beta}_T,\overline{\gamma}_T\end{bmatrix}^\top $$
> $p(\boldsymbol{x}_T)$ 就是纯噪声分布，$K$ 个正常的 token 之间是均匀分布。

网络是如何参数化的很大程度会影响合成质量，除了直接预测后验 $q\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{x}_0\right)$ ，直接让网络预测原始的无噪声的 target 分布 $q(\boldsymbol{x}_0)$ 会有更好的质量。
> 不就是 DDPM 的意思。。。不是预测下一个 time step 而是直接预测无噪的样本。

这里作者预测的是 $p_\theta(\tilde{\boldsymbol{x}}_0|\boldsymbol{x}_t,\boldsymbol{y})$，从而可以计算 reverse transition distribution 为：
$$p_\theta(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t,\boldsymbol{y})=\sum_{\tilde{\boldsymbol{x}}_0=1}^Kq(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t,\tilde{\boldsymbol{x}}_0)p_\theta(\tilde{\boldsymbol{x}}_0|\boldsymbol{x}_t,\boldsymbol{y})$$
然后引入一个辅助的 denoising  目标函数，使模型可以预测 $\boldsymbol{x}_0$：
$$\mathcal{L}_{x_0}=-\log p_\theta(\boldsymbol{x}_0|\boldsymbol{x}_t,\boldsymbol{y})$$

提出用 encoder-decoder transformer 来估计分布 $p_\theta(\tilde{\boldsymbol{x}}_0|\boldsymbol{x}_t,\boldsymbol{y})$，架构如图：
![](image/Pasted%20image%2020231003232547.png)
diffusion image decoder 的输入为 当前的 image token $\boldsymbol{x}_t$、time step $t$ ，然后得到分布 $p_\theta(\tilde{\boldsymbol{x}}_0|\boldsymbol{x}_t,\boldsymbol{y})$，$t$ 是通过 Adaptive Layer Normalization 操作引入的，即：
$$\operatorname{AdaLN}(h,t)=a_t\text{LayerNorm}(h)+b_t$$

推理时，可以跳过一些步骤来实现快速的推理。假设 time stride 是 $\Delta_t$，实际上可以不需要按照 $x_T,x_{T-1},x_{T-2},\dots,x_0$，而可以按照 $x_T,x_{T-\Delta_t},x_{T-2\Delta_t}\dots x_0$ 来采样：
$$p_\theta(x_{t-\Delta_t}|x_t,y)=\sum_{\tilde{x}_0=1}^Kq(x_{t-\Delta_t}|x_t,\tilde{x}_0)p_\theta(\tilde{x}_0|x_t,y)$$
> 其实就是 DDIM 中的方法。

## 实验（略）