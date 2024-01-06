> CVPR，2022，Ludwig Maximilian University

1. 之前的 diffusion 都是 pixel-based（即输入输出都是图片），为了让 diffusion 模型可以在有限计算资源的场景下训练，把 diffusion 用在 latent space 中
2. 在模型中引入 cross-attention layers，使得 diffusion 可以当成强大生成器，可以以输入为条件（如文本）生成高分辨率的合成图片
3. 提出的 latent diffusion models (LDMs) 可以实现 SOTA 的 impanting 和 synthesis

## Introduction

1. 从已有的 pixel space 出发，生成模型的训练可以分为两部分：
	1. 第一是 perceptual compression 阶段，移除高频细节，学习部分语义信息
	2. 第二步，真的的生成模型学习语义信息和 conceptual composition of the data (semantic compression)
2. 目的是找到更适合计算的 perceptually equivalent space，于是训练 AE 得到低维的表征空间，和数据空间感知相近；而且不需要一连串的 spatial compression，也就是在 latent space 中训练 DM 模型，称这个模型为  Latent Diffusion Models（LDMs）
3. 优点在于，只需要训练一个 universal AE，然后就可以复用在多个 DM 中
4. 贡献如下：
	1. 模型在 compression level 上工作，可以实现更逼真和细节的重构
	2. 可以在多个任务上实现 competitive performance，且极大降低了计算成本
	3. 不需要 delicate weighting of reconstruction and generative abilities
![](image/Pasted%20image%2020230926204919.png)
## 相关工作（略）

## 方法

通过对对应的损失项进行下采样，DM 可以忽略一些感知无关的细节，但是任然需要 pixel level 的计算，导致计算量很大。

提出采用 AE 学习一个和 image space 感知近似的空间，但是减少计算复杂度，优点在于：
+ 采样是在低维空间中进行，计算高效
+ exploit the inductive bias of DMs inherited from their UNet architecture
+ 可以获得一个 general-purpose compression models，其 latent space 可以用于训练多个生成模型

### Perceptual Image Compression

给定 RGB 图像 $x\in\mathbb{R}^{H\times W\times 3}$，encoder $\mathcal{E}$ 将 $x$ 编码到 latent representation $z=\mathcal{E}(x)$，decoder $\mathcal{D}$ 则从 latent v重构图像 $\tilde{x}=\mathcal{D}(z)=\mathcal{D}(\mathcal{E}(x))$，其中 ，$z\in\mathbb{R}^{h\times w\times c}$，encoder 对图像的下采样因子为 $f=H/h=W/w$ 。

同时为了避免 arbitrarily high-variance latent spaces，采用 regularizations，第一个是  KL-reg，在 space 中引入 KL-penalty。

### LDM

图像合成中，最好的模型是基于变分下界的 reweighted，类似于 denoising score matching，其损失函数为：
$$L_{DM}=\mathbb{E}_{x,\epsilon\sim\mathcal{N}(0,1),t}\left[\|\epsilon-\epsilon_\theta(x_t,t)\|_2^2\right]$$
其中的 $t$ 从 $[1,T]$ 均匀采样。

由于前面的 AE，我们可以得到一个低维的 latent space，此时的损失为：
$$L_{LDM}:=\mathbb{E}_{\mathcal{E}(x),\epsilon\sim\mathcal{N}(0,1),t}\left[\|\epsilon-\epsilon_\theta(z_t,t)\|_2^2\right]$$
网络模型是 time-conditional UNet。

### 条件生成

可以通过 conditional denoising autoencoder $\epsilon_\theta(z_t,t,y)$ 实现，这里的条件 $y$ 可以是文本、semantic maps 或者其他。

通过在 Unet 中引入 cross-attention mechanism，将 DM 变成一个更灵活的条件生成器。为了从不同的模态中预处理 $y$，引入 domain specific encoder $\tau_y$ 将 $y$ 投影到中间表征 $\begin{aligned}\tau_\theta(y)\in\mathbb{R}^{M\times d_\tau}\end{aligned}$，然后通过 cross-attention layer 映射到 Unet 的 intermediate layer：
$$\begin{aligned}\text{Attention}(Q,K,V)&=\text{softmax}\left(\frac{QK^T}{\sqrt d}\right)\cdot V\end{aligned}$$
其中，
$$Q=W_Q^{(i)}\cdot\varphi_i(z_t),~K=W_K^{(i)}\cdot\tau_\theta(y),~V=W_V^{(i)}\cdot\tau_\theta(y)$$
如下图：
![](image/Pasted%20image%2020230926211637.png)

此时的 condition LDM 的损失为：
$$L_{LDM}:=\mathbb{E}_{\mathcal{E}(x),y,\epsilon\sim\mathcal{N}(0,1),t}\left[\|\epsilon-\epsilon_\theta(z_t,t,\tau_\theta(y))\|_2^2\right]$$
其中的 $\tau_{\theta},\epsilon_\theta$ 联合优化。

## 实验（略）