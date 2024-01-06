> NIPS，2021，Google

1. Classifier guidance 将 diffusion 模型的得分估计和 图像分类器的梯度相结合，需要训练的一个额外的分类器
2. 本文表明：在提出的 classifier-free guidance 中，联合训练一个 conditional 和 unconditional 的 diffusion 模型，将他们的 得分估计的结果组合起来，可以实现采样质量和多样性之间的 trade off，而不需要额外的分类器

> 总的来说，就是训练了一个 conditional 的 diffusion 模型，其中的 condition 可以是真 condition，也可以是论文中所谓的 null token（即 global mean embedding），从而修改最终推理时用的 score 以实现不需要 classifier 的 conditional diffusion。

## Introduction

1. classifier guidance 有几个缺点：
	1. 要训练一个额外的分类器
	2. 可以看成是一种基于梯度的对抗攻击
	3. 优点类似于 GAN 的训练，导致一些评估指标有问题
2. 提出 classifier-free guidance，不需要任何分类器，可以实现真正的 FID 和 IS 的 trade off

## 背景

连续时间的 diffusion 模型表述为：
令 $\mathbf{x}\sim p(\mathbf{x})$ 和 $\mathbf{z}=\{\mathbf{z}_\lambda\mid\lambda\in[\lambda_{\min},\lambda_{\max}]\}$，则 forward process 是一个方差不变的 马尔可夫过程：
$$\begin{aligned}q(\mathbf{z}_\lambda|\mathbf{x})&=\mathcal{N}(\alpha_\lambda\mathbf{x},\sigma_\lambda^2\mathbf{I}),\mathrm{~where~}\alpha_\lambda^2=1/(1+e^{-\lambda}),\sigma_\lambda^2=1-\alpha_\lambda^2\\q(\mathbf{z}_\lambda|\mathbf{z}_{\lambda^{\prime}})&=\mathcal{N}((\alpha_\lambda/\alpha_{\lambda^{\prime}})\mathbf{z}_{\lambda^{\prime}},\sigma_{\lambda|\lambda^{\prime}}^2\mathbf{I}),\mathrm{~where~}\lambda<\lambda^{\prime},\sigma_{\lambda|\lambda^{\prime}}^2=(1-e^{\lambda-\lambda^{\prime}})\sigma_\lambda^2\end{aligned}$$
其中 $\lambda=\log\alpha_{\lambda}^{2}/\sigma_{\lambda}^{2}$，即 $\lambda$ 可以表述为 $\mathbf{z}_\lambda$ 的对数信噪比，forward 过程中，$\lambda$ 逐渐减少。

把初始的 $\mathbf{x}$ 作为条件，forward 过程可以用 reverse 传输写为：
$$\tilde{\mu}_{\lambda^{\prime}|\lambda}(\mathbf{z}_\lambda,\mathbf{x})=e^{\lambda-\lambda^{\prime}}(\alpha_{\lambda^{\prime}}/\alpha_\lambda)\mathbf{z}_\lambda+(1-e^{\lambda-\lambda^{\prime}})\alpha_\lambda\mathbf{x},\quad\tilde{\sigma}_{\lambda^{\prime}|\lambda}^2=(1-e^{\lambda-\lambda^{\prime}})\sigma_{\lambda^{\prime}}^2$$
也就是这个分布是已知的。

reverse 过程则从 $p_\theta(\mathbf{z}_{\lambda_{\mathrm{min}}})=\mathcal{N}(\mathbf{0},\mathbf{I})$ 开始生成，把这个生成过程建模为：
$$p_\theta(\mathbf{z}_{\lambda^{\prime}}|\mathbf{z}_{\lambda})=\mathcal{N}(\tilde{\boldsymbol{\mu}}_{\lambda^{\prime}|\lambda}(\mathbf{z}_{\lambda},\mathbf{x}_{\theta}(\mathbf{z}_{\lambda})),(\tilde{\sigma}_{\lambda^{\prime}|\lambda}^2)^{1-v}(\sigma_{\lambda|\lambda^{\prime}}^2)^v)$$
采样时，从 $\lambda_{\mathrm{min}}$ 开始逐渐增大到 $\lambda_{\mathrm{max}}$，一共执行 $T$ 步。

当参数化 $\mathbf{x}_{\theta}$ 为噪声时， $\mathbf{x}_{\theta}(\mathbf{z}_{\lambda})=(\mathbf{z}_{\lambda}-\sigma_{\lambda}\boldsymbol{\epsilon}_{\theta}(\mathbf{z}_{\lambda}))/\alpha_{\lambda}$，目标函数为：
$$\mathbb{E}_{\boldsymbol{\epsilon},\lambda}\left[\|\boldsymbol{\epsilon}_\theta(\mathbf{z}_\lambda)-\boldsymbol{\epsilon}\|_2^2\right]$$
而对于条件生成，$\mathbf{x}$ 和条件 $\mathbf{c}$ 同时作为输入，即 $\boldsymbol{\epsilon}_\theta(\mathbf{z}_\lambda,\mathbf{c})$。

## Guidance

之前的生成模型（如 GAN、flow-based 模型）的一个好处就是，可以引入 truncated 或者 low temperature 采样，来减少采样的多样性同时提高样本的质量。

但是直接在 diffusion 中做 truncation 或者 low temperature sampling 会导致生成模糊、低质量的图片。

### Classifier Guidance

[Diffusion Models Beat GANs on Image Synthesis 笔记](Diffusion%20Models%20Beat%20GANs%20on%20Image%20Synthesis%20笔记.md) 提出 classifier
guidance，其中的 diffusion score $\epsilon_\theta(\mathbf{z}_\lambda,\mathbf{c})\approx-\sigma_\lambda\nabla_{\mathbf{z}_\lambda}\log p(\mathbf{z}_\lambda|\mathbf{c})$ 修改以使得可以加入一个辅助分类器的对数似然梯度：
$$\tilde{\epsilon}_\theta(\mathbf{z}_\lambda,\mathbf{c})=\epsilon_\theta(\mathbf{z}_\lambda,\mathbf{c})-w\sigma_\lambda\nabla_{\mathbf{z}_\lambda}\log p_\theta(\mathbf{c}|\mathbf{z}_\lambda)\approx-\sigma_\lambda\nabla_{\mathbf{z}_\lambda}[\log p(\mathbf{z}_\lambda|\mathbf{c})+w\log p_\theta(\mathbf{c}|\mathbf{z}_\lambda)]$$
其中 $w$ 用于控制 classifier guidance 的强度。如果把这个 score 作为采样的时候，此时的采样过程就相当于从分布：
$$\tilde{p}_\theta(\mathbf{z}_\lambda|\mathbf{c})\propto p_\theta(\mathbf{z}_\lambda|\mathbf{c})p_\theta(\mathbf{c}|\mathbf{z}_\lambda)^w$$
中进行采样。
> 关于 $w$ 是如何控制强度的，论文的图 2 给出 了一个很好的可视化示例。
> 关于 guiding 一个 conditional 模型和 unconditional 模型，论文中也给出的一段叙述。

### classifier-free guidance

通过训练一个 unconditional diffusion 模型 $p_{\theta}(\mathbf{z})$ （参数化为 $\epsilon_\theta(\mathbf{z}_\lambda)$）和一个 conditional diffusion 模型 $p_{\theta}(\mathbf{z}|\mathbf{c})$（参数化为 $\epsilon_\theta(\mathbf{z}_\lambda,\mathbf{c})$），采用一个神经网络来参数化两个模型，使用一个 null token 表示没有 condition，即 $\epsilon_\theta(\mathbf{z}_\lambda)=\epsilon_\theta(\mathbf{z}_\lambda,\mathbf{c}=\emptyset)$，通过以概率 $p$ 随机设置 $\mathbf{c}$ 为 null token 来同时训练这两个模型。

然后使用下面的线性组合得到的结果进行采样：
$$\widetilde{\epsilon}_\theta(\mathbf{z}_\lambda,\mathbf{c})=(1+w)\epsilon_\theta(\mathbf{z}_\lambda,\mathbf{c})-w\epsilon_\theta(\mathbf{z}_\lambda)$$

