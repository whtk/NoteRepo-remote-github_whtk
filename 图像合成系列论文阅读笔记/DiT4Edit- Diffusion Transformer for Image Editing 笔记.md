> preprint 2024.11 北大、港科大

1. UNet-based 图像编辑方法在高分辨率图像中缺乏对 shape-aware object editing 的支持
2. 提出 DiT4Edit，第一个基于 DiT 的图像编辑框架：
    1. 采用 DPM-Solver inversion 算法获取 inverted latents
    2. 设计统一的 attention control 和 patches merging，用于 transformer computation streams，可以更快生成高质量的编辑图像
3. 实验表明 DiT4Edit 在高分辨率和任意大小图像的编辑中优于 UNet 结构

> 缝缝补补，本质上就是 DiT（MasaCtrl 的 attention 方法）+ DPM-Solver（2M） + Patches Merging

## Introduction

1. 图像编辑：给定输入图像，根据用户意图添加、删除或替换整个对象或对象属性
2. 早期方法微调 diffusion 模型来解决 source 和 target images 之间的一致性问题，但是很耗时
3. 当前的图像编辑主要使用基于 UNet 的 diffusion 模型结构，但是 UNet 的生成能力限制编辑效果
4. 提出 DiT4Edit，第一个基于 diffusion transformer 的编辑框架，可以在更少的推理步下实现更好的编辑效果，优于传统的 UNet-based 方法

## 相关工作（略）

## 方法

目标是基于 DiT 实现高质量的图像编辑。

### 预备知识：LDM

LDM 在 latent space $\mathcal{Z}$ 中进行去噪，用预训练的 image encoder $\mathcal{E}$ 将输入图像 ${x}$ 编码为 latents ${z}=\mathcal{E}({x})$。训练时，通过去噪声来优化 denoising UNet $\epsilon_{\theta}$，条件是 text prompt embedding ${y}$ 和当前图像 ${z_t}$，即第 $t\in[0,T]$ 步的 ${z_0}$ 的带噪样本：
$$\min_\theta E_{z_0,\epsilon\thicksim N(0,I),t}\left\|\epsilon-\epsilon_\theta\left(z_t,t,y\right)\right\|_2^2,$$

训练后，可以将随机噪声 $\epsilon$ 转换为图像样本 ${z}$。

在 DPM 的 inversion 阶段，干净图像 ${x_0}$ 逐渐加入高斯噪声，变成带噪样本 ${x_t}$：
$$q(\boldsymbol{x}_t|\boldsymbol{x}_0)=\mathcal{N}(\boldsymbol{x}_t|\alpha_t\boldsymbol{x}_0,\sigma_t^2\boldsymbol{I}),$$
其中 $\frac{\alpha^2_t}{\sigma^2}$ 是 SNR，是 $t$ 的严格递减函数。DPM 采样比其他方法求 ODE 更快：
$$\frac{d\boldsymbol{x}_t}{dt}=(f(t)+\frac{g^2(t)}{2\sigma_t^2})\boldsymbol{x}_t-\frac{\alpha_tg^2(t)}{2\sigma_t^2}\boldsymbol{x}_\theta(\boldsymbol{x}_t,t),$$
其中 $\boldsymbol{x}_T\sim\mathcal{N}(0,\alpha^2,I)$，$f(t)=\frac{d\log\alpha_t}{dt}$，$g(t)=\frac{d\alpha^2_t}{dt}-2\frac{d\log\alpha_t}{dt}\alpha^2_t$。通过设置 $\boldsymbol{x}_s$ 的值，可以计算方程 $\boldsymbol{x}_t$ 的解：
$$\boldsymbol{x}_t=\frac{\alpha_t}{\alpha_s}\boldsymbol{x}_s-\alpha_t\int_{\lambda_s}^{\lambda_t}e^{-\lambda}\boldsymbol{x}_\theta(\hat{\boldsymbol{x}_\lambda},\lambda)\mathrm{d}\lambda,$$
其中 $\lambda_t=\log(\frac{\alpha_t}{\sigma_t})$ 是 $t$ 的递减函数，$\lambda(t)$ 是反函数。DPM-Solver 可以在 10-20 步内采样出真实图像。

### diffusion 模型架构

和 UNet 结构相比，DiT 具有更好的扩展性，生成更高质量的图像，性能更好。

这里采用 [PixArt- α- Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis](PixArt-%20α-%20Fast%20Training%20of%20Diffusion%20Transformer%20for%20Photorealistic%20Text-to-Image%20Synthesis.md) 作为 backbone。

相比于 UNet，transformer 包含全局 attention，模型能够关注更广的范围，使得 transformer 能够生成高质量的大尺寸图像。

### 基于 DiT 的图像编辑

在预训练的 DiT 基础上，提出的图像编辑框架如图：
![](image/Pasted%20image%2020241226162649.png)

#### DPM-Solver

使用高阶 DPM-Solver 可以提高采样速度。积分项 $\lambda_t\lambda_s\exp(-\lambda x_\theta)d\lambda$，在时间 $t_{ti-1}$ 处使用 $\lambda_{ti-1}$ 的 Taylor 展开，DPM-Solver++ 可以在时间 $t_i$ 处得到精确解：
$$\boldsymbol{x}_{t_i}=\frac{\sigma_{t_i}}{\sigma_{t_{i-1}}}\boldsymbol{x}_{t_{i-1}}+\sigma_{t_i}\sum_{n=0}^{k-1}\underbrace{\boldsymbol{x}_\theta^{(n)}(\boldsymbol{x}_{\lambda_{t_{i-1}}},\lambda_{t_{i-1}})}_{\text{estimated}}\\\int_{\lambda_{t_{i-1}}}\underbrace{e^\lambda\frac{(\lambda-\lambda_{t_{i-1}})^n}{n!}\mathrm{d}\lambda}_{\text{analtically computed}}+\underbrace{\mathcal{O}(h_i^{k+1})}_{\text{omitted}},$$

当 $k=1$ 时，上式等价于 DDIM sampler：
$$\boldsymbol{x}_{t_i}=\frac{\sigma_{t_i}}{\sigma_{t_{i-1}}}\boldsymbol{x}_{t_{i-1}}-\alpha_{t_i}(e^{-h_i}-1)\boldsymbol{x}_\theta(\boldsymbol{x}_{t_{i-1}},t_{i-1}),$$

在实际应用中，设置 $k=2$，称为 DPM-Solver++ (2M)：
$$\boldsymbol{x}_{t_i}=\frac{\sigma_{t_i}}{\sigma_{t_{i-1}}}\boldsymbol{x}_{t_{i-1}}-\alpha_{t_i}(e^{-h_i}-1) \cdot[(1+\frac1{2r_i})\boldsymbol{x}_\theta(x_{t_{i-1}},t_{i-1})-\frac1{2r_i}\boldsymbol{x}_\theta(\boldsymbol{x}_{t_{i-2}},t_{i-2})]$$

但是在高阶采样器的 inversion 阶段，为了得到当前时间步 $t_i$ 的 inversion 结果 $\boldsymbol{x}_{t_i}$，需要近似先前时间步的值，如 $\{t_{i-2},t_{i-3},\ldots\}$。

有人引入了 backward Euler 方法来得到上式中的高阶项近似：
$$\boldsymbol{d}_i^{\prime}=\boldsymbol{z}_\theta(\hat{z}_{t_{i-1}},t_{i-1})+\frac{\boldsymbol{z}_\theta(\boldsymbol{\hat{y}}_{t_{i-1}},t_{i-1})-\boldsymbol{z}_\theta(\boldsymbol{\hat{y}}_{t_{i-2}},t_{i-2})}{2r_i},$$

其中 $\boldsymbol{z}_\theta$ 是 denosing model，$\{\hat{\boldsymbol{y}}_{t_{i-1}},\hat{\boldsymbol{y}}_{t_{i-2}},\ldots\}$ 是通过 DDIM inversion 计算的值，用于估计上式中的 $(\hat{\boldsymbol{x}}_{t_{i-1}},\boldsymbol{x}_{t_{i-2}})$，$r_i=\frac{\lambda_{t_{i-1}}-\lambda_{t_{i-2}}}{\lambda_{t_{i}}-\lambda_{t_{i-1}}}$。然后可以通过以下方式得到当前时间步的 inversion latent $\hat{\boldsymbol{z}}_{t_{i-1}}$：
$$\hat{\boldsymbol{z}}_{{t_{i-1}}}=\hat{\boldsymbol{z}}_{{\boldsymbol{t}_{{\boldsymbol{i}-1}}}}-\rho(\boldsymbol{z}_{{t_{i}}}^{\prime}-\boldsymbol{\hat{z}}_{{t_{i}}}),$$

其中 $\boldsymbol{z}_{t_i}^{\prime}=\frac{\sigma_{t_i}}{\sigma_{t_{i-1}}}\boldsymbol{\hat{z}}_{t_{i-1}}-\alpha_{t_i}(e^{-h_i}-1)\boldsymbol{d}_i^{\prime}$。DiT4Edit 使用 DPM-Solver++ inversion 从输入图像 ${x_0}$ 中得到 inversion latent。

#### Unified Attention Control

在 Prompt to Prompt (P2P) 中，cross attention layers 包含来自 prompt texts 的语义信息。通过在 diffusion 过程中替换 source image 和 target image 之间的 cross attention maps，可以编辑图像。
> 两种常用的 text-guided cross attention 策略是 cross attention replacement 和 cross-attention refinement。

DiT 中的 self-attention 用于 image layout 的形成。
> object 和 layout 信息在 transformer 浅层的 query vectors 中没有完全捕获，但在深层中有很好的表示。随着 transformer 层数的增加，query vectors 捕获细节的能力更加具体。表明 transformer 的全局 attention 可以更有效地捕获长距离信息。
这说明通过控制 self attention 可以实现图像的非刚性编辑（non-rigid editing）。在 MasaCtrl 中，引入了 mutual self attention control 机制。具体来说，在 diffusion 的早期步骤中， $Q_{tar},K_{tar},V_{tar}$ 的特征将用于 self attention 计算，生成更接近目标 prompt 的 image layout，而在后期阶段，重建步骤的特征 $K_{src}$ 和 $V_{src}$ 将用于引导生成更接近原始图像的目标图像 layout。

MasaCtrl 可能是由于在整个编辑过程中都使用 $Q_{tar}$ 从而存在一些问题，这里通过设置阈值 $S$ 来确定何时采用 $Q_{src}$：
$$\text{Mutual Edit}=\begin{cases}\mathrm{Attention}\{Q_{\mathrm{src}},K_{\mathrm{src}},V_{\mathrm{src}}\},\mathrm{~if~}t>S\\\mathrm{Attention}\{Q_{\mathrm{tar}},K_{\mathrm{src}},V_{\mathrm{src}}\},\mathrm{~otherwise}&&&\end{cases}$$

#### Patches Merging

将 patches merging 引入到模型中（因为 attention 计算涉及的 patches 数量远大于 UNet）。计算流程如图：
![](image/Pasted%20image%2020241226164435.png)

给定一个 feature map，首先计算每个 patch 之间的相似度，然后合并最相似的 patch 来减少数量。合并后进行 attention 计算。attention 计算后，解除合并的 patch 


## 实验（略）
