> SLT 2022，Kata.ai Research Team (ID)

1. 提出 Nix-TTS，通过对高质量的、大规模的、非自回归的、端到端的 TTS 教师模型进行知识蒸馏实现轻量化 TTS
2. 采用 model-wise distillation，可以对 encoder 和 decoder 实现灵活的、独立的蒸馏
3. 只有 5.23M 参数，相比于 teacher model 减少了 89.34%

> 本质就是对 VITS 做模型蒸馏。

## Introduction

1. 提出 Nix-TTS，和之前的仅在 duration model 或者仅在 vocoder 上做知识蒸馏不同，本文对整个端到端的模型做蒸馏
2. 提出一种新的 model-wise 蒸馏

## 方法

定义 $\mathcal{F}(\cdot;\omega)$ 为端到端的 TTS 模型，即可以从文本 $c$ 中直接生成原始波形 $x_w$，通常包含 encoder 和 decoder：
$$\mathcal{F}=\mathcal{D}\circ\mathcal{E},\quad x_w=\mathcal{F}(c)=\mathcal{D}(\mathcal{E}(c)),\quad z=\mathcal{E}(c),$$
这里的 $z$ 可以是 deterministic 或者 generative（如概率分布）。

然后定义 $\mathcal{F}_t,\mathcal{F}_s$ 分别为 teacher 和 student，$\{z,x_w\},\{\hat{z},\hat{x}_w\}$ 分别为两个模型的输入输出，给定损失函数 $\mathcal{L}_{\varepsilon},\mathcal{L}_{\mathcal{D}}$，目标是设计并训练 $\mathcal{F}_s$ 使得 $\mathcal{E}_s,\mathcal{D}_s$ 满足$
$$\operatorname{argmin}_{\hat{z}}\mathcal{L}_{\mathcal{E}}(z,\hat{z}),\mathrm{~argmin}_{\hat{x}_w}\mathcal{L}_{\mathcal{D}}(x_w,\hat{x}_w),\mathrm{~}|\omega_s|\ll|\omega_t|,$$
也就是让两个模型的输出尽可能的接近。

采用 VITS 作为 teacher model，VITS 的原理见 [VITS- Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech 笔记](../VITS-%20Conditional%20Variational%20Autoencoder%20with%20Adversarial%20Learning%20for%20End-to-End%20Text-to-Speech%20笔记.md)。

假设训练好了 VITS 模型，其 encoder 和 decoder 结构如下：
+  Prior Encoder 作为 $\mathcal{E}_t$
+ Decoder 作为 $\mathcal{D}_t$

但是 Posterior Encoder 也会编码 latent，从而导致两个 encoder 都可以作为 $\mathcal{E}_t$，但是由于 Prior Encoder 包含的东西更复杂，还是选择 Prior Encoder 作为 $\mathcal{E}_t$。

### Nix-TTS encoder

![](image/Pasted%20image%2020240105203846.png)

Nix-TTS 的 encoder $\mathcal{E}_s$ 用于建模 $q_\theta(z|x)=\mathcal{N}(\mu_q,\sigma_q)$，包含 4 个模块：
+ text encoder
+ text aligner
+ duration predictor
+ latent encoder

text encoder 将 $c$ 编码到 text hidden representation $c_{hidden}$。

text aligner 用的是 [One TTS Alignment To Rule Them All 笔记](../对齐/One%20TTS%20Alignment%20To%20Rule%20Them%20All%20笔记.md) 的方法，现将 $c_{hidden}$ 和 $x_s$ 编码到 $c_{enc},x_{enc}$，然后得到所谓的 soft 对齐 $\mathcal{A}_{soft}$，最后通过 MAS 得到 hard 对齐 $\mathcal{A}_{hard}$：
$$\begin{gathered}\mathcal{A}_{soft}=\mathrm{softmax}(\sum_{j=1}^{J}\sum_{k=1}^{K}(c_{enc}^{j}-x_{enc}^{k})^2)\\\mathcal{A}_{hard}=\mathrm{MAS}(\mathcal{A}_{soft}).\end{gathered}$$

duration predictor 目的是在没有 $x_s$ 的情况下预测 $\mathcal{A}_{hard}$，包含一些卷积层，输出为 duration：
$$d_{hard}^j=\sum_{k=1}^K\mathcal{A}_{hard}^{k,j}$$

latent encoder 和 text encode 的结构一样。

### Nix-TTS decoder

decoder $\mathcal{D}_s$ 用于建模分布 $p_\phi(x|z)$，输入为 latent $z_q$，输出原始波形 $x_w$。其结构和 $\mathcal{D}_t$ 很相似只是参数更少，用的也是 Hiﬁ-GAN 的 generator，训练的时候也有一个 discriminator $\mathcal{C}_s$，其结构和 teacher 模型的 MPD 一致。

为了减少参数，将所有的卷积替换为 depthwise-separable convolutions。

### model-wise 蒸馏

对于 encoder，其损失一方面要最大化在 soft 对齐中给定 $x_s$ 下 $c_{hidden}$ 的似然（$\mathcal{L}_{ForwardSum}$），另一方面要最小化 soft 对齐和 hard 对齐之间的 KL 散度（$\mathcal{L}_{bin}$），同时对于蒸馏过程，为了让两个分布尽可能相似，又要最小化两个高斯分布 $\mathcal{N}(\hat{\mu}_q,\hat{\sigma}_q)\mathrm{~and~}q_\theta(z|x)$ 的 KL 散度（$\mathcal{L}_{kl}$），此时总的 loss 为：
$$\mathcal{L}_{\mathcal{E}}=\mathcal{L}_{ForwardSum}+\mathcal{L}_{bin}+\mathcal{L}_{kl}$$

对于 decoder，也和 VITS 一样，包含 GAN loss，feature mapping loss，mel 谱 重构 loss：
$$\begin{gathered}\mathcal{L}_{adv,disc}=\mathbb{E}_{x_w}[\mathcal{C}_s(x_w)-1)^2+\mathcal{C}_s(\hat{x}_w)^2]\\\mathcal{L}_{adv,gen}=\mathbb{E}_{\hat{x}_w}[\mathcal{C}_s(\hat{x}_w)-1)^2]\\\mathcal{L}_{fmatch}=\sum_{l=1}^{l=L}\frac1{n_l}||\mathcal{C}_s^l(x_w)-\mathcal{C}_s^l(\hat{x}_w)||_1\end{gathered}$$
同时还用了一个 generalized energy distance (GED) loss 来加速训练，提高音频质量，定义为：
$$\mathcal{L}_{ged}=\mathbb{E}_{x_w}[2d_{spec}(x_w,\hat{x}_w^a)-d_{spec}(\hat{x}_w^a,\hat{x}_w^b)]$$
最终损失为：
$$\mathcal{L}_{\mathcal{D}}=\mathcal{L}_{adv,disc}+\mathcal{L}_{adv,gen}+\mathcal{L}_{fmatch}+\mathcal{L}_{recon}+\mathcal{L}_{ged}$$

## 实验（略）

