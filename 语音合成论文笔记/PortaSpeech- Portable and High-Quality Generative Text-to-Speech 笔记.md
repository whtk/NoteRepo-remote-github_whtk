> ZJU，NIPS 2021，renyi

1. VAE 可以在较小的模型大小下捕获长时的语义特征，但是存在一些  blurry 和 unnatural 的问题
2. 而 flow 善于重构频率细节但是当模型参数受限时效果较差
3. 提出 PortaSpeech，一个 portable 和 高质量 的 TTS 模型：
	1. 为了同时建模韵律和 mel 谱细节，采用 lightweight VAE with an enhanced prior followed by a flow-based post-net with strong conditional inputs 作为主要结构
	2. 为了进一步压缩尺寸和内存，在 affine coupling layers 中引入 grouped parameter sharing 
	3. 为了提高表达性，提出 带有混合对齐的 linguistic encoder，将 hard word-level alignment 和 soft phoneme-level alignment 进行组合，可以显示地提取 word-level 语义信息

## Introduction

1. 现有的 TTS 需要实现以下目标：
	1. Fast：推理快
	2. Lightweight：模型小
	3. High-quality：细节捕获
	4. Expressive：很好的建模 F0 和 duration
	5. Diverse：相同文本下生成不同的 intonations
2. 提出 PortaSpeech

## 背景（略）

## PortaSpeech

![](image/Pasted%20image%2020231222104009.png)

包含：
+  linguistic encoder  with mixture alignment
+ variational generator with enhanced prior
+ flow-based post-net with the grouped parameter sharing

首先，带有单词边界的 文本序列 送到 linguistic encoder 来得到 phoneme-level 和 word-level 的 特征，然后训练 VAE-based variational generator 来基于前面的 特征 最大化 GT mel 谱 的 ELBO，且先验分布通过一个小的 volume-preserving normalizing flow 来建模。最后还训练一个 post-net 来最大化 GT mel 谱的似然。

### Linguistic Encoder with Mixture Alignment

之前的非自回归的 TTS 通常采用 duration predictor 来预测每个 phoneme duration，而 GT phoneme duration 通常通过外部的 aligner 或者 单调对齐训练来获得。但是这种对齐有一些问题：
+ 两个 phoneme 之间的边界可能是不确定的，从而导致获得精确的  phoneme-level 边界很困难，引入一些不可逆的误差
+ 误差进一步影响 duration predictor 的训练，从而影响韵律

于是提出采用混合对齐，在 phoneme-level 实现 soft alignment，在 word level 实现 hard alignment。

linguistic encoder 包含 phoneme encoder、word encider、duration predictor 和 word-to-phoneme attention 模块。

假设有输入包含 phoneme 序列和单词边界，如 “"HH AE1 Z | N EH1 V ER0”（has never），其中 | 就表示单词边界：
+ 先将 phoneme 序列编码到 phoneme hidden state $\mathcal{H}_{p}$
+ 然后采用 word-level pooling 来得到 word encoder 的输入表征（其实就是根据单词边界对每个 phoneme 表征取平均）
+ 然后将 编码后的 word-level hidden states 拓展到 mel 谱 的长度（基于 word level duration），记为 $\mathcal{H}_{w}$（拓展后）
+ 最后，word-to-phoneme attention 模块，$\mathcal{H}_{w}$ 作为 Q，$\mathcal{H}_{p}$ 作为 K 和 V，为了使得 attention 尽可能接近对角，在这两个输入表征中添加  word-level relative positional encoding embedding

> duration predictor 输入为 $\mathcal{H}_{p}$，先计算每个 phoneme duration，然后根据 单词边界进行求和得到 word level duration。

### Variational Generator with Enhanced Prior

把 VAE 作为 mel 谱 generator，之前的工作通常采用高斯分布作为先验，从而导致后验受限。为了增强先验，引入 small volume-preserving normalizing flow 将简单的高斯分布转为复杂分布。然后把复杂分布作为 VAE 的先验。此时 generator 优化的目标函数为：
$$\log p(\mathbf{x}|c)\geq\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x},c)}[\log p_\theta(\mathbf{x}|\mathbf{z},c)]-\text{KL}(q_\phi(\mathbf{z}|\mathbf{x},c)|p_{\bar{\theta}}(\mathbf{z}|c))\equiv\mathcal{L}(\phi,\theta,\bar{\theta})$$
其中的 $\bar{\theta}$  表示流模型的参数。$c$ 表示  linguistic encoder 的输出。然后通过蒙特卡洛方法来估计上式中的 KL 散度：
$$\mathrm{KL}(q_\phi(\mathbf{z}|\mathbf{x},c)|p_{\bar{\theta}}(\mathbf{z}|c))=\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x},c)}[\log q_\phi(\mathbf{z}|\mathbf{x},c)-\log p_{\bar{\theta}}(\mathbf{z}|c)]$$
训练时，后验分布 $N(\mu_q,\sigma_q)$ 通过 generator 的 encoder 来编码，$z_q$ 通过重参数从后验分布中采样，然后通过 generator 的 decoder 。同时后验分布通过 VP-flow 转为标准的正态分布。

推理时，从标准正态分布中采样，然后 VP-flow 转为先验分布 $z_p$，然后输入到 decoder。

### Flow-based Post-Net

采用 flow-based post-net with strong condition inputs 来 refine generator 的输出。本质是一个 flow 模型，把 generator 和 linguistic encoder 的输出作为条件，训练时，将 mel 谱 转为 latent prior distribution，且计算数据的精确似然。推理时，从这个分布中采样，然后通过 post-net 来生成高质量的 mel 谱。

为了进一步减小模型大小，在 affine coupling layer 中引入 grouped parameter sharing 机制，在不同的 flow step 中共享部分模型参数，如图：
![](image/Pasted%20image%2020231222154805.png)

将所有的 flow step 分为几组，然后组内的  NN 层的参数被共享。

### 训练和推理

训练时，PortaSpeech 的损失包含以下：
+ duration prediction loss $L_{dur}$
+ generator 重构损失 $L_{VG}$
+ generator 的 KL 散度 损失 $L_{KL}=\log q_\phi(\mathbf{z}|\mathbf{x},c)-\log p_{\bar{\theta}}(\mathbf{z}|c),\:\mathbf{w}\text{here z}\sim q_\phi(\mathbf{z}|\mathbf{x},c)$
+ post-net 的负对数似然损失 $L_{PN}$

推理时：
+ linguistic encoder 先编码文本序列，然后预测 word-level duration，然后根据对齐拓展 hidden states 到 $\mathcal{H}_L$
+ 然后从 enhanced prior 中采样得到 $z$，然后 generator 的 decoder 基于 $\mathcal{H}_L$ 生成 coarse-grained mel 谱 $\bar{M}_c$
+ post-net 将随机采样的 latent 转为 fine-grained mel 谱 $\bar{M}_f$（基于 $\mathcal{H}_L$ 和 $\bar{M}_c$） 
+ $\bar{M}_f$ 通过 vocoder 生成波形

## 实验（略）
