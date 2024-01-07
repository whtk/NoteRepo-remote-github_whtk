> ISCSLP 2022，西工大、腾讯

1. TTS 中的说话人适应需要在有限数据下 fine tune 预训练的 TTS 模型来适应到新的目标说话人
2. 很少研究低计算资源下的说话人自适应
3. 提出 AdaVITS：
	1. 采用 iSTFT-based decoder 减少了原始的 VITS 的资源消耗
	2. 引入 NanoFlow 来减少 prior encoder 中的参数
	3. 用 linear attention 替换文本编码器中的 scaled-dot attention 来减少计算复杂度
	4. 采用  phonetic posteriorgram (PPG) 作为 frame-level linguistic 特征

## Introduction

1. 提出 AdaVITS：
	1. 将基于上采样的 decoder 替换为 iSTFT-based decoder
	2. 采用 ﬂow indication embedding (FLE) 共享 flow block 的参数
	3. 对于 FFT blocks，用 linear attention 替换 scaled-dot attention 来减少计算复杂度
	4. 为了避免简化后模型的不稳定，采用 PPG 作为 frame-level linguistics feature 来约束 phoneme- spectrum 的建模过程
2. 8.97M 模型参数，0.72 GFlops 计算量

## 方法

见 [VITS- Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech 笔记](../VITS-%20Conditional%20Variational%20Autoencoder%20with%20Adversarial%20Learning%20for%20End-to-End%20Text-to-Speech%20笔记.md) 。

采用 PPG 作为 latent $z$ 和 phoneme 之间的中间约束的好处是，PPG 可以显式地解耦 timbre 和 content 信息，实现更灵活的建模。结构如图：
![](image/Pasted%20image%2020240107152245.png)

包含：posterior encoder，prior encoder 和 decoder。posterior encoder 用于从波形中提取 $z$（只在训练时使用，推理时不用）；prior encoder 用于从 phoneme 中提取关于 $z$ 的先验分布 $p(z|c)$ ，decoder 基于 $z$ 和 speaker embedding 合成波形。

### Posterior Encoder

Posterior Encoder 和原始的 VITS 中的一致。

### Prior Encoder

从一个 speaker-independent 的 ASR 模型中提取 PPG，得到不包含 speaker 信息的 frame-level linguistic feature。

在 text encoder 中采用更少的 FFT 层，然后用 length regulator 拓展到 frame level。然后把 FFT 中的 scaled-dot attention 替换为 linear attention。
> 从 phoneme 中提取 PPG 的模型是预训练好的且不进行 fine tune。

得到 PPG 后，PPG encoder 得到先验分布的均值和方差 $\mu_{\theta},\sigma_{\theta}$，PPG encoder 也包含 FFT 模块，用的也是 linear attention。

同时引入 PPG predictor 来提供发音约束，用的是 VISinger 中的 phoneme predictor 的结构，输入为 $z$，输出预测的 PPG，损失为：
$$L_{\mathrm{ppg}}=\left\|\text{PPG}-\mathrm{PPG}\right\|_1$$
且 PPG predictor 只在预训练的时候训练，adaptation 的时候固定住。

对于 flow，其包含多层的 affine coupling，每层都是一些 WaveNet residual block 结构，采用 NanoFlow 中的想法，共享每个 flow 中的 ACL 层的参数。每层都采用 FLE 来区分。

### Decoder

上采样层虽然建模能力很强，但是计算耗时大，于是可以采用 iSTFT 来直接合成波形，如图：
![](image/Pasted%20image%2020240107165143.png)

decoder-v1 采用多个卷积逐渐增加输入维度到 $(f/2+1)*2$，其中 $f$ 表示 FFT size。在 1D 卷积中使用了 group convolution 来减少计算量。最后把输出分为实部和虚部，波形可以通过 iSTFT 产生。同时把 speaker embedding 作为 condition 来提高 speaker similarity。

decoder-v2 可以在复杂度和合成质量之间 trade-off，只用 decoder-v1 来建模高频部分，然后用上采样层来建模低频部分。因为：
+ 上采样层可以建模高频谐波
+ 高频部分需要较少的建模能力
最后通过 PQMF 合并两个频带。

### Discriminator

用的是 multi-resolution spectrum discriminator (MSD) 和 multi-resolution complex-valued spectrum discriminator (MCD) 这两个 discriminator。

提出的 MCD 可以提高相位建模准确度。

### 损失

包含 CVAE 和 GAN loss，CVAE 部分为：
$$L_\mathrm{cvae}=L_\mathrm{kl}+\lambda_\mathrm{recon}*L_\mathrm{recon}+\lambda_\mathrm{ppg}*L_\mathrm{ppg}$$

最终包含 GAN 后的 loss 为：
$$L_\mathrm{G}=L_\mathrm{adv}(\mathrm{G})+\lambda_\mathrm{fm}*L_\mathrm{fm}(\mathrm{G})+L_\mathrm{cvae},\quad L_{\mathrm{D}}=L_{\mathrm{adv}}(\mathrm{D})
$$

## 实验（略）