> ICML 2021，KAIST，AITRICS

1. 提出 StyleSpeech，不仅可以合成高质量的语音，也可以适应到新的说话人
2. 提出 Style-Adaptive Layer Normalization，根据从参考语音中提取的 style 来对齐文本输入的 gain 和 bias
3. 拓展到 Meta-StyleSpeech，引入两个 discriminator

## Introduction

1. 现有的 meta learning 都集中在图像上，不能在 TTS 上用
2. 提出 StyleSpeech 和 Meta-StyleSpeech

## 相关工作（略）

## StyleSpeech

包含 mel-style encoder 和 generator，如图：
![](image/Pasted%20image%2020231207112353.png)

### mel style encoder

mel-style encoder $Enc_{s}$ 输入语音 $X$，输出 一个 vector $w\in\mathbb{R}^{N}$，包含以下三部分：
+ Spectral processing：通过全连接层
+ Temporal processing：采用带有残差连接的 G-CNN 来捕获信息
+ Multi-head self-attention：attention 做完之后在时间上求平均得到最终的 style vector $w$

### Generator

generator $G$ 用于从给定的 phoneme 序列 $t$ 和 style vector $w$ 中生成语音 $\widetilde{X}$，结构基于 FastSpeech 2，包含三个部分：
+ phoneme encoder，将 phoneme embedding 转为 hidden sequence
+ mel 谱 decoder，将长度调整后的序列转为 mel 谱
+ variance adaptor，在 phoneme level 的 序列上预测 pitch 和 energy
 
phoneme encoder 和 mel 谱 decoder 用的都是 FFT 模块，但是都不支持多说话人。下面提出支持多说话人的模块。

提出采用 Style-Adaptive Layer Normalization (SALN)， 将 style vector 作为输入的 gain 和 bias。给定 feature vector $\boldsymbol{h}=(h_1,h_2,...,h_H)$，得到 normalized vector $\boldsymbol{y}=(y_1,y_2,...,y_H)$：
$$\begin{gathered}\boldsymbol{y}=\frac{\boldsymbol{h}-\mu}{\sigma}\\\mu=\frac{1}{H}\sum_{i=1}^{H}h_i,\quad\sigma=\sqrt{\frac{1}{H}\sum_{i=1}^{H}(h_i-\mu)^2}\end{gathered}$$

然后，计算 gain 和 bias：
$$\begin{aligned}SALN(\boldsymbol{h},w)=g(w)\cdot\boldsymbol{y}+b(w)\end{aligned}$$

不同于 LayerNorm 中的固定 gain 和 bias，$g(w)$ 和 $b(w)$ 可以根据 style vector 自适应地对输入特征进行缩放和移位。将 SALN 替换 FFT blocks 中的 layer normalization。通过 SALN，generator 可以根据参考音频样本和 phoneme 输入合成多说话人的不同风格的语音。

### 训练

训练过程中，generator 和 mel-style encoder 都通过最小化生成的 mel-spectrogram 和 GT 之间的重构损失来优化。采用 L1 距离作为损失函数：
$$\begin{aligned}\widetilde{X}&=G(t,w)\quad w=Enc_s(X)\\\mathcal{L}_{recon}&=\mathbb{E}\left[\left\|\widetilde{X}-X\right\|_1\right]\end{aligned}$$
其中 $\widetilde{X}$ 是给定 phoneme 输入 $t$ 和 style vector $w$ 生成的 mel-spectrogram，且 $w$ 从 GT mel-spectrogram $X$ 中提取。

## Meta-StyleSpeech

StyleSpeech 可以通过 SALN 适应新说话人的语音，但可能不适用于分布发生了变化的新说话人的语音。且很难生成新说话人的语音。

于是提出 Meta-StyleSpeech，通过元学习来进一步提高模型适应新说话人的能力。假设只有一个目标说话人的单个语音样本可用。因此，通过 episodic training 模拟新说话人的 one-shot learning。在每个 episode 中，随机采样一个 support (speech, text) 样本 $(X_s,t_s)$ 和一个 query text $t_q$。目标是从 query text $t_q$ 和从 support speech $X_s$ 中提取的 style vector $w_s$ 生成 query speech $X_{eq}$。但是由于没有 GT mel 谱，无法计算 $X_{eq}$ 上的重构损失。于是引入了一个额外的对抗网络，包含两个 discriminator：style discriminator 和 phoneme discriminator。

如图：
![](image/Pasted%20image%2020240202114115.png)

### Discriminators

style discriminator $D_s$ 预测语音是否符合目标说话人的声音。与 mel-style encoder 类似，但包含一组 style prototypes $S=\{s_i\}_{i=1}^{K}$，其中 $s_i\in\mathbb{R}^{N}$ 表示第 $i$ 个说话人的 style prototype，$K$ 是训练集中说话人的数量。给定 style vector $w_s\in\mathbb{R}^{N}$，学习 style prototype $s_i$ 的分类损失：
$$\begin{aligned}\mathcal{L}_{cls}&=-\log\frac{\exp(w_s^Ts_i))}{\sum_{i'}\exp(w_s^Ts_{i'})}\end{aligned}$$

其实就是算 style vector 和所有 style prototypes 之间的点积，得到 style logits，然后算交叉熵损失。

style discriminator 将生成的语音 $\widetilde{X}_{q}$ 映射到一个 $M$ 维向量 $\boldsymbol{h}(\widetilde{X}_{q})\in\mathbb{R}^{M}$，然后计算一个标量。关键思想是强制生成的 speech 聚集在每个说话人的 style prototype 周围。换句话说，generator 学习如何从单个短的参考语音样本中合成符合目标说话人的共同风格的语音。style discriminator 的输出计算如下：
$$D_s(\widetilde{X}_q,s_i)=w_0{s_i}^TVh(\widetilde{X}_q)+b_0$$
其中 $V\in\mathbb{R}^{N\times M}$ 是一个线性层，$w_0$ 和 $b_0$ 是可学习参数。style discriminator 的损失函数为：
$$\mathcal{L}_{D_s}=\mathbb{E}_{t,w,s_i\sim S}[(D_s(X_s,s_i){-}1)^2{+}D_s(\widetilde{X}_q,s_i)^2]$$

discriminator 损失遵循 LS-GAN，用最小二乘损失函数替换了原始 GAN 的二分类交叉熵。

phoneme discriminator $D_t$ 输入 $\widetilde{X}_{q}$ 和 $t_q$，根据 phoneme sequence $t_q$ 来区分真实和生成语音。根据 phoneme 将 mel 谱中的每个 frame 连接起来。然后 discriminator 计算每个 frame 的标量然后求平均得到一个标量。phoneme discriminator 的损失函数为：
$$\mathcal{L}_{D_t}\:=\:\mathbb{E}_{t,w}[(D_t(X_s,t_s)\:-\:1)^2\:+\:D_t(\widetilde{X}_q,t_q)^2]$$

最终 query speech generator loss 定义为每个 discriminator 的对抗损失之和：
$$\begin{gathered}\mathcal{L}_{adv}=\mathbb{E}_{t,w,s_i\sim S}[(D_s(G(t_q,w_s),s_i)-1)^2]+\\\mathbb{E}_{t,w}[(D_t(G(t_q,w_s),t_q)-1)^2].\end{gathered}$$

此外还用了 support speech 的重构损失，实验发现可以提高生成的 mel 谱的质量：
$$\mathcal{L}_{recon}=\mathbb{E}\left[\left\|G(t_s,w_s)-X_s\right\|_1\right]$$

### Episodic meta-learning

meta-learning 交替更新 generator 和 mel-style encoder 以最小化 $L_{recon}$ 和 $L_{adv}$，以及更新 discriminators 以最小化 $L_{D_s}$、$L_{D_t}$ 和 $L_{cls}$。最终的 meta-training loss 定义为：
$$\begin{aligned}\mathcal{L}_G&=\alpha\mathcal{L}_{recon}+\mathcal{L}_{adv}\\\\\mathcal{L}_D&=\mathcal{L}_{D_s}+\mathcal{L}_{D_t}+\mathcal{L}_{cls}\end{aligned}$$实验中，$\alpha=10$。

## 实验（略）
