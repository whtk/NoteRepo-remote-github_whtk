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
<!-- The mel-style encoder, Encs, takes a reference speech X as input. The goal of the mel-style encoder is to extract a vector w ∈ RN which contains the style such as speaker identity and prosody of given speech X. Similar to Arik et al. (2018), we design the mel-style encoder to comprise of the following three parts:
1) Spectral processing: We first input the mel-spectrogram into fully-connected layers to transform each frames of mel- spectrogram into hidden sequences.
2) Temporal processing: We then use gated CNNs (Dauphin
et al., 2017) with residual connection to capture the sequen- tial information from the given speech.
3) Multi-head self-attention: Then we apply a multi-head self-attention with residual connection to encode the global information. In contrast to Arik et al. (2018) where the multi-head self attention is applied across audio samples, we apply it at the frame level so that the mel-style encoder can better extract style information even from a short speech sample. Then, we temporally average the output of the self-attention to get a one-dimensional style vector w.
 -->
mel-style encoder $Enc_{s}$ 输入语音 $X$，输出 一个 vector $w\in\mathbb{R}^{N}$，包含以下三部分：
+ Spectral processing：通过全连接层
+ Temporal processing：采用带有残差连接的 G-CNN 来捕获信息
+ Multi-head self-attention：attention 做完之后在时间上求平均得到最终的 style vector $w$

### Generator
<!-- The generator, G, aims to generate a speech Xe given a phoneme (or text) sequence t and a style vector w. We build the base generator architecture upon FastSpeech2 (Ren et al., 2020), which is one of the most popular single-speaker mod- els in non-autoregressive TTS. The model consists of three parts; a phoneme encoder, a mel-spectrogram decoder and a variance adaptor. The phoneme encoder converts a sequence of phoneme embedding into a hidden phoneme sequence. Then, the variance adaptor predicts different variances in the speech such as pitch and energy in phoneme-level 1. Furthermore, the variance adaptor predict a duration of each phonemes to regulate the length of the hidden phoneme sequence into the length of speech frames. Finally, the mel- spectrogram decoder converts the length-regulated phoneme hidden sequence into mel-spectrogram sequence. Both the phoneme encoder and mel-spectrogram decoder are com- posed of Feed-Forward Transformer blocks (FFT blocks) based on the Transformer (Vaswani et al., 2017) architec- ture. However, this model does not generate speech with diverse speakers, and thus we propose a novel component to support multi-speaker speech generation, in the following paragraph. -->
generator $G$ 用于从给定的 phoneme 序列 $t$ 和 style vector $w$ 中生成语音 $\widetilde{X}$，结构基于 FastSpeech 2，包含三个部分：
+ phoneme encoder，将 phoneme embedding 转为 hidden sequence
+ mel 谱 decoder，将长度调整后的序列转为 mel 谱
+ variance adaptor，在 phoneme level 的 序列上预测 pitch 和 energy
 
phoneme encoder 和 mel 谱 decoder 用的都是 FFT 模块，但是都不支持多说话人。下面提出支持多说话人的模块。
<!-- Style-Adaptive Layer Norm Conventionally, the style vector is provided to the generator simply through either the concatenation or the summation with the encoder output or the decoder input. In contrast, we apply an alternative approach by proposing the Style-Adaptive Layer Normaliza- tion (SALN). SALN receives the style vector w and predicts the gain and bias of the input feature vector. More pre- cisely, given feature vector h = (h1, h2, . . . , hH ) where H is the dimensionality of the vector, we derive the normalized vector y = (y1,y2,...,yH) as follows: -->
Style-Adaptive Layer Normalization (SALN) 将 style vector 作为输入的 gain 和 bias。给定 feature vector $\boldsymbol{h}=(h_1,h_2,...,h_H)$，得到 normalized vector $\boldsymbol{y}=(y_1,y_2,...,y_H)$：
$$\begin{gathered}\boldsymbol{y}=\frac{\boldsymbol{h}-\mu}{\sigma}\\\mu=\frac{1}{H}\sum_{i=1}^{H}h_i,\quad\sigma=\sqrt{\frac{1}{H}\sum_{i=1}^{H}(h_i-\mu)^2}\end{gathered}$$
<!-- Then, we compute the gain and bias with respect to the style vector w. -->
然后，计算 gain 和 bias：
$$\begin{aligned}SALN(\boldsymbol{h},w)=g(w)\cdot\boldsymbol{y}+b(w)\end{aligned}$$
<!-- Unlike the fixed gain and bias as in LayerNorm (Ba et al., 2016), g(w) and b(w) can adaptively perform scaling and shifting of the normalized input features based on the style vector. We substitute SALN for layer normalizations in FFT blocks in the phoneme encoder and the mel-spectrogram decoder. The affine layer which convert the style vector into bias and gain is a single fully connected layer. By utilizing SALN, the generator can synthesize various styles of speech of multiple speakers given the reference audio sample in addition to the phoneme input. -->
不同于 LayerNorm 中的固定 gain 和 bias，$g(w)$ 和 $b(w)$ 可以根据 style vector 自适应地对输入特征进行缩放和移位。将 SALN 替换 FFT blocks 中的 layer normalization。通过 SALN，generator 可以根据参考音频样本和 phoneme 输入合成多说话人的不同风格的语音。

### 训练
<!-- In the training process, both the generator and the mel-style encoder are optimized by minimizing a reconstruction loss between a mel-spectrogram synthesized by the generator and a ground truth mel-spectrogram2. We use the L1 dis- tance as a loss function, as follows: -->
训练过程中，generator 和 mel-style encoder 都通过最小化生成的 mel-spectrogram 和 GT 之间的重构损失来优化。采用 L1 距离作为损失函数：
$$\begin{aligned}\widetilde{X}&=G(t,w)\quad w=Enc_s(X)\\\mathcal{L}_{recon}&=\mathbb{E}\left[\left\|\widetilde{X}-X\right\|_1\right]\end{aligned}$$
<!-- where Xe is a generated mel-spectrogram given the phoneme input, t, and the style vector, w, which extracted from a ground truth mel-spectrogram, X.
 -->
其中 $\widetilde{X}$ 是给定 phoneme 输入 $t$ 和 style vector $w$ 生成的 mel-spectrogram，且 $w$ 从 GT mel-spectrogram $X$ 中提取。

## Meta-StyleSpeech
<!-- Although StyleSpeech can adapt to the speech from a new speaker by utilizing SALN, it may not generalize well to the speech from an unseen speaker with a shifted distribution. Furthermore, it is difficult to generate the speech to follow the voice of the unseen speaker, especially with few speech audio samples that are also short in length. Thus, we further propose Meta-StyleSpeech, which is meta-learned to further improve the model’s ability to adapt to unseen speakers. In particular, we assume that only a single speech audio sample of the target speaker is available. Thus, we simulate one-shot learning for new speakers via episodic training. In each episode, we randomly sample one support (speech, text) sample, (Xs , ts ), and one query text, tq , from the target speaker i. Our goal is then to generate the query speech Xeq from the query text tq and the style vector ws which is extracted from the support speech Xs. However, a challenge here is that we can not apply reconstruction loss on Xeq, since no ground-truth mel-spectrogram is available. To handle this issue, we introduce an additional adversarial network with two discriminators; a style discriminator and a phoneme discriminator. -->
StyleSpeech 可以通过 SALN 适应新说话人的语音，但可能不适用于分布发生了变化的新说话人的语音。且很难生成新说话人的语音。

提出 Meta-StyleSpeech，通过元学习来进一步提高模型适应新说话人的能力。假设只有一个目标说话人的单个语音样本可用。因此，通过 episodic training 模拟新说话人的 one-shot learning。在每个 episode 中，随机采样一个 support (speech, text) 样本 $(X_s,t_s)$ 和一个 query text $t_q$。目标是从 query text $t_q$ 和从 support speech $X_s$ 中提取的 style vector $w_s$ 生成 query speech $X_{eq}$。但是由于没有 GT mel 谱，无法计算 $X_{eq}$ 上的重构损失。于是引入了一个额外的对抗网络，包含两个 discriminator：style discriminator 和 phoneme discriminator。

### Discriminators
<!-- The style discriminator, Ds, predicts whether the speech follows the voice of the target speaker. The discriminator has similar architecture with mel-style encoder except it contains a set of style prototypes S = {si}Ki=1, where si ∈ RN denotes the style prototype for the i th speaker and K is the number of speakers in the training set. Given the style vector, ws ∈ RN , as input, the style prototype si is learned with following classification loss. -->
style discriminator $D_s$ 预测语音是否符合目标说话人的声音。与 mel-style encoder 类似，但包含一组 style prototypes $S=\{s_i\}_{i=1}^{K}$，其中 $s_i\in\mathbb{R}^{N}$ 表示第 $i$ 个说话人的 style prototype，$K$ 是训练集中说话人的数量。给定 style vector $w_s\in\mathbb{R}^{N}$，学习 style prototype $s_i$ 的分类损失：
$$\begin{aligned}\mathcal{L}_{cls}&=-\log\frac{\exp(w_s^Ts_i))}{\sum_{i'}\exp(w_s^Ts_{i'})}\end{aligned}$$
<!-- In detail, the dot product between the style vector and all style prototypes is computed to produce style logits, fol- lowed by cross entropy loss that encourages the style proto- type to represent the target speaker’s common style such as speaker identity.-->
其实就是算 style vector 和所有 style prototypes 之间的点积，得到 style logits，然后算交叉熵损失。
<!-- The style discriminator then maps the generated speech Xeq to a M -dimensional vector h(Xeq ) ∈ RM and compute a single scalar with the style prototype. The key idea here is to enforce the generated speech to be gathered around the style prototype for each speaker. In other words, the generator learns how to synthesize speech that follows the common style of the target speaker from a single short reference speech sample. Similar to the idea of Miyato & Koyama (2018), the output of the style discriminator is then computed as: -->
style discriminator 将生成的语音 $\widetilde{X}_{q}$ 映射到一个 $M$ 维向量 $\boldsymbol{h}(\widetilde{X}_{q})\in\mathbb{R}^{M}$，然后计算一个标量。关键思想是强制生成的 speech 聚集在每个说话人的 style prototype 周围。换句话说，generator 学习如何从单个短的参考语音样本中合成符合目标说话人的共同风格的语音。style discriminator 的输出计算如下：