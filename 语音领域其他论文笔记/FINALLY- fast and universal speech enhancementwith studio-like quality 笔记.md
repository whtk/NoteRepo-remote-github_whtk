> NIPS 2024，Samsung Research
<!-- 翻译&理解 -->
<!-- In this paper, we address the challenge of speech enhancement in real-world
recordings, which often contain various forms of distortion, such as background
noise, reverberation, and microphone artefacts. We revisit the use of Generative
Adversarial Networks (GANs) for speech enhancement and theoretically show
that GANs are naturally inclined to seek the point of maximum density within
the conditional clean speech distribution, which, as we argue, is essential for the
speech enhancement task. We study various feature extractors for perceptual loss
to facilitate the stability of adversarial training, developing a methodology for
probing the structure of the feature space. This leads us to integrate WavLM-based
perceptual loss into MS-STFT adversarial training pipeline, creating an effective
and stable training procedure for the speech enhancement model. The resulting
speech enhancement model, which we refer to as FINALLY, builds upon the HiFi++
architecture, augmented with a WavLM encoder and a novel training pipeline.
Empirical results on various datasets confirm our model’s ability to produce clear,
high-quality speech at 48 kHz, achieving state-of-the-art performance in the field
of speech enhancement. -->
1. 语音增强的难点：实际录音包含各种形式的失真，如背景噪声、混响和麦克风伪影
2. 本文从理论上证明 GAN 对于 speech enhancement 任务至关重要
3. 研究了不同的 perceptual loss 特征提取器，提高了对抗训练的稳定性
4. 将 WavLM-based perceptual loss 集成到 MS-STFT 对抗训练中，从而提高模型训练稳定性

## Introduction
<!-- Speech recordings are often contaminated with background noise, reverberation, reduced frequency
bandwidth, and other distortions. Unlike classical speech enhancement (Ephraim & Malah, 1984;
Pascual et al., 2017), which considers each task separately, universal speech enhancement (Serrà
et al., 2022; Su et al., 2021; Liu et al., 2022) aims to restore speech from all types of distortions
simultaneously. Thus, universal speech enhancement seeks to generalize across a wide range of
distortions, making it more suitable for real-world applications where multiple distortions may
coexist. -->
1. 通用的语音增强需要在各类失真中恢复语音
<!-- Recent studies have categorized the problem of speech enhancement as a task of learning the clean
speech distribution conditioned on degraded signals (Lemercier et al., 2023; Serrà et al., 2022; Richter
et al., 2023). This problem is often addressed using diffusion models (Ho et al., 2020; Song et al.,
2020), which are renowned for their exceptional ability to learn distributions. Diffusion models
have recently achieved state-of-the-art results in universal speech enhancement (Serrà et al., 2022). 
However, the impressive performance of diffusion models comes with the high computational cost of
their iterative inference process.-->
2. diffusion 模型效果很好，但是计算速度慢
<!-- It is important to note that the speech enhancement problem does not require the model to learn the
entire conditional distribution. In practice, when presented with a noisy speech sample, the goal is
often to obtain the most probable clean speech sample that retains the lexical content and voice of the
original. This contrasts with applications such as text-to-image synthesis (Ramesh et al., 2022; Lee
et al., 2024; Rombach et al., 2022), where the objective is to generate a variety of images for each text
prompt due to the higher level of uncertainty and the need for diverse options to select the best image.
For most speech enhancement applications, such as voice calls and compensation for poor recording
conditions, capturing the entire conditional distribution is not necessary. Instead, it is more important
to retrieve the most likely sample of this distribution (the main mode), which might be a simpler task. -->
3. 语音增强不需要模型学习整个条件分布，只需要获取最可能的干净语音样本
<!-- Diffusion models’ main advantage over generative adversarial networks (GANs) (Goodfellow et al.,
2014) is their ability to capture different modes of the distribution. However, we argue that this prop-
erty is not typically required for the task of speech enhancement and may unnecessarily complicate
the operation of the neural network. Conversely, we show that GANs tend to retrieve the main mode
of the distribution—precisely what speech enhancement should typically do. -->
<!-- Therefore, in this work, we revisit the GAN framework for speech enhancement and demonstrate
that it provides rapid and high-quality universal speech enhancement. Our model outperforms both
diffusion models and previous GAN-based models, achieving an unprecedented level of quality on
both simulated and real-world data. -->
4. 本文展示了 GAN 在 speech enhancement 任务中的优势，贡献如下：
<!-- We theoretically analyse the adversarial training with the least squares GAN (LS-GAN) loss
and demonstrate that a generator predicting a single sample per input condition (producing
a conditional distribution that is a delta function) is incentivized to select the point of
maximum density. Therefore, we establish that LS-GAN training can implicitly regress for
the main mode of the distribution, aligning with the objectives of the speech enhancement
problem.
2. We investigate various feature extractors as backbones for perceptual loss and propose
criteria for selecting an extractor based on the structure of its feature space. These criteria are
validated by empirical results from a neural vocoding task, indicating that the convolutional
features of the WavLM neural network(Chen et al., 2022b) are well-suited for perceptual
loss in speech generation.
3. We develop a novel model for universal speech enhancement that integrates the proposed
perceptual loss with MS-STFT discriminator training (Défossez et al., 2023) and enhances
the architecture of the HiFi++ generator (Andreev et al., 2022) by combining it with a
self-supervised pre-trained WavLM encoder (Chen et al., 2022b). Our final model delivers
state-of-the-art performance on real-world data, producing high-quality, studio-like speech
at 48 kHz. -->
5. 贡献如下：
    1. 理论分析 LS-GAN loss，证明了 generator 会选择密度最大的点
    2. 研究了不同的 perceptual loss 特征提取器，提出了选择标准
    3. 提出了一种新的 universal speech enhancement 模型，集成了 perceptual loss 和 MS-STFT discriminator training，增强了 HiFi++ generator 架构

<!-- Mode Collapse and Speech Enhancement -->
## Mode Collapse 和语音增强
<!-- The first question that we address is what is the practical purpose of a speech enhancement
model. The practical goal of a speech enhancement model is to restore the audio signal containing the
speech characteristics of the original recording, including the voice, linguistic content, and prosody.
Thus, loosely speaking, the purpose of the speech enhancement task for many applications is not
“generative” in its essence, in the sense that the speech enhancement model should not generate new
speech content but rather “refine” existing speech as if it was recorded in ideal conditions (studio-like
quality). From the mathematical point of view, this means that the speech enhancement model should
retrieve the most probable reconstruction of the clean speech ygiven the corrupted version x, i.e.,
y= arg maxy pclean(y|x). -->
语音增强的目标是，给定一个损坏的音频信号 $x$ ，恢复原始录音的语音特征，包括声音、语言内容和韵律。数学上看，语音增强模型应该找到给定 $x$ 的最可能的干净语音 $y$，即 $y=\arg\max_yp_\text{clean}(y|x)$ 。
<!-- This formulation re-considers the probabilistic speech enhancement formulation, which is widely
used in the literature. In such formulation, the speech enhancement model is aimed to capture
the entire conditional distribution pclean(y|x). This formulation might be especially appealing in
situations with high generation ambiguity, e.g., a low SNR scenario where clean speech content
could not be restored unambiguously. In this case, the speech enhancement model could be used
to generate multiple reconstructions, the best of which is then selected by the end user. However,
we note that this formulation might be redundant and not particularly relevant for many practical
applications since the ambiguity in generation can be resolved by more straightforward means such
as conditioning on linguistic content (Koizumi et al., 2023c). -->
但是上面的公式可能对于很多实际应用来说是多余的，因为生成的不确定性可以通过更直接的方式解决，比如基于语言内容的条件生成。
<!-- In practice, for many applications, a more natural way of formalizing speech enhancement is to treat
it as a regression problem which aims at predicting the point of highest probability of the conditional
distribution arg maxy pclean(y|x). This formulation has the advantage of simplifying the task, since
finding the highest mode of the distribution might be significantly easier than learning the entire
distribution. Therefore, the speech enhancement models built for this formulation are likely to be
more efficient after deployment since they solve a simpler task. We note that in the context of speech
enhancement, the speed of inference is always of major concern in practice. -->
在实践中，应该将语音增强看作一个回归问题更，即预测条件分布的最高概率点。找到分布的最高点比学习整个分布容易得多。
<!-- Given this formulation, we argue that the framework of generative adversarial networks (GANs) is
more naturally suited for the speech enhancement problem than diffusion models. We show that
GAN training naturally leads to the mode-seeking behaviour of the generator, which aligns with the
formulation introduced above. Additionally, GANs enjoy one forward pass inference, which is in
contrast to the iterative nature of diffusion models. -->
作者认为 GAN 比 diffusion 模型更适合语音增强。因为 GAN 训练自然地导致生成器的 mode-seeking behaviour，更符合上面的公式。此外，GAN 只需要一次 forward，而不需要迭代。
<!-- Let pg(y|x) be a family of waveform distributions produced by the generator gθ(x). Mao et al. (2017)
showed that training with Least Squares GAN (LS-GAN) leads to the minimization of the Pearson χ2
divergence χ2
Pearson pg
pclean+pg
2 . We propose that if pg(y|x) approaches δ(y−gθ(x)) under some
parametrization, the minimization of this divergence leads to gθ(x) = arg maxy pclean(y|x). This
means that if the generator deterministically predicts the clean waveform from the degraded signal,
the LS-GAN loss encourages the generator to predict the point of maximum pclean(y|x) density. We
note that although prior work by (Li & Farnia, 2023) demonstrated the mode-covering property for
the optimization of Pearson χ2 divergence, our result pertains to a deterministic generator setting,
which is outside the scope of analysis provided by Li & Farnia (2023). -->
令 $p_g(y|x)$ 为由 generator $g_\theta(x)$ 生成的波形分布。已有研究表明，使用 LS-GAN 可以最小化 Pearson $\chi^2$ divergence $\chi^2_\text{Pearson}(p_g || \frac{p_\text{clean} + p_g}{2})$。如果 $p_g(y|x)$ 在某种参数化下接近 $\delta(y-g_\theta(x))$，那么最小化这个 divergence 会导致 $g_\theta(x)=\arg\max_y p_\text{clean}(y|x)$。这意味着如果生成器从损坏的信号中确定性地预测干净的波形，LS-GAN loss 会鼓励生成器预测最大 $p_\text{clean}(y|x)$ 密度的点。

证明略。

<!-- Perceptual Loss for Speech Generation -->
## 用于语音生成的感知损失
<!-- Adversarial training is known for its instability issues (Brock et al., 2018). It often leads to suboptimal
solutions, mode collapse, and gradient explosions. For paired tasks, including speech enhancement,
adversarial losses are often accompanied by additional regressive losses to stabilize training and
guide the generator towards useful solutions (Kong et al., 2020; Su et al., 2020). In the context of
GAN mode-seeking behaviour discussed above, regressive losses could be seen as a merit to push the
generator towards the “right” (most-probable) mode. Therefore, finding an appropriate regression
loss to guide adversarial training is of significant importance. -->
对抗训练不稳定，可能导致 suboptimal 的解、mode collapse 和梯度爆炸。对于语音增强任务，对抗损失通常有额外的回归损失来稳定训练。在 GAN mode-seeking 中，回归损失可以看作是推动生成器走向“正确”（最可能）模式。

<!-- Historically, initial attempts to apply deep learning methods to speech enhancement were based on
treating this problem as a predictive task (Defossez et al., 2020; Hao et al., 2021; Chen et al., 2022a;
Isik et al., 2020). Following the principle of empirical risk minimization, the goal of predictive
modelling is to find a model with minimal average error over the training data. Given a noisy
waveform or spectrogram, these approaches attempt to predict the clean signal by minimizing point-
wise distance in waveform and spectrum domains or jointly in both domains, thus treating this
problem as a predictive task. However, given the severe degradations applied to the signal, there is
an inherent uncertainty in the restoration of the speech signal (i.e., given the degraded signal, the
clean signal is not restored unambiguously), which often leads to oversmoothing (averaging) of the
predicted speech. A similar phenomenon is widely known in computer vision (Ledig et al., 2017). -->
<!-- One promising idea to reduce the averaging effect is to choose an appropriate representation space
for regression, which is less “entangled” than waveform or spectrogram space. In simpler terms, the
regression space should be designed so that averaged representation of sounds that are indistinguish-
able to humans (such as the same phoneme spoken by the same speaker with the same prosody) is
still representation of this sound (see Appendix A.1). -->
之前都是将语音增强问题视为预测任务。给定一个带噪的波形或频谱图，通过在波形和频谱中最小化点间距离或在两个域中最小化来预测干净信号。然而，由于信号的严重退化，对于恢复语音信号存在固有的不确定性，通常导致预测的语音过度平滑。为了减少平滑效应，可以选择一个比波形或频谱空间更少“纠缠”的回归空间。且此空间应该设计为，对于人类无法区分的声音（如相同发音人说的相同音素）的平均表示仍然是这个声音的表示。
<!-- We formulate two heuristic rules to compare different regression spaces based on their structure: -->
提出两个规则来比较不同回归空间的结构：
<!-- Clustering rule: Representations of identical speech sounds should form one cluster that is
separable from clusters formed by different sounds. -->
+ 聚类规则：相同 speech sound 的表征应该形成一个簇，与不同语音形成的簇可分离。
<!-- SNR rule: Representations of speech sounds contaminated by different levels of additive
noise should move away from the cluster of clean sounds monotonically with the increase in
the noise level. -->
+ SNR 规则：受不同水平的加性噪声污染的 speech sound 的表征应该随着噪声水平的增加单调地远离干净声音的簇。
<!-- The clustering rule ensures that minimizing the distance between samples in the feature space causes
the samples to correspond to the same sound. The SNR rule ensures that minimizing the distance
between features does not contaminate the signal with noise, meaning noisy samples are placed
distantly from clean samples. -->
聚类规则确保在特征空间中最小化样本之间的距离会导致样本对应于相同的声音。SNR 规则确保在特征之间最小化距离不会使信号受到噪声污染，即嘈杂的样本与干净的样本相距较远，如图：
![](image/Pasted%20image%2020241015112519.png)

<!-- In practice, we check these conditions by the following procedure: -->
实现上，通过以下步骤检查这些条件：