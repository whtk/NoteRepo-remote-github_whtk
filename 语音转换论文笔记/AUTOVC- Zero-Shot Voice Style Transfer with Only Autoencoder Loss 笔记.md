> ICML 2019，
<!-- 翻译 & 理解 -->
<!-- 
Non-parallel many-to-many voice conversion, as well as zero-shot voice conversion, remain under- explored areas. Deep style transfer algorithms, such as generative adversarial networks (GAN) and conditional variational autoencoder (CVAE), are being applied as new solutions in this field. However, GAN training is sophisticated and diffi- cult, and there is no strong evidence that its gen- erated speech is of good perceptual quality. On the other hand, CVAE training is simple but does not come with the distribution-matching property of a GAN. In this paper, we propose a new style transfer scheme that involves only an autoencoder with a carefully designed bottleneck. We formally show that this scheme can achieve distribution- matching style transfer by training only on a self- reconstruction loss. Based on this scheme, we proposed AUTOVC, which achieves state-of-the- art results in many-to-many voice conversion with non-parallel data, and which is the first to perform zero-shot voice conversion. -->
1. 提出一种新的 style transfer 方法，只需要一个 autoencoder：
    + 仅使用重构损失就可以实现 distribution matching style transfer
2. 提出 AUTOVC，实现了 many-to-many VC 和 zero-shot voice VC

## Introduction
<!-- Despite the continuing research efforts in voice conversion, three problems remain under-explored. First, most voice conversion systems assume the availability of parallel train-ing data, i.e. speech pairs where the two speakers utter the same sentences. Only a few can be trained on non-parallel data. Second, among the few existing algorithms that work on non-parallel data, even fewer can work for many-to-many conversion, i.e. converting from multiple source speakers to multiple target speakers. Last but not least, no voice conver- sion systems are able to perform zero-shot conversion, i.e. conversion to the voice of an unseen speaker by looking at only a few of his/her utterances. -->
1. VC 中的三个问题：
    + 大多数系统需要平行数据训练
    + 可以使用非平行数据的系统中，很少能实现 many-to-many conversion
    + 没有系统可以实现 zero-shot conversion
<!-- With the recent advances in deep style transfer, the tradi- tional voice conversion problem is being recast as a style transfer problem, where the vocal qualities can be regarded as styles, and speakers as domains. There are many style transfer algorithms that do not require parallel data, and are applicable to multiple domains, so they are readily available as new solutions to voice conversion. In particular, genera- tive adversarial network (GAN) (Goodfellow et al., 2014) and conditional variational autoencoder (CVAE) (Kingma & Welling, 2013; Kingma et al., 2014), are gaining popularity in voice conversion. -->
2. VC 问题为 style transfer 问题：vocal qualities 可以看作 style，speakers 可以看作 domain
<!-- However, neither GAN nor CVAE is perfect. GAN comes with a nice theoretical justification that the generated data would match the distribution of the true data, and has achieved state-of-the-art results, particularly in computer vision. However, it is widely acknowledged that GAN is very hard to train, and its convergence property is fragile. Also, although there is an increasing number of works that introduce GAN to speech generation (Donahue et al., 2018) and speech domain transfer (Pascual et al., 2017; Subakan & Smaragdis, 2018; Fan et al., 2018; Hosseini-Asl et al., 2018), there is no strong evidence that the generated speech sounds real. Speech that is able to fool the discriminators has yet to fool human ears. On the other hand, CVAE is easier to train. All it needs to do is to perform self-reconstruction and maximize a variational lower bound of the output probabil- ity. The intuition is to infer a hypothetical style-independent hidden variable, which is then combined with the new style information to generate the style-transferred output. How- ever, CVAE alone does not guarantee distribution matching, and often suffers from over-smoothing of the conversion output (Kameoka et al., 2018b). -->
3. GAN 和 CVAE 都有问题：
    + GAN 难以训练，收敛性不好
    + CVAE 容易训练，但是不能保证 distribution matching，转换输出过于平滑
<!-- Motivated by this, in this paper, we propose a new style transfer scheme, which involves only a vanilla autoen- coder with a carefully designed bottleneck. Similar to CVAE, the proposed scheme only needs to be trained on the self-reconstruction loss, but it has a distribution matching property similar to GAN’s. This is because the correctly- designed bottleneck will learn to remove the style informa- tion from the source and get the style-independent code, which is the goal of CVAE, but which the training scheme of CVAE is unable to guarantee. -->
4. 作者提出了一种新的 style transfer 方法，只需要一个 autoencoder：
    + 和 CVAE 类似，只需要 self-reconstruction loss 训练
    + 有 distribution matching property，类似 GAN
    + 通过正确设计的 bottleneck，可以学习去除源的 style 信息，得到 style-independent code
<!-- Based on this scheme, we propose AUTOVC, a many-to- many voice style transfer algorithm without parallel data. AUTOVC follows the autoencoder framework and is trained only on autoencoder loss, but it introduces carefully-tuned dimension reduction and temporal downsampling to con- strain the information flow. As we will show, this simple scheme leads to a significant performance gain. AUTOVC achieves superior performance on a traditional many-to- many conversion task, where all the speakers are seen in the training set. Also, equipped with a speaker embedding trained for speaker verification (Heigold et al., 2016; Wan et al., 2018), AUTOVC is among the first to perform zero- shot voice conversion with decent performance. Consider- ing the quality of the results and the simplicity of its training scheme, AUTOVC opens a new path towards a simpler and better voice conversion and general style transfer systems. The implementation will become publicly available. -->
5. 提出 AUTOVC：
    + 无需平行数据的 many-to-many VC 算法
    + 只需要 autoencoder loss 训练
    + 引入 dimension reduction 和 temporal downsampling 约束信息流
    + 在传统 many-to-many conversion 任务上表现优异
    + 通过 speaker embedding，实现 zero-shot voice conversion

## 相关工作（略）

> 本文关注的是没有 text transcriptions 的 VC。

## Style Transfer Autoencoder
<!-- In this section, we will discuss how and why an autoencoder can match the data distribution as GAN does. Although our intended application is voice conversion, the discussion in this section is applicable to other style transfer applications as well. As general mathematical notations, upper-case let- ters, e.g. X, denote random variables/vectors; lower-case letters, e.g. x, denote deterministic values or instances of random variables; X(1 : T) denotes a random process, with (1 : T ) denoting a collection of time indices running from 1 to T . For notational ease, sometimes the time in- dices are omitted to represent the collection of the random process at all times. pX (·|Y ) denotes the probability mass function (PMF) or probability density function (PDF) of X conditionalonY;pX(·|Y =y),orsometimespX(·|y)with- out causing confusions, denotes the PMF/PDF of X con- ditional on Y taking a specific value y; similarly, E[X|Y ], E[X|Y = y] and E[X|y] denote the corresponding condi- tional expectations. It is worth mentioning that E[X|Y ] is still a random, but E[X|Y = y] or E[X|y]. H(·) denotes the entropy, and H(·|·) denotes the conditional entropy. -->
$X$ 表示随机变量/向量，$x$ 表示确定值或随机变量的实例，$X(1:T)$ 表示随机过程，$p_X(\cdot|Y)$ 表示 $X$ 在 $Y$ 条件下的概率质量函数或概率密度函数，$E[X|Y]$ 表示条件期望，$H(\cdot)$ 表示熵，$H(\cdot|\cdot)$ 表示条件熵。
<!-- Assume that speech is generated by the following stochastic process. First, a speaker identity U is a random variable drawn from the speaker population pU (·). Then, a content vector Z = Z(1 : T) is a random process drawn from the joint content distribution pZ(·). Here content refers to the phonetic and prosodic information. Finally, given the speaker identity and content, the speech segment X = X(1 : T) is a random process randomly sampled from the speech distribution, i.e. pX (·|U, Z ), which characterizes the distribution of the speaker U ’s speech uttering the content Z. X(t) can represents a sample of a speech waveform, or a frame of a speech spectrogram. In this paper, we will be working on speech spectrograms. Here, we assume that each speaker produces the same amount of gross information, i.e. -->
假设语音由以下随机过程生成。首先，说话者身份 $U$ 是从说话者总体 $p_U(\cdot)$ 中采样的随机变量。内容向量 $Z = Z(1:T)$ 是从 joint content distribution $p_Z(\cdot)$ 中采样的随机过程。这里的内容指的是语音的语音和韵律信息。给定说话人身份和内容，语音片段 $X = X(1:T)$ 是从语音分布 $p_X(\cdot|U, Z)$ 中采样的随机过程，即 说话人 $U$ 说出内容 $Z$ 的语音分布。$X(t)$ 可以表示语音波形的样本，或频谱图的帧。本文用的是频谱图。假设每个说话人产生相同数量的总信息，即：
$$H(X|U=u)=h_{\mathrm{speech}}=\mathrm{constant},$$
<!-- regardless of u. -->
和 $u$ 无关。
<!-- Now, assume two sets of variables (U1, Z1, X1) and (U2, Z2, X2), are independent and identically distributed (i.i.d.) random samples generated from this process. (U1, Z1, X1) belong to the source speaker and (U2, Z2, X2) belong to the target speaker. Our goal is to design a speech converter that produces the conversion output, Xˆ1→2, which preserves the content in X1, but matches the speaker charac- teritics of speaker U2. Formally, an ideal speech converter should have the following desirable property: -->
假设 $(U_1, Z_1, X_1)$ 和 $(U_2, Z_2, X_2)$ 是从这个过程生成的 iid 的随机样本。$(U_1, Z_1, X_1)$ 属于 source speaekr，$(U_2, Z_2, X_2)$ 属于 target speaker。目标是，设计 speech converter 得到 ${\hat{X}}_{1\to2}$，保留 $X_1$ 中的内容，但匹配说话人 $U_2$ 的特征。理想的 speech converter 要满足：
$$p_{\hat{X}_{1\to2}}(\cdot|U_2=u_2,Z_1=z_1)=p_X(\cdot|U=u_2,Z=z_1)$$
<!-- Eq. (2) means that given the target speaker’s identity U2 = u2 and the content in the source speech Z1 = z1, the con- verted speech should sound like u2 uttering z1. -->
即给定目标说话人的身份 $U_2=u_2$ 和源语音中的内容 $Z_1=z_1$，转换后的语音应该听起来像 $u_2$ 说 $z_1$。
<!-- When U1 and U2 are both seen in the training set, the prob- lem is a standard multi-speaker conversion problem, which has been addressed by some existing works. When U1 or U2 is not included in the training set, the problem becomes the more challenging zero-shot voice conversion problem, which is also a target task of the proposed AUTOVC. -->
当 $U_1$ 和 $U_2$ 都在训练集中时，问题是标准的多说话人转换问题，已经有一些现有的工作解决了这个问题。当 $U_1$ 或 $U_2$ 不在训练集中时，问题变成了 zero-shot voice conversion 问题，即 AUTOVC 的任务。

### Autoencoder 框架
<!-- AUTOVC solves the voice conversion problem with a very simple autoencoder framework, as shown in Fig. 1. The framework consists of three modules, a content encoder Ec(·) that produces a content embedding from speech, a speaker encoder Es(·) that produces a speaker embedding from speech, and a decoder D(·,·) that produce speech from content and speaker embeddings. The inputs to these modules are different for conversion and training. -->
如图：
![](image/Pasted%20image%2020240605180904.png)

包含三个模块：
+ content encoder $E_c(\cdot)$：从语音中产生 content embedding
+ speaker encoder $E_s(\cdot)$：从语音中产生 speaker embedding
+ decoder $D(\cdot,\cdot)$：从 content 和 speaker embeddings 产生语音

<!-- Conversion: As shown in Fig. 1(a), during the actual con- version, the source speech X1 is fed into the content encoder to have content information extracted. The target speech is fed into the speaker encoder to provide target speaker information. The decoder produces the converted speech based on the content information in the source speech and the speaker information in the target speech. -->
转换（图 a）：
+ 源语音 $X_1$ 输入 content encoder 提取内容信息
+ 目标语音输入 speaker encoder 提供目标说话人信息
+ decoder 基于源语音中的内容信息和目标语音中的说话人信息生成转换后的语音

$$C_1=E_c(X_1),\quad S_2=E_s(X_2),\quad\hat{X}_{1\to2}=D(C_1,S_2).$$
<!-- Here C1 and X1→2 are both random processes. S2 is simply a random vector. -->
这里 $C_1$ 和 $\hat{X}_{1\to2}$ 都是随机过程，$S_2$ 是一个随机向量。
<!-- Training: Throughout the paper, we will assume the speaker encoder is already pre-trained to extract some form of speaker dependent embedding, so by training we refer to the training of the content encoder and the decoder. As shown in Fig. 1(b), since we do not assume the availabil- ity of parallel data, only self-reconstruction is needed for training. More specifically, the input to the content encoder is still X1, but the input to the style encoder becomes an utterance from the same speaker U1, denoted as X1′ .1 Then for each input speech X1, AUTOVC learns to reconstruct itself: -->
训练（图 b）：
+ 假设 speaker encoder 已经预训练，提取了说话人 embedding，所以训练指的是 content encoder 和 decoder 的训练