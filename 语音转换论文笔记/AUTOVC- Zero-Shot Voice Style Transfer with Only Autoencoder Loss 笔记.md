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
+ 本文假设 speaker encoder 已经预训练好了，可以提取 speaker embedding
+ 这里的训练指的是 content encoder 和 decoder 的训练
+ 由于训练的时候没有并行数据，所以只要 self-reconstruction

content encoder 的输入仍然是 $X_1$，但是 style encoder 的输入变成了同一个说话人 $U_1$ 的一段话，记为 $X_1^\prime$，对于每个输入语音 $X_1$，AUTOVC 学习重构自己：
$$C_1=E_c(X_1),\quad S_1=E_s(X_1'),\quad\hat{X}_{1\to1}=D(C_1,S_1).$$
<!-- The loss function to minimize is simply the weighted combi- nation of the self-reconstruction error and the content code reconstruction error, i.e. -->
损失函数是 self-reconstruction error 和 content code reconstruction error 的加权组合：
$$\min_{E_c(\cdot),D(\cdot,\cdot)}L=L_\text{recon}+\lambda L_\text{content},$$
其中：
$$\begin{aligned}&L_{\text{recon}}=\mathbb{E}[\|\hat{X}_{1\to1}-X_1\|_2^2],\\&L_{\text{content}}=\mathbb{E}[\|E_c(\hat{X}_{1\to1})-C_1\|_1].\end{aligned}$$
<!-- As it turns out, this simple training scheme is sufficient to produce the ideal distribution-matching voice conversion, as will be shown in the next section. -->
这种简单的方法即可得到好的 distribution-matching voice conversion。

### Why does it work?
<!-- We will formally show this autoencoder-based training scheme is able to achieve ideal voice conversion (Eq. (2)). The secret recipe is to have a proper information bottleneck. We will first state the theoretical guarantee and then present an intuitive explanation. -->
<!-- Theorem 1. Consider the autoencoder framework depicted in Eqs. (3) and (4). Given the following assumption: -->
定理 1. 考虑 autoencoder 框架。给定以下假设：
<!-- 1. The speaker embedding of different utterances of the same speaker is the same. Formally, if U1 = U2, Es(X1) = Es (X2 ). -->
1. 同一个说话人的不同 utterance 的 speaker embedding 是相同的，即 $U_1=U_2$ 时，$E_s(X_1)=E_s(X_2)$。
<!-- The speaker embedding of different speakers is different. Formally, if U1 ̸= U2, Es(X1) ̸= Es(X2). -->
2. 不同说话人的 speaker embedding 是不同的，即 $U_1\neq U_2$ 时，$E_s(X_1)\neq E_s(X_2)$。
<!-- 3. {X1(1 : T)} is an ergodic stationary order-τ Markov process with bounded second moment, i.e. -->
3. ${X_1(1:T)}$ 是一个符合边界二阶矩的遍历平稳的 order-$\tau$ 马尔可夫过程：
$$p_{X_1(t)}(\cdot|X_1(1:t-1),U_1)=p_{X_1(t)}(\cdot|X_1(t-\tau:t-1),U_1).$$
<!-- 4. Denote n as the dimension of C1. Then n = ⌊n∗ +T2/3⌋,
where n∗ is the optimal coding length of pX1 (·|U1)2. -->
4. 记 $n$ 为 $C_1$ 的维度。则 $n=\left\lfloor n^*+T^{2/3} \right\rfloor$，其中 $n^*$ 是 $p_{X_1}(\cdot|U_1)^2$ 的最优编码长度。

<!-- Then the following holds. For each T, there exists a content encoder Ec∗(·;T) and a decoder D∗(·,·;T), s.t. limT→∞ L = 0, and -->
则对于每个 $T$，存在 content encoder $E_c^*(\cdot;T)$ 和 decoder $D^*(\cdot,\cdot;T)$，使得 $\lim_{T\to\infty}L=0$，且：
$$\lim_{T\to\infty}\frac{1}{T}KL\left(p_{\hat{X}_{1\to2}}(\cdot|u_2,z_1)||p_X(\cdot|U=u_2,Z=z_1)\right)=0,$$
<!-- where KL(·||·) denotes the KL-divergence. -->
其中 $KL(\cdot||\cdot)$ 表示 KL 散度。
<!-- The conclusion of Thm. 1 can be interpreted as follows. If
the number of frames T is large enough, and if the bottle- neck dimension n is properly set, then the global optimizer of the loss function in Eq. (5) would approximately satisfy the ideal conversion property in Eq. (2). This conclusion is quite strong, because a major justification of applying GAN to style transfer, despite all its hassles, is that it can ideally match the distribution of the true samples from the target domain. Now Thm. 1 conveys the following message: to achieve the desired distribution matching, an autoencoder is all you need. The formal proof of Thm. 1 will be pre- sented in the appendix. Here, we will present an intuitive explanation, which is also the gist of our proof. The basic idea is that the bottleneck dimension of the content encoder needs to be set such that it is just enough to code the speaker independent information. -->
定理 1 的结论可以解释为：如果帧数 $T$ 足够大，且 bottleneck 维度 $n$ 设置合适，那么 loss function 的全局最优解将近似满足理想转换。
> 定理 1 传达了以下信息：为了实现所需的分布匹配，只需要一个 autoencoder。

而且 content encoder 的 bottleneck 维度需要设置得刚好足够 encode 说话人信息。
<!-- As shown in Fig. 2, speech contains two types of informa- tion: the speaker information (shown as solid color) and the speaker-independent information (shown as striped), which we will refer to as the content information3. Suppose the bottleneck is very wide, as wide as the input speech X1. The most convenient way to do self-reconstruction is to copy X1 as is to the content embedding C1, and this will guarantee a perfect reconstruction. However as the dimension of C1 decreases, C1 is forced to lose some information. Since the autoencoder attempts to achieve perfect reconstruction, it will choose to lose speaker information because the speaker information is already supplied in S1. In this case, perfect reconstruction is still possible, but the C1 may contain some speaker information, as shown in Fig. 2(a). -->
如图：
![](image/Pasted%20image%2020240606104607.png)

语音包含两种信息：说话人信息（实色）和说话人无关信息（条纹）即内容信息：
+ 假设 bottleneck 很宽（图 a），和输入语音 $X_1$ 一样宽。最方便的自重构方式是将 $X_1$ 直接复制到 content embedding $C_1$，这样可以保证完美重构。但是随着 $C_1$ 维度的减小，$C_1$ 被迫丢失一些信息。由于 autoencoder 试图实现完美重构，它会选择丢失说话人信息，因为说话人信息已经在 $S_1$ 中提供。在这种情况下，完美重构仍然是可能的，但 $C_1$ 可能包含一些说话人信息。
<!-- On the other hand, if the bottleneck is very narrow, then the content encoder will be forced to lose so much information that not only the speaker information but also the content information is lost. In this case, the perfect reconstruction is impossible, as shown in Fig. 2(b). -->
+ 如果 bottleneck 很窄（图 b），那么 content encoder 将被迫丢失太多信息，不仅说话人信息，还有内容信息也会丢失。在这种情况下，完美重构是不可能的
<!-- Therefore, as shown in Fig. 2(c), when the dimension of C1 is chosen such that the dimension reduction is just enough to get rid of all the speaker information but no content infor- mation is harmed, we have reached our desirable condition, under which two important properties hold: -->
+ 当 $C_1$ 的维度被选择得刚好足够去除所有说话人信息但不损害内容信息时（图 c），达到最优条件，满足两个性质：
<!-- 1. Perfect reconstruction is achieved. -->
    + 完美重构
<!-- 2. The content embedding C1 does not contain any infor- mation about the source speaker U1, which we refer to as speaker disentanglement. -->
    + content embedding $C_1$ 不包含任何关于源说话人 $U_1$ 的信息，即 speaker disentanglement
<!-- We will now show by contradiction how these two properties imply an ideal conversion. Suppose when AUTOVC is performing an actual conversion (source and target speakers are different), the quality is low, or does not sound like the target speaker at all. By property 1, we know that the reconstruction (source and target speakers are the same) quality is high. However, according to Eq. (3), the output speech Xˆ1→2 can only access C1 and S2, both of which do not contain any information of the source speaker U1. -->

> 反证法：假设当 AUTOVC 进行实际转换（源和目标说话人不同）时，质量很低，或者根本不像目标说话人。根据性质 1，我们知道重构（源和目标说话人相同）质量很高。然而，输出语音 $\hat{X}_{1\to2}$ 只能访问 $C_1$ 和 $S_2$，两者都不包含源说话人 $U_1$ 的任何信息。
<!-- In other words, from the conversion output, one can never tell if it is produced by self-reconstruction or conversion, as shown in Fig. 2(d). If the conversion quality is low, but the reconstruction quality is high, one will be able to distinguish between conversion and reconstruction above chance, which leads to a contradiction. -->
> 换句话说，从转换输出中，永远无法判断是由自重构还是转换产生的。如果转换质量低，但重构质量高，人们将能够在一定程度上区分转换和重构，这导致了矛盾。

## AUTOVC 架构
<!-- As shown in Fig. 3, AUTOVC consists of three major mod- ules: a speaker encoder, a content encoder, a decoder. AU- TOVC works on the speech mel-spectrogram of size N-by- T , where N is the number of mel-frequency bins and T is the number of time steps (frames). A spectrogram inverter is introduced to convert the output mel-spectrogram back to the waveform, which will also be detailed in this section. -->
如图：
![](image/Pasted%20image%2020240606110631.png)

包含三个模块：
+ speaker encoder
+ content encoder
+ decoder

工作在大小为 $N\times T$ 的 speech mel-spectrogram 上，$N$ 是 mel 频率 bin 的数量，$T$ 是时间步（帧）的数量。引入了一个 spectrogram inverter，将输出 mel-spectrogram 转换回波形。
<!-- According to assumptions 1 and 2 in Thm. 1, the goal of the speaker encoder is to produce the same embedding for different utterances of the same speaker, and different em- beddings for different speakers. For conventional many-to- many voice conversion, the one-hot encoding of speaker identities suffices. However, in order to perform zero-shot conversion, we need to apply an embedding that is general- izable to unseen speakers. Therefore, inspired by (Jia et al., 2018), we follow the design in (Wan et al., 2018). As shown in Fig. (3)(b), the speaker encoder consists of a stack of two LSTM layers with cell size 768. Only the output of the last time is selected and projected down to dimension 256 with a fully connected layer. The resulting speaker embedding is a 256-by-1 vector. The speaker encoder is pre-trained on the GE2E loss (Wan et al., 2018) (the softmax loss version), which maximizes the embedding similarity among different utterances of the same speaker, and minimizes the similarity among different speakers. Therefore, it is very consistent with assumptions 1 and 2 in Thm. 1. -->
如图，speaker encoder 包含两个 cell size 为 768 的 LSTM 层。只选择最后一个时间的输出，并通过一个全连接层将其投影到维度 256。得到的 speaker embedding 是一个 $256\times 1$ 的向量。speaker encoder 在 GE2E loss 上预训练（softmax loss 版本），该 loss 最大化同一个说话人的不同 utterance 之间的嵌入相似性，并最小化不同说话人之间的相似性。因此，它与定理 1 的假设 1 和 2 非常一致。
<!-- In our implementation, the speaker encoder is pre-trained on the combination of VoxCeleb1 (Nagrani et al., 2017) and Librispeech (Panayotov et al., 2015) corpora, where there are a total of 3549 speakers. -->
本文的 speaker encoder 在 VoxCeleb1 和 Librispeech 语料上预训练。
<!-- As shown in Fig. 3(a), the input to the content encoder is the 80-dimensional mel-spectrogram of X1 concatenated with the speaker embedding, Es(X1), at each time step. The concatenated features are fed into three 5 × 1 convolutional layers, each followed by batch normalization and ReLU activation. The number of channels is 512. The output then passes to a stack of two bidirectional LSTM layers. Both the forward and backward cell dimensions are 32, so their combined dimension is 64. -->
如图，content encoder 的输入是 $X_1$ 的 80 维 mel-spectrogram 和 speaker embedding $E_s(X_1)$ 在每个时间步上的拼接。拼接特征输入到三个 $5\times 1$ 的卷积层，每个后面跟着 batch normalization 和 ReLU 激活。通道数为 512。输出传递到两个双向 LSTM 层。正向和反向 cell 的维度都是 32，所以它们的维度是 64。
<!-- As a key step of constructing the information bottleneck, both the forward and backward outputs of the bidirectional LSTM are downsampled by 32. The downsampling is per- formed differently for the forward and backward paths. For the forward output, the time steps {0, 32, 64, · · · } are kept; for the backward output, the time steps {31, 63, 95, · · · } are kept. Figs. 3(e) and (f) also demonstrate how the down- sampling is performed (for the ease of demonstration, the downsampling factor is set to 3). The resulting content embedding is a set of two 32-by-T /32 matrices, which we will denote C1→ and C1← respectively. The downsampling can be regarded as dimension reduction along the temporal axis, which, together with the dimension reduction along the channel axis, constructs the information bottleneck. -->
构建信息 bottleneck 的关键步骤是，双向 LSTM 的正向和反向输出都进行 32 倍下采样。对于正向输出，保留时间步 {0, 32, 64, ...}；对于反向输出，保留时间步 {31, 63, 95, ...}。下采样的过程如图所示。得到的 content embedding 是两个 $32\times T/32$ 的矩阵，分别记为 $C_1\to$ 和 $C_1\leftarrow$。下采样可以看作是沿时间轴的维度减少，与沿通道轴的维度减少一起构建了信息 bottleneck。
<!-- The architecture of the decoder is inspired by (Shen et al., 2018); and is shown in Fig. 3(c). First, the content and speaker embeddings are both upsampled by copying to re- store to the original temporal resolution. Formally, denotes the upsampled features as U→ and U← respectively. Then -->
decoder 如图，首先，content 和 speaker embeddings 都通过复制进行上采样，恢复到原始时间分辨率。形式上，分别表示为 $U_\to$ 和 $U_\leftarrow$。然后：
$$U_\to(:,t)=C_{1\to}(:,\lfloor t/32\rfloor)\\U_\leftarrow(:,t)=C_{1\leftarrow}(:,\lfloor t/32\rfloor),$$
<!-- where (:, t) denotes indexing the t-th column. Figs. 3(e) and (f) also demonstrate the copying. The underlying intuition is that each embedding at each time step should contain both past and future information. For the speaker embedding, simply copy the vector T times. -->
其中 $(:,t)$ 表示索引第 $t$ 列。基本思想是，每个时间步的每个 embedding 都应该包含过去和未来的信息。对于 speaker embedding，简单地复制向量 $T$ 次。
<!-- Then, the upsampled embeddings are concatenated and fed into three 5×1 convolutional layers with 512 channels, each followed by batch normalization and ReLU, and then three LSTM layers with cell dimension 1024. The outputs of the LSTM layer are projected to dimension 80 with a 1 × 1 convolutional layer. This projection output is the initial estimate of the converted speech, denoted as X ̃1→2. -->
然后，拼接上采样的 embeddings，输入到三个 512 通道的 $5\times 1$ 卷积层，每个后面跟着 batch normalization 和 ReLU，然后是三个 cell 维度为 1024 的 LSTM 层。LSTM 层的输出通过一个 $1\times 1$ 卷积层投影到维度 80。这个投影输出是转换语音的初始估计，记为 $\tilde{X}_{1\to2}$。
<!-- In order to construct the fine details of the spectrogram better on top of the initial estimate, we introduce a post- network after the initial estimate, as introduced in Shen et al. (2018). The post network consists of five 5×1 convolutional layers, where batch normalization and hyperbolic tangent are applied to the first four layers. The channel dimension for the first four layers is 512, and goes down to 80 in the final layer. We will refer to the output of the post- network as the residual signal, denoted as R1→2. The final conversion result is produced by adding the residual to the initial estimate, i.e. -->
为了更好地构建 spectrogram 的细节，我们在初始估计之后引入一个 post-network，由五个 $5\times 1$ 卷积层组成，前四层应用 batch normalization 和双曲正切。前四层的通道维度为 512，最后一层降到 80。我们将 post-network 的输出称为残差信号，记为 $R_{1\to2}$。最终的转换结果是将残差添加到初始估计中：
$$\hat{X}_{1\to2}=\tilde{X}_{1\to2}+R_{1\to2}.$$
<!-- During training, reconstruction loss is applied to both the ini- tial and final reconstruction results. Formally, in addition to the loss specified in Eq. (5), we add an initial reconstruction loss defined as -->
训练时，重构损失应用于初始和最终重构结果。初始重构损失为：
$$\large L_{\text{recon}0}=\mathbb{E}[\|\tilde{X}_{1\to1}-X_1\|_2^2],$$
<!-- where X ̃1→1 is the reciprocal of X ̃1→2 in the reconstruction case, i.e. when U2 = U1. The total loss becomes -->
其中 $\tilde{X}_{1\to1}$ 是重构情况下 $\tilde{X}_{1\to2}$ 的逆，即当 $U_2=U_1$ 时。总损失为：
$$\large\min_{E_c(\cdot),D(\cdot,\cdot)}L=L_{\mathrm{recon}}+\mu L_{\mathrm{recon}0}+\lambda L_{\mathrm{content}}.$$
<!-- We apply the WaveNet vocoder as introduced in Van Den Oord et al. (2016), which consists of four deconvo- lution layers. In our implementation, the frame rate of the mel-spetrogram is 62.5 Hz and the sampling rate of speech waveform is 16 kHz. So the deconvolution layers will up- sample the spectrogram to match the sampling rate of the speech waveform. Then, a standard 40-layer WaveNet con- ditioning upon the upsampled spectrogram is applied to generate the speech waveform. We pre-trained the WaveNet vocoder using the method described in Shen et al. (2018) on the VCTK corpus. -->
用 WaveNet vocoder，包含四个反卷积层。具体实现中，mel-spetrogram 的帧率是 62.5 Hz，语音波形的采样率是 16 kHz。因此，反卷积层将 mel-spetrogram 上采样以匹配语音波形的采样率。然后，应用了一个标准的 40 层 WaveNet，根据上采样的 spectrogram 生成语音波形。我们在 VCTK 上预训练 WaveNet vocoder。

## 实验（略）
