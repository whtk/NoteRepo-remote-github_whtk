> ICML 2022，MIT-IBM Watson AI Lab
<!-- 翻译 & 理解 -->
<!-- Self-supervised learning (SSL) in speech involves training a speech representation network on a large-scale unannotated speech corpus, and then applying the learned representations to down- stream tasks. Since the majority of the down- stream tasks of SSL learning in speech largely fo- cus on the content information in speech, the most desirable speech representations should be able to disentangle unwanted variations, such as speaker variations, from the content. However, disentan- gling speakers is very challenging, because remov- ing the speaker information could easily result in a loss of content as well, and the damage of the latter usually far outweighs the benefit of the for- mer. In this paper, we propose a new SSL method that can achieve speaker disentanglement without severe loss of content. Our approach is adapted from the HuBERT framework, and incorporates disentangling mechanisms to regularize both the teachers (masked prediction labels) and the stu- dents (learned representations). We evaluate the benefit of speaker disentanglement on a set of content-related downstream tasks, and observe a consistent and notable performance advantage of our speaker-disentangled representations.1 -->
1. 理想的 SSL 需要从 content 中解耦 unwanted variations 如 speaker variations，但是很困难，因为去除 speaker 信息可能会导致内容丢失
2. 提出一种新的 SSL 方法，可以实现 speaker disentanglement 而不严重丢失 content：
    1. 基于 HuBERT 框架，加入 disentangling 机制来规范 teachers 和 students
3. 在一系列 content-related downstream tasks 上评估 speaker disentanglement 的效果，效果很好

> 很神奇的 disentanglement，学生模型学习教师模型，教师模型先去掉一部分的 speaker 信息，然后学生模型通过对比学习来进一步得到 speaker invariant 的表征。最后一步在 predictor 中加入 speaker 信息，这样来进一步确保前面的 SSL 特征不需要很多 speaker 信息，即又可以消除剩下的 speaker 信息。

## Introduction
<!-- While speech SSL has demonstrated advantages in a sur- prisingly wide range of tasks, one of the primary foci of speech SSL is on tasks that process the content of speech, such as speech recognition/phone classification, speech con- tent generation, etc. For these tasks, the most desirable speech representations should be the ones that can disen- tangle content information in speech from other interfering variations, such as speaker variations. However, among the most widely-used existing speech representations, few can achieve a reasonable disentanglement of speaker variations. For example, the HUBERT representation (Hsu et al., 2021) can achieve a speaker identification accuracy of up to 81.4% on the SUPERB benchmark (Yang et al., 2021). This obser- vation suggests that there may still be room for performance gain for SSL on content-related speech processing tasks, if the disentanglement of speaker is adequately addressed. -->
1. speech SSL 主要集中在处理 speech content 的任务上，如 speech recognition/phone classification、speech content generation 等，但是现有 speech representations 很难很好地 disentangle speaker variations
<!-- However, it has been widely acknowledged that disentan- gling speakers is very challenging. Since no text annotations are accessible during the training of the speech representa- tion network, any attempt to remove speaker variations from speech representation could easily lead to a loss of content information (Choi et al., 2021). In most content-related downstream tasks, the cost of losing content information far outweighs the advantage in disentangling speakers. -->
2. disentangling speakers 很困难，因为在训练 speech representation network 时没有文本注释，去除 speaker variations 可能会导致 content 信息丢失
<!-- In this paper, we seek to investigate the following two re- search questions. First, is there a way to disentangle speaker variations during SSL training without significant content loss? To this end, we propose CONTENTVEC, an SSL frame- work that is adapted from the HUBERT training paradigm. The key idea of HUBERT is that by having some relatively poor speech representations, such as MFCC, serve as the teacher labels for the masked prediction task, one can derive speech representations (which are sometimes referred to as students) that are far better than the teachers in many aspects, including content preservation. This inspires us that by combining HUBERT’s teacher-student framework with speaker disentanglement techniques, we could potentially restore the content loss caused by the latter. -->
3. 本文研究以下两个问题：
    1. 是否可以在 SSL 训练中 disentangle speaker variations 而不严重丢失 content
    2. 提出 CONTENTVEC，基于 HuBERT，通过将 HUBERT 的 teacher-student 框架与 speaker disentanglement 结合，恢复后者导致的 content 丢失
<!-- This has led us to the design of CONTENTVEC, which in- corporates into HUBERT three disentangling mechanisms - disentanglement in teachers and disentanglement in stu- dents, and speaker conditioning. Specifically, disentangle- ment in teachers refers to removing the speaker information from the teacher labels. Disentanglement in students refers to introducing a regularization loss that directly enforces speaker invariance on the speech representations. Speaker conditioning refers to inputting speaker information to the masked prediction task, so that the need for the speech rep- resentation to encode speaker information is relieved. As we will show, all three modules are essential in shaping the speaker information flow across the speech representation network layers, and thereby achieve a superior disentangle-
ment quality while keeping the content information intact. -->
4. CONTENTVEC 包含三个 disentangling 机制：
    1. disentanglement in teachers：从 teacher labels 中去除 speaker 信息
    2. disentanglement in students：引入正则化 loss，直接在 speech representations 上强制 speaker invariance
    3. speaker conditioning：在 masked prediction task 中输入 speaker 信息，减轻 speech representation 编码 speaker 信息
<!-- The second research question we would like to explore is: How much performance gain, if any, can speaker disentangl- ment in SSL features contribute to downstream tasks? Our extensive evaluation shows that speaker disentanglement can achieve a consistent performance advantage over the baseline speech representations on content-related applica- tions. The findings of this paper can shed some light on next-generation speech representations that can supply more targeted information to the downstream tasks and enable more powerful content processing directly on speech. -->
5. 第二个问题是：speaker disentanglement 对 downstream tasks 的性能提升有多大
    1. 评估表明，speaker disentanglement 可以在 content-related 应用上取得一致的性能优势

## 相关工作（略）

## 方法

数学标记：
用 $X,\boldsymbol{X}$ 表示随机变量和随机向量，用 $x,\boldsymbol{x}$ 表示确定的标量和向量。
<!-- Denote X = [X1,··· ,XT] as the sequence of a speech features, where Xt is the speech feature vector at frame t, and T is the total number of frames. Our goal is to learn a speech representation network R = f (X ), where R = [R1,··· ,RT ] and Rt is the representation for frame t. R should desirably satisfy the following two properties. -->
设 $\boldsymbol{X}\:=\:\begin{bmatrix}\boldsymbol{X}_1,\cdots,\boldsymbol{X}_T\end{bmatrix}$ 表示 speech features 序列，其中 $\boldsymbol{X}_t$ 是第 $t$ 帧的 speech feature 向量，$T$ 是帧数。目标是学习 speech representation network $\boldsymbol{R}=f(\boldsymbol{X})$，其中 $\boldsymbol{R}=\begin{bmatrix}\boldsymbol{R}_1,\cdots,\boldsymbol{R}_T\end{bmatrix}$，$\boldsymbol{R}_t$ 是第 $t$ 帧的 representation。$R$ 应该满足以下两个性质：
<!-- • R should preserve as much content information as pos- sible, and the content information roughly corresponds to the phonetic/text transcriptions of the utterance.
• R should be invariant across speaker variations. -->
1. $R$ 应尽可能保留 content 信息，content 信息大致对应于 utterance 的音素/文本转录
2. $R$ 应在 speaker variations 下保持不变
<!-- As mentioned, the pursuit of one goal can easily compro- mise another. In the following , we will describe our method to strike a better balance and discuss the rationale behind. -->
<!-- The CONTENTVEC framework builds upon the mask- prediction framework of HUBERT. Specifically, there are three components in the HUBERT framework: 1) the speech representation network f(·), 2) the predictor p(·), and 3) the teacher label generator g(·). -->
CONTENTVEC 框架基于 [HuBERT- Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units 笔记](HuBERT-%20Self-Supervised%20Speech%20Representation%20Learning%20by%20Masked%20Prediction%20of%20Hidden%20Units%20笔记.md) 的 mask-prediction 框架，HuBERT 框架包含三个组件：
+ speech representation network $f(\cdot)$
+ predictor $p(\cdot)$
+ teacher label generator $g(\cdot)$
<!-- During training, the speech representation network takes the partially masked speech utterance, X ̃ , as the input, and produces a representation for the masked speech sequence, R ̃ = f (X ̃ ). On the other hand, the teacher label generator generates a label sequence L = g(X) from the unmasked speech. The goal of the predictor is to predict the teacher labels L from the masked speech representation R ̃. The teacher label generator g(·) is usually predefined and fixed during training. The other two modules, f (·) and p(·), are trained jointly to minimize the following prediction loss: -->
训练时，speech representation network 输入部分 mask 的 speech utterance $\tilde{\boldsymbol{X}}$，产生 masked speech sequence 的 representation $\tilde{\boldsymbol{R}}=f(\tilde{\boldsymbol{X}})$。另一方面，teacher label generator 从 没有 mask 的 speech 生成 label sequence $L=g(\boldsymbol{X})$。predictor 的目标是从 masked speech representation $\tilde{\boldsymbol{R}}$ 预测 teacher labels $L$。teacher label generator $g(\cdot)$ 通常在训练期间预定义并固定。另外两个模块 $f(\cdot)$ 和 $p(\cdot)$ 联合训练，最小化以下预测损失：
$$\mathcal{L}_{pred}=\mathbb{E}[\ell_m(p\circ f(\boldsymbol{\tilde{X}}),g(\boldsymbol{X}))]$$
<!-- where lm denotes the cross-entropy loss computed over the masked frames only. To make our description more intuitive, we will refer to f(X ̃ ) as students, and g(X) as teachers. -->
其中 $\ell_m$ 表示仅在 mask 帧上计算的交叉熵损失。这里将 $f(\boldsymbol{\tilde{X}})$ 称为 students，$g(\boldsymbol{X})$ 称为 teachers。
<!-- It has been reported (Hsu et al., 2021) that even if the HU- BERT teacher is poor (e.g., losing content), the student can still preserve the content far better than the teacher, thanks to the masked prediction mechanism. This observation in- spires us to test the hypothesis that one can combine speaker disentanglement techniques (potentially causing loss of con- tent) with the masked prediction framework, and in this way, preserve content more faithfully than using a speaker disen- tanglement algorithm on its own. Since teachers, students, and the predictor are three major components of the masked prediction, CONTENTVEC introduces three disentanglement mechanisms, disentanglement in teachers, disentanglement in students, and speaker conditioning, to tackle the three components respectively, as shown in Figure 1. -->
已有研究表明，即使 HUBERT teacher 很差（如丢失 content），student 仍然可以比 teacher 保留更多的 content，原因在于 masked prediction。

从而可以将 speaker disentanglement（可能导致 content 丢失）与 masked prediction 结合，比单独使用 speaker disentanglement 更能保留 content。

CONTENTVEC 引入三个 disentanglement，分别处理三个组件，如图：
![](image/Pasted%20image%2020240605102357.png)

### 在 Teachers 中的 Disentanglement
<!-- Disentanglement in teachers aims to remove the speaker information in the teacher labels. Recently, there has been marked progress in unsupervised voice conversion systems, which can now significantly obscure the source speaker in- formation without losing too much content (Polyak et al., 2021). Inspired by this, we adopt a voice conversion model to convert all utterances to the same speaker before generat- ing the teacher labels. -->
Disentanglement in teachers 的目标是从 teacher labels 中去除 speaker 信息。这里采用 VC 模型将所有 utterances 转换为相同说话人，然后生成 teacher labels。
<!-- Specifically, as shown in Figure 1(c), the teacher labels, L = g(X), are generated via the following three steps. First, all the utterances X in the training set are converted to a single speaker using a competent unsupervised voice con- version system. Second, the converted utterances are passed through a pre-trained unsupervised speech representation network, in our case HUBERT, to generate a set of speech representations, which should contain very little speaker in- formation. Finally, the speech representations are quantized to discrete teacher labels using k-means clustering. -->
如上图c所示，teacher labels $L=g(\boldsymbol{X})$ 通过以下三个步骤生成：
+ 训练集中的所有 utterances $\boldsymbol{X}$ 使用一个好的无监督 VC 转换为单个说话人
+ 转换后的 utterances 通过预训练的无监督 speech representation network（HuBERT）生成 speech representations，此时应该包含很少的 speaker 信息
+ speech representations 通过 k-means 聚类量化为 discrete teacher labels

<!-- It is worth noting that although the teacher speech represen- tation described above already achieves speaker disentan- glement, its content preservation is not satisfactory because any voice conversion systems sometimes (for some speak- ers) cause a non-negligible content loss (Choi et al., 2021). In order to ameliorate this shortcoming of modern voice conversion, we use voice conversion as a teacher to train better students, instead of directly applying its output to downstream tasks. -->
> 为什么不直接用这些 teacher representations 作为最终的表征来做 downstream tasks？

尽管上述 teacher speech representation 已经实现了 speaker disentanglement，但其 content 保留不够好。于是使用 VC 作为 teacher 来训练 students，而不是直接将其用于 downstream tasks。

### 在 Students 中的 Disentanglement
<!-- Disentanglement in students enforces speaker-invariant stu- dent representations, which can be achieved with SIMCLR (Chen et al., 2020), a contrastive-learning-based algorithm. -->
Disentanglement in students 使得 student representations 在 speaker 变化下不变，可以通过 SIMCLR 实现。
<!-- Specifically, as shown in Figure 1(a), each speech utterance, X, is passed into two random transformations that alter only the speaker information, before it is masked. Denote the two masked, transformed copies of X as X ̃ (1) and X ̃ (2) . Then, this pair of utterances are passed through the speech repre- setnation network, f (·), to generate the representations R(1) and R(2), and the following contrastive loss is introduced to penalize dissimilarity between R(1) and R(2): -->
如上图 a，每个 speech utterance $\boldsymbol{X}$ 通过两个随机变换，只改变 speaker 信息，然后 mask。记两个 mask、变换后的 $\boldsymbol{X}$ 为 $\boldsymbol{X}^{(1)}$ 和 $\boldsymbol{X}^{(2)}$。然后，这对 utterances 通过 speech representation network $f(\cdot)$ 生成 representations $\boldsymbol{R}^{(1)}$ 和 $\boldsymbol{R}^{(2)}$，引入 contrastive loss 惩罚 $\boldsymbol{R}^{(1)}$ 和 $\boldsymbol{R}^{(2)}$ 之间的不相似性：
$$\begin{aligned}
\mathcal{L}_{contr}& =\sum_{t=1}^T\frac{\exp(\cos\sin(\boldsymbol{R}_t^{(1)},\boldsymbol{R}_t^{(2)})/k)}{\sum_{\tau\in\{t\}\cup\mathcal{I}_t}\exp(\cos\sin(\boldsymbol{R}_t^{(1)},\boldsymbol{R}_\tau^{(1)})/k)}  \\
&+\sum_{t=1}^T\frac{\exp(\cos\sin(\boldsymbol{R}_t^{(2)},\boldsymbol{R}_t^{(1)})/k)}{\sum_{\tau\in\{t\}\cup\mathcal{I}_t}\exp(\cos\sin(\boldsymbol{R}_t^{(2)},\boldsymbol{R}_\tau^{(2)})/k)},
\end{aligned}$$
<!-- where cossim(·, ·) denotes the cosine similarity, and It de-
notes a set of random time indices at which the representa-
tions are chosen as the negative examples for time t. The
contrastive loss consists of two terms so that it is symmetric
with respect to R(1) and R(2). According to Equation (2),
the negative examples for the utterance pair, (R(1) , R(1) ), tt
are uniformly randomly drawn from the remaining frames within the same utterances. As an extention to Equation (2), the contrastive loss can be applied to an intermediate layer, instead of the final layer, of f(·). Section 3.6 will discuss how the choice of layer in which the contrastive loss is imposed would affect the disentanglement behavior. -->
其中 $\cos\sin(\cdot,\cdot)$ 表示余弦相似度，$\mathcal{I}_t$ 表示随机时间索引，用于选择负样本。contrastive loss 有两项，使得对 $R^{(1)}$ 和 $R^{(2)}$ 对称。utterance pair $(R^{(1)},R^{(1)})$ 的负样本是从同一 utterance 的剩余帧中均匀随机选择的。contrastive loss 也可用于 $f(\cdot)$ 的中间层，而不是最后一层。
<!-- The biggest challenge of applying the contrastive loss is how to design a random transformation that only alters the speaker identity of the utterance with minimal changes in the other aspects. To this end, we adopt the random transforma- tion algorithm proposed by Choi et al. (2021). Specifically, the algorithm consists of three steps of transformations. First, all the formant frequencies within an utterance are scaled by a factor of ρ1; second, F0 in every frame is scaled by a factor of ρ2; finally, a random equalizer is applied to accommodate any channel effects. ρ1 and ρ2 are both ran- domly drawn from the uniform distribution U ([1, 1.4]), and then flipped to their reciprocals with probability 0.5. Since the majority of voice information resides in the formant fre- quency and F0 frequency ranges (e.g., (Eide & Gish, 1996)), while content information resides in the relative formant frequency ratios (Stevens, 1987), uniform scaling of all the formant and F0 tends to change the speaker information while retaining the content. -->
应用 contrastive loss 的最大挑战是如何设计一个随机变换，只改变 utterance 的 speaker identity，而对其他方面的改变最小。这里采用 Choi 等人提出的随机变换算法，包括三个步骤的变换：
+ 所有 utterance 中的共振频率按因子 $\rho_1$ 缩放
+ 每帧的 F0 按因子 $\rho_2$ 缩放
+ 采用随机均衡器

其中，$\rho_1$ 和 $\rho_2$ 都从均匀分布 $U([1,1.4])$ 采样，然后以概率 0.5 转为倒数。

> 为什么这么做可以只改变 speaker 信息而保留 content 信息：由于大多数 voice 信息位于共振频率和 F0 频率范围，而 content 信息位于相对共振频率比，所有共振频率和 F0 的均匀缩放倾向于改变 speaker 信息而保留 content。

<!-- To further strengthen the invariance, the same random trans- formations are also applied to the student representations in the masked prediction task, i.e., Equation (1) is modified as -->
为了进一步加强不变性，相同的随机变换也应用于 masked prediction 任务中的 student representations，此时损失变为：
$$\mathcal{L}_{pred}=\mathbb{E}[\ell_m(p\circ f(\boldsymbol{\tilde{X}}^{(1)}),g(\boldsymbol{X}))+\ell_m(p\circ f(\boldsymbol{\tilde{X}}^{(2)}),g(\boldsymbol{X}))].$$
<!-- Again, the masked prediction loss is applied to both f (X ̃ (1) ) and f (X ̃ (2) ) for symmetry. -->
同样，masked prediction loss 应用于 $f(\boldsymbol{\tilde{X}^{(1)}})$ 和 $f(\boldsymbol{\tilde{X}^{(2)}})$ 以保持对称性。

### Speaker Conditioning
<!-- Although disentanglement in teacher can remove the ma- jority of the speaker information from the teacher labels, certain speaker information would remain. As a result, the student representations are undesirably forced to carry the same amount of speaker information as the teachers do in order to reasonably predict the teacher labels. To break this entailment between the speaker information in students and in teachers, we feed the speaker embeddings to the predictor. Speaker embeddings are produced by a speaker embedding network, in our case a pre-trained GE2E (Wan et al., 2018), which takes a speech utterance as input and outputs a vec- tor summarizing the speaker information in the utterance. Therefore, by conditioning the predictor on the speaker em- bedding, we can supply whatever speaker information is needed for the mask prediction task, so that the students do not have to carry the speaker information themselves. -->
尽管 disentanglement in teacher 可以从 teacher labels 中去除大部分 speaker 信息，但仍会保留一些。因此，student representations 不得不携带与 teachers 相同数量的 speaker 信息，以合理预测 teacher labels。为了打破 student 和 teacher 之间的 speaker 信息联系，将 speaker embeddings 输入 predictor。speaker embeddings 由 speaker embedding network 产生，这里是一个预训练的 GE2E 模型，其输入为 speech utterance，输出 speaker 信息。因此，通过在 predictor 上加入 speaker embedding，可以提供 mask prediction 所需的 speaker 信息，使得 students 不必自己带有 speaker 信息。
<!-- Formally, the masked prediction loss now becomes -->
此时 masked prediction loss 变为：
$$\begin{aligned}
\mathcal{L}_{pred}=& \mathbb{E}[\ell_m(p(f(\boldsymbol{\tilde{X}}_1),s(\boldsymbol{X})),g(\boldsymbol{X}))  \\
&+\ell_m(p(f(\boldsymbol{\tilde{X}}_2),s(\boldsymbol{X})),g(\boldsymbol{X}))],
\end{aligned}$$
<!-- where s(X) denotes the speaker embeddings. The final loss is the superposition of the prediction and contrastive losses: -->
其中 $s(\boldsymbol{X})$ 表示 speaker embeddings。最终损失是 prediction 和 contrastive losses 的叠加：
$$\mathcal{L}=\mathcal{L}_{pred}+\lambda\mathcal{L}_{contr}.$$
<!-- As can be observed, although CONTENTVEC requires speaker labels to identify speaker information, speaker la- bels are only used in pre-training the speaker embedding network. The training of CONTENTVEC itself only requires speaker embeddings, not speaker labels. Since the speaker embedding network is pre-trained on a separate dataset, and can well generalize to unseen speakers, the training set for CONTENTVEC does not need to contain any speaker labels. -->
尽管 CONTENTVEC 需要 speaker labels 来识别 speaker 信息，但 speaker labels 仅用于预训练 speaker embedding network。CONTENTVEC 本身的训练只需要 speaker embeddings，不需要 speaker labels。由于 speaker embedding network 在单独的数据集上预训练，并且可以很好地泛化到未见过的说话者，因此 CONTENTVEC 的训练集不需要包含任何 speaker labels。

<!-- An Information Flow Perspective -->
### 信息流视角
<!-- To provide an intuitive illustration of how the aforemen- tioned modules work collaboratively towards disentangling speakers, Figure 2 shows a conceptual curve of how the amount of speaker information changes along different lay- ers of the speech representation network f(·) and the pre- dictor p(·). The vertical axis denotes the amount of speaker information, and the horizontal axis denotes the number of layers. The white area denotes the speech representation net- work layers, and the grey area denotes the prediction layers, which are on top of the speech representation network. To the left, the speaker information is equal to the full speaker information in the input utterance. To the right, the speaker information should be roughly equal to the speaker informa- tion in the teacher labels, which is much lower than that in the input but is still not zero. Due to the information pro- cessing inequality, the speaker information is monotonically decreasing as the layer progresses, except for the predictor layers where speaker information is re-injected. -->
下图显示了 speech representation network $f(\cdot)$ 和 predictor $p(\cdot)$ 不同层中 speaker 信息量曲线：
![](image/Pasted%20image%2020240605111029.png)

纵轴表示 speaker 信息量，横轴表示层数。白色区域表示 speech representation network 层，灰色区域表示在 speech representation network 之上的 prediction 层。左侧，speaker 信息等于输入 utterance 的完整 speaker 信息。右侧，speaker 信息应该大致等于 teacher labels 中的 speaker 信息，远低于输入中的 speaker 信息，但仍不为零。
> speaker 信息随着层的增加单调递减（除 predictor 层，因为重新加入 speaker 信息）。

<!-- As can be observed, there are two places where the speaker information undergoes abrupt changes. The first is where the contrastive loss (Equation (2)) is imposed, and the speaker information is largely reduced. The second is where the speaker information is re-injected, and the speaker informa- tion slightly increases. As a result, the speaker information should reach its minimum at the intersection between the speech representation network and the predictor. Figure 2 shows that all the modules in CONTENTVEC are essential to a successful speaker disentanglement. -->
speaker 信息 突变的两个地方：
+ contrastive loss 作用时，speaker 信息大幅减少
+ speaker 信息重新加入时，speaker 信息略有增加
从而 speaker 信息应在 speech representation network 和 predictor 交点处达到最小。

## 实验