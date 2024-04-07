> 高丽大学，2023.7.31 preprint 
<!-- 翻译&理解 -->
<!-- Expressive text-to-speech systems have undergone significant advancements owing to prosody modeling, but con- ventional methods can still be improved. Traditional approaches have relied on the autoregressive method to predict the quantized prosody vector; however, it suffers from the issues of long- term dependency and slow inference. This study proposes a novel approach called DiffProsody in which expressive speech is synthesized using a diffusion-based latent prosody generator and prosody conditional adversarial training. Our findings confirm the effectiveness of our prosody generator in generating a prosody vector. Furthermore, our prosody conditional discriminator sig- nificantly improves the quality of the generated speech by accu- rately emulating prosody. We use denoising diffusion generative adversarial networks to improve the prosody generation speed. Consequently, DiffProsody is capable of generating prosody 16 times faster than the conventional diffusion model. The superior performance of our proposed method has been demonstrated via experiments -->
1. 传统的韵律建模依赖于自回归方法来预测量化的 prosody vector，但是会出现长时依赖问题，而且推理很慢
2. 提出 prosody vector，采用 diffusion-based latent prosody generator 和 prosody conditional adversarial training 来合成 expressive speech
3. 提出的  prosody generator 很有效，且 prosody conditional discriminator 可以通过模仿韵律提高生成的质量
4. 采用 denoising diffusion generative adversarial networks 来提高韵律的生成速度，比传统的生成模型快 16 倍

## Introduction
<!-- RECENT advancements in neural text-to-speech (TTS) models have significantly enhanced the naturalness of synthetic speech. In several studies [1]–[7], prosody modeling has been leveraged to synthesize speech that closely resem- bles human expression. Prosody, which encompasses various speech properties, such as pitch, energy, and duration, plays a crucial role in the synthesis of expressive speech. -->
1. prosody 包含了 pitch、energy 和 duration 等 speech properties，在 expressive speech 的合成中起着至关重要的作用
<!-- In some studies [8], [9], reference encoders have been used to extract prosody vectors for prosody modeling. A global style token (GST) [10] is an unsupervised style modeling method that uses learnable tokens to model and control various styles. Meta-StyleSpeech [11] proposes the application of style vectors extracted using a reference encoder through a style- adaptive layer norm. Progressive variational autoencoder TTS [12] presents a method for gradual style adaptation. A zero- shot method for speech synthesis that comprises the use of a normalization architecture, speaker encoder, and feedforward transformer-based architecture [13] was proposed. Despite the intuitive and effective nature of using a reference encoder, these methods cannot reflect the details of prosody in their synthetic speech without ground-truth (GT) prosody informa- tion. -->
2. 采用 reference encoder 提取 prosody vector 的效果很好，但是在没有 GT prosody 信息的情况下，无法反映语音中的 prosody 细节
<!-- Recently, methods for inferring prosody from text in the absence of reference audio have been developed [14], [15]. FastPitch [16], for instance, synthesizes speech under text and fundamental frequency conditions. FastSpeech 2 [3] aims to generate natural, human-like speech by using extracted prosody features, such as pitch, energy, and duration, through an external tool and introduce a variance adaptor module that predicts these features. Some studies [17], [18] have proposed hierarchical models through the design of prosody features at both coarse and fine-grained levels. However, the separate modeling of prosodic features may yield unnatural results owing to their inherent correlation. -->
3. 独立地建模 prosodic features 可能会导致不自然的结果，因为这些特征之间有内在的相关性
<!-- Some studies have predicted a unified prosody vector, thus enhancing the representation of prosody, given the interde- pendence of prosody features. Text-predicted GST [19] is a method for modeling prosody without reference audio by predicting the weight of the style token from the input text. [20] proposed a method for paragraph-based prosody mod- eling by introducing a paragraph encoder. Gaussian-mixture- model-based phone-level prosody modelling [21] is a method for sampling reference prosody from Gaussian components. [22] proposed a method for modeling prosody using style, perception, and frame-level reconstruction loss. There are also studies in which prosody is modeled using pre-trained language models [23]–[26]. ProsoSpeech [27] models prosody with vector quantization (VQ) using large amounts of data and predicts the index of the codebook using an autoregressive (AR) prosody predictor. However, when predicting prosody vectors, an AR prosody predictor encounters challenges related to long-term dependencies. -->
<!-- To address these issues, we propose DiffProsody, a novel approach that generates expressive speech by employing a diffusion-based latent prosody generator (DLPG) and prosody conditional adversarial training. -->
1. 提出 DiffProsody，采用 diffusion-based latent prosody generator（DLPG）和 prosody conditional adversarial training 来生成 expressive speech
<!-- 
The primary contributions of this work are as follows:
• We propose a diffusion-based latent prosody modeling method that can generate high-quality latent prosody representations, thereby enhancing the expressiveness of synthetic speech. Furthermore, we adopted denoising diffusion generative adversarial networks (DDGANs) to reduce the number of timesteps, resulting in speeds that were 2.48× and 16× faster than those of the AR model and denoising diffusion probabilistic model (DDPM) [28], respectively.
• We propose prosody conditional adversarial training to ensure an accurate reflection of prosody using the TTS module. A significant improvement in smoothness, at- tributable to vector quantization, was observed in the generated speech.
• Objective and subjective evaluations demonstrated that the proposed method outperforms comparative models. -->
2. 贡献包含：
	1. 提出 diffusion-based latent prosody 建模方法，能够生成高质量的 latent prosody representation，从而增强合成语音的表达性，采用 denoising diffusion generative adversarial networks (DDGANs) 来减少 time step 数量，从而加速 AR 和 DDPM
	2. 提出 prosody conditional adversarial training，确保 TTS 模块准确反映 prosody

## 相关工作（略）

## DiffProsody
<!-- The proposed method, called DiffProsody, aims to enhance speech synthesis by incorporating a diffusion-based latent prosody generator (DLPG) and prosody conditional adversarial training. The overall structure and process of DiffProsody are presented in Figure 1. In the first stage, we trained a TTS module and a prosody encoder using a text sequence and a reference Mel-spectrogram as inputs. The prosody conditional discriminator evaluates the prosody vector from the prosody encoder and the Mel-spectrogram from the TTS module to provide feedback on their quality. In the second stage, we train a DLPG to sample a prosody vector that corresponds to the input text and speaker. During inference, the TTS module synthesizes speech without relying on a reference Mel-spectrogram. Instead, it uses the output of a DLPG. This facilitates the generation of expressive speech that accurately reflects the desired prosody. -->
DiffProsody 通过引入 diffusion-based latent prosody generator（DLPG）和 prosody conditional adversarial training 来增强语音合成，整体结构和流程如下图：
![](image/Pasted%20image%2020240407192957.png)

第一阶段，训练 TTS 模块和 prosody encoder，其输入为文本序列和 reference Mel-spectrogram。prosody conditional discriminator 计算来自 prosody encoder 和 Mel-spectrogram 的 prosody vector。

第二阶段，训练 DLPG 来采样与输入文本和 speaker 对应的 prosody vector。

推理时，合成语音过程中不依赖 reference Mel-spectrogram，而是使用 DLPG 的输出，得到指定 prosody 的 expressive speech。

### TTS 模块
<!-- The TTS module is designed to transform text into Mel- spectrograms using speaker and prosody vectors as conditions. The overall structure of the model is presented in Figure 1a. The TTS module comprises a text encoder and a decoder. The text encoder processes the text at both the phoneme and word levels, as illustrated in Figure 1c. The input text, denoted as xtxt , is converted into a text hidden representation htxt , by the phoneme encoder Ep and word encoder Ew. The Ep takes the phoneme-level text xph and the Ew takes as input the word- level text xwd. The htxt is then obtained as the element-wise sum of the outputs of Ep(xph) and Ew(xwd) expanded to the phoneme-level. -->
TTS 模块使用 speaker 和 prosody vectors 作为条件将文本转换为 Mel-spectrograms，如上图 a。包括 text encoder 和 decoder。text encoder 在 phoneme 和 word levels 处理文本，如上图 c。输入文本 $x_{txt}$ 通过 phoneme encoder $E_p$ 和 word encoder $E_w$ 转换为 text hidden representation $h_{txt}$。$E_p$ 输入为 phoneme-level text $x_{ph}$，$E_w$ 输入为 word-level text $x_{wd}$。$h_{txt}$ 为 $E_p(x_{ph})$ 和 $E_w(x_{wd})$ 输出进行 element-wise 求和，然后拓展到  phoneme-level：
$$\mathbf{h}_{txt}=E_p(\mathbf{x}_{ph})+expand(E_w(\mathbf{x}_{wd}))$$
<!-- where expand is an operation that expands the word- level features to the phoneme-level. Obtaining the quantized prosody vector zpros involves using htxt and speaker hidden representation hspk as inputs for the prosody module. In addition, hspk is acquired using a pre-trained speaker encoder. We use Resemblyzer4, an open-source model trained with generalized end-to-end loss (GE2E) [42], to extract hspk. -->
expand 将 word-level features 拓展到 phoneme-level。然后用 $h_{txt}$ 和 speaker hidden representation $h_{spk}$ 输入 prosody module 得到量化的 prosody vector $z_{pros}$。
> $h_{spk}$ 通过预训练的 speaker encoder 获取，使用 Resemblyzer 开源模块提取 $h_{spk}$。

<!-- During the first stage of training, a prosody encoder is employed, which receives the target Mel-spectrogram. In the inference, z′pros is obtained by inputting htxt and hspk into a DLPG, and this is performed without a reference Mel- spectrogram. Finally, the information related to the text, speaker, and prosody is combined by expanding the latent vectors htxt, hspk, and zpros to the phoneme-level and then performing an element-wise summation. -->
第一阶段的训练中，使用 prosody encoder输入 target Mel-spectrogram。

推理时，将 $h_{txt}$ 和 $h_{spk}$ 输入 DLPG 得到 $z^{\prime}_{pros}$（不依赖 reference Mel-spectrogram）。最后，通过对 latent vectors $h_{txt}$、$h_{spk}$ 和 $z_{pros}$ 拓展到 phoneme-level 来组合这些信息，然后进行 element-wise 求和：
$$\mathbf{h}_{total}=\mathbf{h}_{txt}+\mathbf{h}_{spk}+\mathbf{z}_{pros}$$
<!-- The phoneme duration is modeled using the duration pre- dictor DP. The goal of the DP is to predict the phoneme duration at the frame-level based on the input variable htotal. -->
phoneme duration 通过 duration predictor 建模，DP 输入 $h_{total}$ 预测 frame-level 的 phoneme duration：
$$dur'=DP(\mathbf{h}_{total})$$
<!-- In addition, there is a length regulator LR that expands the input variable to the frame-level using the phoneme duration dur. The expanded htotal is then transformed to Mel- spectrogram y′ by Dmel. -->
length regulator LR 使用 phoneme duration $dur$ 将输入拓展到 frame-level。然后通过 $D_{mel}$ 将拓展的 $h_{total}$ 转换为 Mel-spectrogram $y^{\prime}$：
$$\mathbf{y}'=D_{mel}(LR(\mathbf{h}_{total},dur))$$
<!-- For TTS modeling, we use two types of losses: the mean square error (MSE) and structural similarity index (SSIM) loss. These losses aid in accurately modeling the TTS. For the duration modeling, we use the MSE loss.-->
使用了 MSE 和 SSIM loss。对于 duration，使用 MSE loss：
$$\begin{aligned}\mathcal{L}_{rec}&=\mathcal{L}_{MSE}(\mathbf{y},\mathbf{y}^{\prime})+\mathcal{L}_{SSIM}(\mathbf{y},\mathbf{y}^{\prime}).\\\\\mathcal{L}_{dur}&=\mathcal{L}_{MSE}(dur,dur^{\prime}).\end{aligned}$$

### Prosody 模块
<!-- Figure 1b presents the prosody module, which includes a prosody encoder Epros that derives a prosody vector from a reference Mel-spectrogram, a DLPG that produces a prosody vector using text and speaker hidden states, and a codebook Z = {zk}Kk=1 ∈ RK×dz, where K represents the size of the codebook and dz is the dimension of the codes. During the training of Epros, instead of a full-band Mel-spectrogram, we used a low-frequency band Mel-spectrogram to alleviate disentanglement, as in the case of ProsoSpeech [27]. Figure 1d presents the structure of Epros, which comprises two convolutional stacks and a word-level pooling layer. To extract the target prosody, Epros uses the lowest N bins of the target Mel-spectrogram y[0:N], along with the htxt and hspk, as its inputs. The output of this process is a prosody vector, hpros ∈ RL×dz , where L is the word-level length of the input text. -->
上图 b 为 prosody 模块，包含：
+ prosody encoder $E_{pros}$，用来从 reference Mel-spectrogram 中得到 prosody vector
+ DLPG，使用 text 和 speaker hidden states 生成 prosody vector
+ codebook $Z=\{z_k\}_{k=1}^{K}\in\mathbb{R}^{K\times dz}$，$K$ 为 codebook 大小，$dz$ 为 code 的维度。

训练 $E_{pros}$ 时，使用 low-frequency band Mel-spectrogram 代替 full-band Mel-spectrogram，以减轻 disentanglement。$E_{pros}$ 结构如上图 d，包含两个 convolutional stacks 和一个 word-level pooling layer。为了提取目标 prosody，$E_{pros}$ 使用 target Mel-spectrogram $y[0:N]$ 的最低的 $N$ 个 bins，以及 $h_{txt}$ 和 $h_{spk}$ 作为输入。输出为 prosody vector $h_{pros}\in\mathbb{R}^{L\times dz}$，$L$ 为输入文本的 word-level 长度：
$$\mathbf{h}_{pros}=E_{pros}(\mathbf{y}_{[0:N]},\mathbf{h}_{txt},\mathbf{h}_{spk})$$
<!-- During the inference stage, the prosody vector h′pros is obtained using the prosody generator trained in the second stage. -->
推理时，使用第二阶段训练的 prosody generator 得到 prosody vector $h^{\prime}_{pros}$：
$$\mathbf{h}'_{pros}=DLPG(\mathbf{h}_{txt},\mathbf{h}_{spk}).$$
<!-- The DLP G process is described in section III-D. To obtain the discrete prosody token sequence zpros ∈ RL×dz , the vector quantization layer Z maps each prosody vector hipros ∈ Rdz to the nearest element of the codebook entry zk ∈ Rdz . -->
为了得到离散的 prosody token sequence $z_{pros}\in\mathbb{R}^{L\times dz}$，VQ 层 $Z$ 将每个 prosody vector $h_{pros}\in\mathbb{R}^{dz}$ 映射到最近的 codebook entry $z_k\in\mathbb{R}^{dz}$：
$$\mathbf{z}_{pros}^i=\underset{\mathbf{z}_k\in Z}{\operatorname*{\arg\min}}||\mathbf{h}_{pros}^i-\mathbf{z}_k||_2\mathrm{~for~}i=1\mathrm{~to~}L,$$
<!-- where zipros is i-th element of zpros. In the first stage, the TTS module is trained jointly with the codebook Z and prosody encoder Epros. -->
其中 $z_{pros}^i$ 为 $z_{pros}$ 的第 $i$ 个元素。第一阶段，TTS 模块与 codebook $Z$ 和 prosody encoder $E_{pros}$ 联合训练，损失为：
$$\mathcal{L}_{vq}=||sg[\mathbf{h}_{pros}]-\mathbf{z}_{pros}||_2^2+\beta||\mathbf{h}_{pros}-sg[\mathbf{z}_{pros}]||_2^2,$$
<!-- where sg[·] denotes the stop-gradient operation. Moreover, we employ an exponential moving average (EMA) [43] to enhance the learning efficiency by applying it to codebook updates. -->
$sg[·]$ 表示 stop-gradient 算子。用 exponential moving average (EMA) 来提高学习效率，用于 codebook 更新。

### Prosody conditional 对抗学习