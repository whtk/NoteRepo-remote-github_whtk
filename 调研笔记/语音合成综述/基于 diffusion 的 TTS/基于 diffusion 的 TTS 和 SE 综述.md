> 论文：A Survey on Audio Diffusion Models: Text To Speech Synthesis and Enhancement in Generative AI 

1. diffusion 模型在语音合成中的两个方向：
	1. TTS
	2. 语音增强
2. 本文调研了基于 diffusion 的 语音合成工作
3. 根据采用的 diffusion 模型的阶段分为三类：
	1. 声学模型
	2. 声码器
	3. 端到端
4. 同时也根据是否有特定信号加入或者从原始语音中移除，对各种语音增强任务做了分类

## Introduction

1. TTS 任务可以大致被分为三个阶段：
	1. 早期工作
	2. 统计参数语音合成
	3. 基于神经网络
2. 语音增强的输入和输出都是语音，常见的增强任务包括语音降噪、去混响、speech super-resolution

## 语音合成

### TTS 发展概况
![](image/Pasted%20image%2020230803114331.png)
b 中，通过 vocoder 直接从语言特征中生成波形。如 WaveNet、Parallel Wavenet、DeepVoice 1、 DeepVoice 2、HiFi-GAN
c 中，采用深度声学模型从文本中直接生成 mel 谱。如 DeepVoice 3、TransformerTTS、Fast Speech 1 和 Speech 2

大部分基于 diffusion 的模型都是 c，即先采用声学模型生成声学特征，然后用 vocoder 生成波形。另一种是端到端的方法。

总结如下：
![](image/Pasted%20image%2020230803115001.png)

### 声学模型
把 diffusion 用于 声学模型的模型有：
![](image/Pasted%20image%2020230803162328.png)
Diff-TTS 是第一个把 DDPM 用于 mel 谱生成的，首先一个 text encoder 提取上下文信息，然后通过 length predictor 和 duration predictor 进行对齐，然后采用一个 decoder 基于 DDPM 生成 mel 谱。

DDIM 可以加速采样，Grad-TTS 基于 SDE，采用 WaveGrad 中的 Unet 结构作为 mel 谱生成的 decoder。如果把 波形而非 mel 谱 作为 decoder 的输出，那么就变成了 端到端 的结构。

#### 高效声学模型
通过知识蒸馏进行加速：
+ ProDiff 中指出，先前的模型主要估计数据密度的梯度（gradient-based parameterization），需要成百上千的迭代才可以生成高质量的输出。而 ProDiff 采用 generator-based parameterization，直接估计干净数据；同时提出通过知识蒸馏减少数据方差，ProDiff 是第一个用于交互式、实际应用的语音合成方法
通过 Denoising Diffusion GANs 进行加速：
+ 采用 GAN 建模降噪分布可以得到更大的 step size，从而更小的 step 数
+ DiffGAN-TTS 采用预训练的 GAN 作为声学模型进行加速，同时引入 active shallow diffusion 机制。实验表明，DiffGAN-TTS 只需要一个 step 就可以生成高质量的音频

#### 多说话人情况下的自适应
前面的工作都是通过一个 text-conditioned diffusion 模型来生成 mel 谱，但是都需要 target speaker 的文本。通过在 Grad-TTS 使用 iterative latent variable sampling ， Grad-TTS with ILVR 提出在推理时，混合 latent variable 和目标说话人的声音，可以实现 zero shot 的合成。

Grad-StyleSpeech 将参考语音的 mel 谱编码成 styled vector，然后训练 diffusion 模型时引入这个 vector。

另一种是采用大规模的无文本数据，Guided-TTS 提出一个两阶段的方法，首先用这些数据训练 unconditional 的 DDPM 模型，然后基于 phoneme classifier 和 speaker embedding 的引导生成 mel 谱。而 Guided-TTS 2 采用 speaker-conditional 的 DDPM 模型。采用预训练的 diffusion 模型，针对目标说话人，通过一段短的参考语音 fine tune 预训练的模型。

#### 离散潜空间下的声学模型
很多研究提出，通过  VQ-VAE 将 mel 谱压缩至离散的 token 来提高自回归的生成效率。

Diffsound 采用这种设定，但是提出了基于 diffusion 的 decoder 以非自回归方式来生成 tokens。

NoreSpeech 也在 mel 谱生成中采用了 VQ-VAE，和 Diffsound 不同，NoreSpeech 由于目标是噪声条件下的说话人风格迁移，因此离散的是 style features。

#### 精细化控制
可控情感模型：大多数的方法都需要额外的优化去计算情感强度值（relative attributes rank (RAR)），但是会导致质量的下降。EmoDiff 提出 oft-labeled guidance 技术来直接控制情感强度。具体来说，先训练一个 unconditional 声学模型，然后基于 diffusion 路径训练情感分类器。推理时，音频在情感分类器的 soft label 的指导下输出。

### vocoder
直到 2020 年之前都是自回归模型占主要，自回归模型的生成质量高但是推理速度慢。

最近把 diffusion 用在 vocoder 中的工作总结如下：
![](image/Pasted%20image%2020230803193723.png)

WaveGrad 率先通过估计数据对数密度的梯度将 score matching 和 diffusion 模型组合。具体来说，提出两个变体，基于  discrete refinement step index 和 continuous noise level，发现 continuous 更有效且灵活。WaveGrad 可以在 6 个 refinement step 下生成高质量的合成样本。

DiffWave 的 vocoder 以 mel 谱为 condition，相比于强自回归模型可以实现相竞争的生成质量。在 unconditional 和 class-conditional 的条件下，可以生成 realistic voices 和 consistent word-level pronunciation。

#### 高效 vocoder
DDPM 需要成百上千的采样迭代以生成高质量的语音。改善 noise schedule 可以加速 DDPM。vocoder 的很多工作都采用了不同的 noise schedule 用于训练和推理，如 WaveGrad、DiffWave 等。另一种方法在 时间轴上寻找一系列的 training schedule，但是找到一个短且有效的 schedule 还是很有挑战。

通过额外的网络进行 schedule prediction：为了找到一个更短的 noise schedule 用于采样，有人提出使用额外的 schedule network 来直接预测 schedule。加上原来的 score network，模型被称为 bilateral denoising diffusion model (BDDM)。BDDM 在训练的时候收敛更快。在 vocoder 中，BDDM 可以只用 7 步生成和人类语音无法区别的语音样本，而且很快。

通过联合训练实现高效推理：InferGrad 以一个额外的损失将推理过程引入训练中。具体来说，InferGrad 最小化 GT 样本和 和少量的 inference schedule 迭代后生成的样本之间的差异。

#### 统计视角下的改进
Improvement with noise prior：DDPM 中，高斯噪声被用于 diffusion 过程，从而可以不用计算之前的 step 来采样任意状态。由于两个高斯分布的和还是高斯分布，denosing diffusion gamma models(DDGM) 表明，Gamma 分布也满足这个条件，而且更适合被用作噪声。采用 WaveGrad 中的 noise schedule，DDGM 改善了其生成质量。

另一项工作表明，高斯噪声先验不足以表征所有的样本模式（如 different voiced 和 unvoiced segment），导致真实的数据分布和先验分布之间存在差距。PriorGrad 采用自适应先验来提升 conditional diffusion 模型的性能。具体来说，先基于 conditional 数据计算其均值和方差，在相同的均值和方差下，噪声和数据分布相似。经验表明，PriorGrad 可以显著提高推理速度同时生成高质量的输出（不管是用于 vocoder 还是 声学模型）。

### 端到端框架
1. 部分端到端：也有两个阶段，但是训练的时候是联合训练
2. 全端到端：一个模型直接从文本生成波形而没有声学特征的显式表征。一系列的这类模型采用 adversarial decoder，包括 FastSpeech 2、EATS 、 EFTS-Wav

大多数端到端模型仍然依赖于生成 mel 谱和文本-语音的对齐。Wave-Tacotron 没有 mel 谱，是 flow-based 的方法。

WaveGrad 将 mel 谱转换为波形，而 WaveGrad2 采用端到端的方式，输入为 phoneme 序列，直接生成音频。

Controllable Raw audio synthesis with High-resolution (CRASH) 和 WaveGrad 2 同时提出，用于鼓声合成。基于 SDE，CRASH 采用 noise-conditioned U-net 来 估计 score function，同时引入 class-mixing sampling 来生成混合声音。

全频带生成：之前的工作都是生成 band-limited 音频，DAG 采用端到端的方式生成 full-band 音频。基于 SDE，DAG 引入 encoder-decoder 结构分别进行下采样和上采样。

基于 Itô SDE 的模型：Itôn 基于 Itô SDE 实现端到端的语音合成。其引入一个 dual-denoiser 结构来分别生成 mel 谱和波形。此外，Itôn 还采取两阶段训练策略，先训练 encoder 和 mel denoiser，然后训练 wave denoiser。

## 语音增强
