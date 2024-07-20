> NIPS 2023，NVIDIA，首尔大学

1. 现有的 zero-shot TTS 存在以下几个问题：
	1. 缺乏鲁棒性
	2. 采样速度慢
	3. 依赖于预训练的 codec 表征
2. 提出 P-Flow，快速的、data- efficient zero-shot  TTS，采用 speech prompt 实现说话人自适应，然后用 flow matching generative decoder 实现高质量和快速的语音合成
	1. speech-prompted text encoder 采用 speech prompts 和 text 来生成speaker- conditional 的表征
	2. flow matching generative decoder 用这些表征来合成语音

## Introduction

1. 用 codec 得到的表征，又复杂又耗计算，而且还没有可解释性；本文就采用标准的 mel 谱
2. 同时为了提高推理速度，用最近的 ODE-based 生成模型 —— Flow Matching
3. 本文贡献：
	1. 提出一种 speech prompt 方法，超过了 speaker  embedding，可以提供 in-context learning 的能力
	2. 采用 flow matching 模型提高速度和质量
	3. 可以在更少的训练数据、更少的 encoder 下实现 comparable 的性能

## 相关工作（略）

## 方法

### P-Flow

P-Flow 训练和 mask-autoencoders 类似，给定 <文本，语音> 对，定义语音的 mel 谱 为 $x$，文本为 $c$，$m^p$  为 indicator mask，用于随机 mask 3s 的音频段。定义 $(1-m^p)\cdot x$ ，其中 $p$ 表示这个变量会被替换为任意的 3s 的 prompt。

P-Flow 的训练目标为：给定 $c$ 和 $x^p$ 的条件下重构 $x$，即学习条件概率 $p(x|c,x^p)$。

引入一个 text encoder $f_{enc}$（架构用的是非自回归的 transformer ），输入 文本 $c$ 和随机段 $x^p$ 来生成 speaker- conditional 的 text representation $h_{c}=f_{enc}(x^{p},c)$，然后通过 flow- matching 生成模型将其映射为 mel 谱，最后通过 vocoder 生成语音。

![](image/Pasted%20image%2020231128211250.png)
<!-- AsshowninFig.1a,weinputthemel-spectrogramofarandom segment xp as a speech prompt along with the text input c. We then project both to the same dimensions to use as inputs to the text encoder. The role of the speech-prompted text encoder is to generate a speaker-conditional text representation hc = fenc(xp, c) using the speaker information extracted from the prompt xp. Similar to large-scale codec language models, we employ a non- autoregressive transformer architecture that can attend to speech prompts at arbitrary text positions. -->
如上如，模型输入随机段 $x^p$ 和文本 $c$ 作为 speech prompt，然后将两者投影到相同的维度作为 text encoder 的输入。
<!-- To train the speech-prompted text encoder to effectively extract speaker information from the speech prompt, we use an encoder loss that directly minimizes the distance between the text encoder representation and the mel-spectrogram. In addition to its original purpose in Grad-TTS [30] for reducing sampling steps in a diffusion-based single-speaker TTS model, it also encourages the encoder to incorporate speaker-related details into the generated text representation. -->
为了训练 speech-prompted text encoder，采用了一个 encoder loss 来直接最小化 text encoder representation 和 mel 谱之间的距离。这个 loss 用于减少 sampling 步数，使得 encoder 将 speaker 相关的细节引到生成的 text representation 中。
<!-- As speech-prompted encoder output hc and the mel-spectrogram x have different lengths, we align the text encoder output with the mel-spectrogram using the monotonic alignment search (MAS) algorithm proposed in Glow-TTS [21]. By applying MAS, we derive an alignment A = MAS(hc,x) that minimizes the overall L2 distance between the aligned text encoder output and the mel-spectrogram. Based on the alignment A, we determine the duration d for each text token and expand the encoder output hc by duplicating encoder representations according to the duration of each text token. This alignment process results in text encoder output h that aligns with mel-spectrogram x. The fairly straightforward reconstruction loss is written as Lenc = MSE(h,x). -->
由于 $h_c$ 和 $x$ 的长度不同，采用 MAS 来得到对齐 $A=MAS(h_{c},x)$，最小化对齐后的 text encoder 输出和 mel 谱之间的 L2 距离。根据对齐 $A$，确定每个文本 token 的 duration $d$，然后通过复制 encoder representation 来扩展 encoder 输出 $h_c$。这个对齐过程得到的 text encoder 输出 $h$ 和 mel 谱 $x$ 对齐。重构 loss 定义为 $L_{enc}=MSE(h,x)$。
<!-- In practice, even though the model is not given the exact positioning of xp within x during training, we found the model to still collapse to a trivial copy-pasting of xp. To avoid this, we simply mask out the reconstruction loss for the segment corresponding to xp. Despite this, the final model is still capable of inferring a continuous mel-spectrogram sequence. We define the masked encoder loss Lpenc by using mp for the random segment xp in the mel-spectrogram x: -->
实际用的时候发现，即使 $x^p$ 是随机选的，训练的时候模型会 collapse 为简单复制粘贴 $x^p$，于是把 $x^p$ 这段的 loss mask 掉，此时的 loss 定义为：
$$L_{enc}^p=MSE(h\cdot m^p,x\cdot m^p)$$
> 也就是不计算 mask 部分的 loss
<!-- By minimizing this loss, the encoder is trained to extract speaker information as much as possible from the speech prompt in order to generate an aligned output h that closely resembles the given speech x, which results in enhancing in-context capabilities for speaker adaptation. -->
通过最小化这个 loss，encoder 尽可能多地从 speech prompt 中提取 speaker 信息，以生成一个和给定 speech x 相似的输出 $h$，从而增强 speaker 自适应的 in-context 能力。

<!-- FlowMatchingDecoders: Toperformhigh-qualityandfastzero-shotTTS,weuseaflow-matching generative model as the decoder for modeling the probability distribution p(x|c,xp) = p(x|h). Our flow-matching decoder models the conditional vector field vt(·|h) of Continuous Normalizing Flows (CNF), which represents the conditional mapping from standard normal distribution to data distribution. The decoder is trained using a flow-matching loss Lpcf m that also applies a mask mp for the random segment as in Lpenc. More details will be provided in Section 3.2. -->
为了实现 TTS，采用 flow-matching 生成模型作为 decoder 建模概率分布 $p(x|c,x^p)\:=\:p(x|h)$。decoder 模型建模 CNF 的条件向量场 $v_t(\cdot|h)$，此向量代表从标准正太分布转为数据分布的条件映射。decoder 用的是 flow-matching loss $L_{cfm}$（这里也用了 mask $m^p$）。
<!--  Toreproducetext-tokendurationsduringinferencewhereMASisunavailable, we use a duration predictor trained in a manner similar to [21] . We use the hidden representation of the speech-prompted text encoder as its input without additional speaker conditioning, given this representation already contains speaker information. It is trained simultaneously with the rest of the model with detached inputs to avoid affecting the text-encoder training. The duration predictor estimates the log-scale duration log dˆ for each text token, and the training objective for the duration predictor Ldur is to minimize the mean squared error with respect to log-scale duration log d obtained via MAS during training. -->
然后用的是类似 Glow-TTS 中的 duration predictor，输入为 text encoder 的输出。这个模块用于估计对数 duration $\log\widehat{d}$，其目标函数为，最小化与 MAS 得到的 $\log d$ 之间的 MSE loss。

总的训练损失为 $L=L_{enc}^{p}+L_{cfm}^{p}+L_{dur}$。推理的时候用从参考样本中的随机段作为 prompt，还有文本作为输入。然后用 duration predictor 得到的 $\hat{d}$，拓展之后通过 flow-matching 得到 mel 谱。

### Flow Matching Decoder
<!-- We use Flow Matching to model the mel- spectrogram decoder task’s conditional distribu- tion: p(x|h). We first provide a brief overview of flow matching, followed by describing our sampling procedure and additional qualitative improvements through a guidance-related tech- nique. -->
采用 flow matching 来建模条件分布 $p(x|h)$。
<!-- Flow Matching
[23, 35, 24] is a method for fitting to the time-
dependent probability path between our data
density p1(x) and our simpler sampling density
p0(x) (assumed to be the standard normal). It
is closely related to Continuous Normalizing
Flows, but is trained much more efficiently in
a simulation-free fashion, much like the typical
setup for Diffusion and Score Matching mod-
els [13, 16, 34]. We adopt Conditional Flow
Matching as specified in [23] as their formu-
lation encourages simpler and often straighter
trajectories between source and target distributions. Simpler trajectories allow for test-time sampling in fewer steps without the need for additional distillation. We will ignore the conditional variable h for notational simplicity in this overview. -->
flow matching 用于拟合数据发布 $p_1(x)$ 和简单分布 $p_0(x)$ 之间的时间相关的概率路径。采用 Conditional Flow Matching，其路径更简单直接，从而可以得到更少的采样次数。为了简单，下面分析中忽略条件 $h$。
<!-- Following Lipman et al. [23], we define the flow φ : [0, 1] × Rd → Rd as the mapping between our two density functions using the following ODE: -->
采用下述 ODE 定义 flow $\phi:[0,1]\times\mathbb{R}^d\to\mathbb{R}^d$ 为两个密度函数之间的映射：
$$\frac d{dt}\phi_t(x)=v_t(\phi_t(x));\quad\phi_0(x)=x$$
<!-- Here, vt(x) is the time-dependent vector field and specifying the trajectory of the probability flow through time. vt(x) is also our learnable component, henceforth denoted as vt(φ(x); θ). To sample from the distribution, we sample from the sampling distribution p0 as our initial condition at t = 0 and solve the ODE in Eq. (2). Notably, the formulation [23] encourages straighter trajectories, ultimately allowing us to cheaply approximate the ODE solution with 10 Euler steps with minimal loss in quality. -->
其中，$v_t(x)$ 是时间相关的向量场，表示 probability flow 沿着时间维度的轨迹。$v_t(x)$ 是可学习的，记为 $v_t(\phi(x);\theta)$。推理采样时，从采样分布 $p_0$ 中采样作为初始条件，然后求解 ODE。
<!-- It turns out that determining the marginal flow φt(x) is difficult in practice. Lipman thus formulates it as marginalizing over multiple conditional flows φt,x1 (x) as follows: -->
实际使用时，很难确定 marginal flow $\phi_t(x)$ ，因此将其定义为多个条件 flow 的边缘化：
$$\phi_{t,x_1}(x)=\sigma_t(x_1)x+\mu_t(x_1)$$
<!-- Here, σt(x1) and μt(x1) are time-conditional affine transformations for parameterization the trans- formation between Gaussian distributions p1 and p0. Finally, let q(x1) be the true but likely non- Gaussian distribution over our data. We define p1 as a mixture-of-Gaussian approximation of q by perturbing individual samples with small amounts of white noise with σmin (empirically set to 0.01). We can specify our trajectories without complications from stochasticity as in SDE formulations. -->
其中，$\sigma_t(x_1)$ 和 $\mu_t(x_1)$ 是以时间为条件的仿射变换，用于参数化 Gaussian 分布 $p_1$ 和 $p_0$ 之间的转换。$q(x_1)$ 是实际的数据分布。$p_1$ 是 $q$ 的高斯混合近似，通过给每个样本加入小量的白噪声来实现（$\sigma_{\min}$ 设置为 0.01）。而且可以指定轨迹，而不受 SDE 公式中的随机性的影响。
<!-- Taking advantage of this, Lipman et al. recommend simple linear trajectories, yielding the following parameterization for φt: -->
可以采样简单的线性轨迹，得到如下参数化的 $\phi_t$：
$$\mu_t(x)=tx_1,\:\sigma_t(x)=1-(1-\sigma_{\min})t$$
<!-- Training the vector field is performed using the conditional flow matching objective function: -->
此时采用 conditional flow matching 目标函数训练 vector field：
$$L_{CFM}(\theta)=\mathbb{E}_{t\thicksim U[0,1],x_1\thicksim q(x_1),x_0\thicksim p(x_0)}\|v_t(\phi_{t,x_1}(x_0);\theta)-\frac d{dt}\phi_{t,x_1}(x_0)\|^2$$
<!-- Plugging Eq. (4) in to Eq. (3) and (5), we get our final CFM objective: -->
从而最终的 CFM 目标函数为：
$$L_{CFM}(\theta)=\mathbb{E}_{t,q(x_1),p(x_0)}\|\upsilon_t(\phi_{t,x_1}(x_0);\theta)-(x_1-(1-\sigma_{\min})x_0)\|^2$$
<!-- Recall that our flow matching decoder models the distribution p(x|h). Because h is the output of the text encoder which was provided by the subsegment xp ∈ x, we found it again necessary to mask out the loss for parts of the output corresponding to xp to prevent trivial solutions. Let the generic vt(xt; θ) be parameterized in our setup as vˆθ(xt, h, t) to account for the conditional. Here, t is represented with a continuous sinusoidal embedding. This gives us the masked CFM objective: -->
flow matching decoder 建模 $p(x|h)$，因为 $h$ 是 text encoder 的输出，由子段 $x^p\in x$ 提供，所以需要 mask 掉 $x^p$ 部分的 loss。此时 $v_t(x_t;\theta)$ 参数化为 $\hat{v}_{\theta}(x_t,h,t)$，这里 $t$ 用 continuous sinusoidal embedding 表示。得到 mask CFM 目标函数：
$$L_{CFM}^p(\theta)=\mathbb{E}_{t,q(x_1),p(x_0)}\|m^p\cdot(\hat{v}_\theta(\phi_{t,x_1}(x_0),h,t)-(x_1-(1-\sigma_{\min})x_0))\|^2$$
<!-- Theconditionalflowmatchinglossmarginalizesoverconditionalvectorfieldstoachieve the marginal vector field, the latter of which is used during sampling. While the linearly interpolated conditional trajectories as specified in Eq. (4) do not guarantee the same degree of straightness in the resulting marginal, we still get something fairly close. Within the context of this work, we found the conditional flow matching formulation to result in simple enough trajectories such that it is sufficient to use the first-order Euler’s method with around 10 steps to solve the ODE during inference. Sampling with N Euler steps is performed with the following recurrence relation: -->
conditional flow matching loss 边缘化 conditional vector field 得到 marginal vector field 用于采样。本文作者发现，conditional flow matching 可以得到足够简单的轨迹，因此在推理时用 10 步 Euler 方法求解 ODE。N Euler steps 的采样过程的递推如下：
$$x_0\sim\mathcal{N}(0,I);\quad x_{t+\frac1N}=x_t+\frac1N\hat{v}_\theta(x_t,h,t)$$
<!-- Wefindthatpronunciationclaritycanbefurtherenhancedbyapplyingtechniques from a classifier-free guidance method [14]. In a related work, GLIDE [29] amplifies their text- conditional sampling trajectory by subtracting the trajectory for an empty text sequence. We employ a similar formulation, guiding our sampling trajectory away from the average feature vector computed from h, denoted as h ̄. h ̄ is computed by averaging the expanded representation h along the time axis to obtain a fixed-size vector and then duplicated along the time axis. Let γ be our guidance scale. Our guidance-amplified Euler formulation is as follows: -->
采用 classifier-free guidance 来进一步增强 pronunciation clarity。本文采用 GLIDE 中的方法，通过减去空文本序列的轨迹来放大 text-conditional sampling trajectory。我们采用类似的公式，通过计算平均特征向量 $h$ 来指导采样轨迹，记为 $\bar{h}$。$\bar{h}$ 通过沿时间轴对 $h$ 进行平均得到固定大小的向量，然后沿时间维度重复。$\gamma$ 为 guidance scale。此时  guidance-amplified Euler 公式如下：
$$x_{t+\frac1N}=x_t+\frac1N(\hat{v}_\theta(x_t,h,t)+\gamma(\hat{v}_\theta(x_t,h,t)-\hat{v}_\theta(x_t,\bar{h},t))$$

### 模型细节
<!-- The high-level model architecture is shown in Fig. 1. Our model comprises 3 main components: the prompt-based text encoder, a duration predictor to recover phoneme durations during inference, and a Wavenet-based flow matching decoder. Experiments demonstrate strong zero-shot results despite our text-encoder comprising only a small transformer architecture of 3M parameters. We provide additional architectural details for each component in Section B. -->
模型包括 3 个主要部分：
+ 基于 prompt 的 text encoder
+ duration predictor 
+ 基于 Wavenet 的 flow matching decoder

实验表明，即使 text encoder 只有 3M 参数，也能实现很好的 zero-shot。

## 实验
