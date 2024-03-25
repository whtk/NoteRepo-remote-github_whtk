> TASLP 2019，NII，Xin Wang

<!-- 翻译&理解 -->
<!-- Neural waveform models have demonstrated better performance than conventional vocoders for statistical paramet- ric speech synthesis. One of the best models, called WaveNet, uses an autoregressive (AR) approach to model the distribution of waveform sampling points, but it has to generate a waveform in a time-consuming sequential manner. Some new models that use inverse-autoregressive flow (IAF) can generate a whole waveform in a one-shot manner but require either a larger amount of training time or a complicated model architecture plus a blend of training criteria.-->
1. 神经模型 vocoder 比传统的 vocoder 效果好，但是很慢
<!-- As an alternative to AR and IAF-based frameworks, we pro- pose a neural source-filter (NSF) waveform modeling framework that is straightforward to train and fast to generate waveforms. This framework requires three components to generate wave- forms: a source module that generates a sine-based signal as excitation, a non-AR dilated-convolution-based filter module that transforms the excitation into a waveform, and a conditional module that pre-processes the input acoustic features for the source and filter modules. This framework minimizes spectral- amplitude distances for model training, which can be efficiently implemented using short-time Fourier transform routines. As an initial NSF study, we designed three NSF models under the proposed framework and compared them with WaveNet using our deep learning toolkit. It was demonstrated that the NSF models generated waveforms at least 100 times faster than our WaveNet-vocoder, and the quality of the synthetic speech from the best NSF model was comparable to that from WaveNet on a single-speaker Japanese speech corpus. -->
2. 提出 neural source-filter (NSF) 框架，可以直接训练、快速合成，包含三个组件：
    1. source module 生成 sine-based 信号作为激励
    2. non-AR dilated-convolution-based filter module 将激励转换为波形
    3. conditional module 预处理输入的声学特征
3. 模型最小化频谱幅度之间的距离进行训练，其可以使用短时傅里叶变换高效实现
4. 设计了三个 NSF 模型，并使用 deep learning toolkit 进行了比较，结果表明 NSF 模型比 WaveNet-vocoder 快 100 倍，且质量相当

## Introduction

<!-- In this paper, we propose a neural source-filter (NSF) waveform modeling framework, which is straightforward to implement, fast in generation, and effective in producing high- quality speech waveforms for TTS. NSF models designed under this framework have three components: a source module that produces a sine-based excitation signal, a filter module that uses a dilated-convolution-based network to convert the excitation into a waveform, and a conditional module that pro- cesses the acoustic features for the source and filter modules. The NSF models do not rely on AR or IAF approaches and are directly trained by minimizing the spectral amplitude distances between the generated and natural waveforms. -->
1. 提出 neural source-filter (NSF) 框架，可以直接实现、快速生成、高质量的语音波形，包含三个组件：
    1. source module 生成 sine-based 激励信号
    2. filter module 使用 dilated-convolution-based 网络将激励转换为波形
    3. conditional module 为 source 和 filter 模块处理声学特征
<!-- We describe the three specific NSF models designed under our proposed framework: a baseline NSF (b-NSF) model that adopts a WaveNet-like filter module, simplified NSF (s-NSF) model that simplifies the filter module of the baseline NSF model, and harmonic-plus-noise NSF (hn-NSF) model that uses separate source-filter pairs for the harmonic and noise components of waveforms. While we previously introduced the b-NSF model [20], the s-NSF and hn-NSF models are newly introduced in this paper. Among the three NSF models, the hn-NSF model outperformed the other two in a large-scale listening test. Compared with WaveNet, the hn-NSF model generated speech waveforms with comparably good quality but much faster. -->
2. 提出了三个 NSF 模型：
    1. baseline NSF (b-NSF) 模型，采用类似 WaveNet 的 filter module
    2. simplified NSF (s-NSF) 模型，简化了 baseline NSF 模型的 filter module
    3. harmonic-plus-noise NSF (hn-NSF) 模型，对谐波分量和噪声分量使用独立的 source-filter

## 神经波形模型回顾（略）

## Neural Source-Filter 模型

![](image/Pasted%20image%2020240324214846.png)
<!-- We propose a framework of neural waveform modeling that is fast in generation and straightforward in implementation. As Figure 1 illustrates, our proposed framework contains three modules: a condition module that processes input acoustic features, source module that produces a sine-based excitation signal given the F0, and neural filter module that converts the excitation into a waveform using dilated convolution (CONV) and feedforward transformation. Rather than the MSE or the likelihood over waveform sampling points, the proposed framework uses spectral-domain distances for model training. Because our proposed framework explicitly uses a source-filter structure, we refer to all of the models based on the framework as NSF models. -->
如图，包含三个模块：
1. condition module 处理输入的声学特征
2. source module 根据 F0 生成 sine-based 激励信号
3. neural filter module 使用 dilated convolution 和 feedforward 将激励转换为波形
<!-- Our proposed NSF framework does not rely on AR or IAF approaches. An NSF model converts an excitation signal into an output waveform without sequential transformation. It can be simply trained using the stochastic gradient descent (SGD) method under a spectral domain criterion. Therefore, its time complexity is theoretically irrelevant to the waveform length, i.e., O(1) in both the model training and waveform generation stages. Neither does an NSF model use knowledge distilling, which makes it straightforward in training and implementation. -->
好处在于：
1. 不依赖 AR 或 IAF 方法
2. 可以使用 SGD 在频域下训练
3. 时间复杂度与波形长度无关，训练和生成波形的时间复杂度都是 O(1)

<!-- A. Training criterion based on spectral amplitude distances -->
### 基于频谱幅度距离的训练测度
<!-- A good metric L(ob1:T , o1:T ) ∈ R≥0 should measure the
distance between the perceptual quality of a generated wave-
formob1:T andthatofanaturalwaveformo1:T.Additionally,a
gradient ∂L based on the metric should be easily calculated ∂ ob1:T
so that a model can be trained using the SGD method. For the NSF models, we use spectral amplitude distances calculated on the basis of short-time Fourier transform (STFT). Although spectral amplitude distance has been used in classical speech coding methods [30], [31], [32], we further compute multiple distances with different short-time analysis configurations, as illustrated in Figure 2. -->
一个好的测度 $\mathcal{L}(\widehat{\boldsymbol{o}}_{1:T},\boldsymbol{o}_{1:T})\:\in\:\mathbb{R}_{\geq0}$ 应该衡量生成波形 $\widehat{\boldsymbol{o}}_{1:T}$ 和自然波形 $\boldsymbol{o}_{1:T}$ 的感知质量之间的距离。且梯度 $\frac{\partial\mathcal{L}}{\partial\widehat{\boldsymbol{o}}_{1:T}}$ 容易计算。对于 NSF 模型，使用基于短时傅里叶变换（STFT）计算的频谱幅度距离如图：
![](image/Pasted%20image%2020240325102900.png)

<!-- Given the natural and generated waveforms,
we follow the STFT convention and conduct framing and win-
dowing on the waveforms, after which the spectrum of each
frame is computed using the discrete Fourier transform (DFT). -->
对于自然和生成的波形，先进行 framing 和 windowing，然后使用离散傅里叶变换（DFT）计算每个 frame 的频谱。定义 $\begin{aligned}\widehat{\boldsymbol{x}}^{(n)}=[\widehat{x}_1^{(n)},\cdots,\widehat{x}_M^{(n)}]^\top\in\mathbb{R}^M\end{aligned}$ 为生成波形 $\widehat{\boldsymbol{o}}_{1:T}$ 的第 $n$ 帧，其中 $M$ 为帧长，$\widehat{x}_m^{(n)}$ 为加窗后的波形。然后定义 $\begin{aligned}\widehat{\boldsymbol{y}}^{(n)}=[\widehat{y}_1^{(n)},\cdots,\widehat{y}_K^{(n)}]^\top\in\mathbb{C}^K\end{aligned}$ 为 $\widehat{\boldsymbol{x}}^{(n)}$ 的频谱（采用 $K$ 点 DFT）。同理定义 $\boldsymbol{x}^{(n)}$ 和 $\boldsymbol{y}^{(n)}$ 为自然波形的第 $n$ 帧和频谱（假定波形被分为 $N$ 帧）。此时对数幅度谱距离计算为：
$$\mathcal{L}=\frac1{2NK}\sum_{n=1}^N\sum_{k=1}^K\left[\log\frac{\operatorname{Re}(y_k^{(n)})^2+\operatorname{Im}(y_k^{(n)})^2+\eta}{\operatorname{Re}(\widehat{y}_k^{(n)})^2+\operatorname{Im}(\widehat{y}_k^{(n)})^2+\eta}\right]^2,$$

模型训练的时候，需要计算梯度向量 $\frac{\partial\mathcal{L}}{\partial\widehat{\boldsymbol{o}}_{1:T}}\in\mathbb{R}^T$ 然后反向传播到 neural filter module 中，下面对此梯度的计算进行分解。

由于 $\begin{aligned}\widehat{\boldsymbol{y}}^{(n)}=[\widehat{y}_1^{(n)},\cdots,\widehat{y}_K^{(n)}]^\top\in\mathbb{C}^K\end{aligned}$ 为 $\widehat{\boldsymbol{x}}^{(n)}$ 的频谱，有：
$$\begin{gathered}
\operatorname{Re}(\widehat{y}_k^{(n)}) \begin{aligned}=\sum_{m=1}^M\widehat{x}_m^{(n)}\cos\left(\frac{2\pi}K(k-1)(m-1)\right),\end{aligned} \\
\operatorname{Im}(\widehat{y}_k^{(n)}) =-\sum_{m=1}^M\widehat{x}_m^{(n)}\sin\left(\frac{2\pi}K(k-1)(m-1)\right), 
\end{gathered}$$
其中 $k=1,\cdots,K$，由于 $\mathrm{Re}(\widehat{y}_k^{(n)})\text{, Im}(\widehat{y}_k^{(n)})$ 和 $\frac{\partial\mathcal{L}}{\partial\text{Re}(\widehat{y}_k^{(n)})},\frac{\partial\mathcal{L}}{\partial\text{Im}(\widehat{y}_k^{(n)})}$ 为实数，所以可以直接计算梯度：
$$\begin{gathered}
\frac{\partial\mathcal{L}}{\partial\widehat{x}_m^{(n)}} =\sum_{k=1}^K\left[\frac{\partial\mathcal{L}}{\partial\mathrm{Re}(\widehat{y}_k^{(n)})}\frac{\partial\mathrm{Re}(\widehat{y}_k^{(n)})}{\partial\widehat{x}_m^{(n)}}+\frac{\partial\mathcal{L}}{\partial\mathrm{Im}(\widehat{y}_k^{(n)})}\frac{\partial\mathrm{Im}(\widehat{y}_k^{(n)})}{\partial\widehat{x}_m^{(n)}}\right] \\
=\sum_{k=1}^K\frac{\partial\mathcal{L}}{\partial\mathrm{Re}(\widehat{y}_k^{(n)})}\cos(\frac{2\pi}K(k-1)(m-1))- \\
\sum_{k=1}^K\frac{\partial\mathcal{L}}{\partial\text{エm}(\widehat{y}_k^{(n)})}\sin(\frac{2\pi}K(k-1)(m-1)). 
\end{gathered}$$
<!-- As long as we can compute ∂L for each m and n, the for t ∈ {1, · · · , T } can be easily accumulated from ∂L given the relationship between o and each xbm-->
一旦计算了对于每个 $m,n$ 的梯度 $\frac{\partial\mathcal{L}}{\partial\widehat{x}_m^{(n)}}$ , 就可以根据 $\boldsymbol{o}_t$ 和每个 $\widehat{x}_m^{(n)}$ 之间的关系容易地累积得到 $\frac{\partial\mathcal{L}}{\partial\widehat{\boldsymbol{o}}_{1:T}}$。

<!-- The pre-
vious explanation shows that L ∈ R and ∂L ∈ RT ∂ ob1:T
can be computed no matter how we configure the window length, frame shift, and DFT bins, i.e., the values of N, M , K . It is thus straightforward to merge multiple distances {L1 , · · · , LS } with different windowing and framing configu- rations. In such a case, the ultimate distance can be defined as L = L1 + · · · + LS . Accordingly, the gradients can be merged as ∂L = ∂L1 +···+ ∂LS , as illustrated in Figure 2.
model learn the spectral details of natural waveforms in differ- ent spatial and temporal resolutions. We used three distances in this study, which are explained in Section IV-B. -->
上述说明表明，无论如何配置窗长、帧移和 DFT bins，都可以计算 $\mathcal{L}\in\mathbb{R}$ 和 $\frac{\partial\mathcal{L}}{\partial\widehat{\boldsymbol{o}}_{1:T}}\in\mathbb{R}^T$，因此可以直接合并多个不同窗长和帧移的距离。在这种情况下，最终距离可以定义为 $\mathcal{L}=\mathcal{L}_1+\cdots+\mathcal{L}_S$。因此，梯度可以合并为（如上图所示） $\frac{\partial\mathcal{L}}{\partial\widehat{\boldsymbol{o}}_{1:T}}=\frac{\partial\mathcal{L}_1}{\partial\widehat{\boldsymbol{o}}_{1:T}}+\cdots+\frac{\partial\mathcal{L}_S}{\partial\widehat{\boldsymbol{o}}_{1:T}}$。
采用多频谱距离可以让模型学习不同空间和时间分辨率下自然波形的频谱细节。
<!-- The short- time spectral amplitude distances may be more appropriate than the waveform MSE in Equation (1) for the NSF models. For a single speech frame, it is assumed with an NSF model using a spectral amplitude distance that the spectral amplitude vector z = y ⊙ y∗ ∈ RK follows a multivariate log- normal distribution with a diagonal covariance matrix, where y = DFT(x) ∈ CK is the K-point DFT of a waveform frame x and ⊙ denotes element-wise multiplication. Although we cannot derive an analytical form of p(x) from p(z), we can at least infer that the distribution of x or the original waveform o1:T assumed with the model is at least not an isotropic Gaussian distribution. An NSF model with a spec- tral amplitude distance can potentially model the waveform temporal correlation within an analysis window.
 -->
短时频谱幅度距离可能比波形 MSE 更适合 NSF 模型。对于单个语音帧，假设使用频谱幅度距离的 NSF 模型，频谱幅度向量 $\boldsymbol{z}=\boldsymbol{y}\odot\boldsymbol{y}^*\in\mathbb{R}^K$ 符合对角协方差矩阵的多元对数正态分布，其中 $\boldsymbol{y}=\operatorname{DFT}(\boldsymbol{x})\in\mathbb{C}^K$ 是波形帧 $\boldsymbol{x}$ 的 $K$ 点 DFT，$\odot$ 表示逐元素相乘。尽管我们无法从 $p(\boldsymbol{z})$ 推导出 $p(\boldsymbol{x})$ 的解析形式，但至少可以推断出模型假设的 $\boldsymbol{x}$ 或原始波形 $\boldsymbol{o}_{1:T}$ 的分布至少不是各向同性的高斯分布。使用频谱幅度距离的 NSF 模型可以潜在地模拟窗口内的波形时间相关性。
<!-- Using the spectral amplitude distance is reasonable also because the perception of speech sounds are affected by the spectral acoustic cues such as formants and their transition [35], [36], [37]. Although the spectral amplitude distance ignores other acoustic cues, such as phase [38] and timing [39], we only considered the spectral amplitude distance in this study because we have not found a phase or timing distance that is differentiable and effective. -->
使用频谱幅度距离也是合理的，因为语音声音的感知受到频谱声学线索的影响，如共振峰及其过渡。

### Baseline NSF 模型

如图：
![](image/Pasted%20image%2020240325105640.png)
<!-- We now give the details on the b-NSF model. As Figure 3 illustrates, the b-NSF model uses three modules to convert an input acoustic feature sequence c1:B of length B into a speech waveformob1:T oflengthT:asourcemodulethatgeneratesan excitationsignale1:T,afiltermodulethattransformse1:T into an output waveform, and a condition module that processes c1:B for the source and filter modules. The model is trained using the spectral distance explained in the previous section. -->
b-NSF 模型使用三个模块将长度为 $B$ 的输入声学特征序列 $\boldsymbol{c}_{1:B}$ 转换为长度为 $T$ 的语音波形 $\widehat{\boldsymbol{o}}_{1:T}$。

<!-- Condition module: The input of the condition module is c = {c ,··· ,c }, where c = [f ̃,s⊤]⊤ includes the F0 value f ̃ and the spectral features s for the b-th frame. The F0 sub-sequence {f ̃ , · · · , f ̃ } is upsampled to  f by replicating each f ̃ for ⌈T/B⌉ times, after which 1:T b
f1:T is fed to the source module. The spectral features are processed by a bi-directional recurrent layer with long-short- term memory (LSTM) units [40] and a 1-dimensional CONV layer with a window size of 3. The processed spectral features are then concatenated with the F0 and upsampled as c ̃1:T . The layer size of the Bi-LSTM and CONV layers is 64 and 63, respectively. The dimension of c ̃t is 64. Note that there is no golden network structure for the condition module. We used the structure in Figure 3 because it has been used in our WaveNet-vocoder [41] 4. -->
condition module：输入为 $\boldsymbol{c}_{1:B}=\{\boldsymbol{c}_1,\cdots,\boldsymbol{c}_B\}$，其中 $\boldsymbol{c}_b=[\tilde{f}_b,\boldsymbol{s}_b^\top]^\top$ 包含第 $b$ 帧的 F0 值 $\tilde{f}_b$ 和频谱特征 $\boldsymbol{s}_b$。F0 子序列 $\{\tilde{f}_1,\cdots,\tilde{f}_B\}$ 通过复制每个 $\tilde{f}_b$ $\lceil\frac{T}{B}\rceil$ 次进行上采样，然后输入到 source module。频谱特征通过具有 LSTM 单元和窗口大小为 3 的 1 维 CONV 层的双向循环层处理，处理后的频谱特征与 F0 拼接并上采样为 $\tilde{\boldsymbol{c}}_{1:T}$。Bi-LSTM 和 CONV 层的层大小分别为 64 和 63，$\tilde{\boldsymbol{c}}_{t}$ 的维度为 64。

<!-- Source module: Given the F0, the source module con- structs an excitation signal on the basis of sine waveforms and random noise. In voiced segments, the excitation signal is a mixture of sine waveforms whose frequency values are determined by F0 and its harmonics. In unvoiced regions, the excitation signal is a sequence of Gaussian noise. -->
source module：给定 F0，source module 基于正弦波形和随机噪声构建激励信号。在有声段，激励信号是一系列正弦波形的混合，其频率值由 F0 及其谐波决定。在无声区域，激励信号是一系列高斯噪声。
<!-- The input F0 sequence is f1:T , where ft ∈ R≥0 is the F0
value of the t-th time step, and ft > 0 and ft = 0 denote being
voiced and unvoiced, respectively. A sine waveform e<0> with 1:T
the fundamental frequency can be generated as -->
输入 F0 序列为 $\boldsymbol{f}_{1:T}$，其中 $f_t\in\mathbb{R}_{\geq0}$ 是第 $t$ 个时间步的 F0 值，$f_t>0$ 和 $f_t=0$ 分别表示有声和无声。具有基频的正弦波形 $\boldsymbol{e}^{<0>}_{1:T}$ 可以生成为：
$$e_t^{<0>}=\begin{cases}\alpha\sin(\sum_{k=1}^t2\pi\frac{f_k}N_s+\phi)+n_t,&\mathrm{if~}f_t>0\\\frac\alpha{3\sigma}n_t,&\mathrm{if~}f_t=0&\end{cases},$$
<!-- where nt ∼ N (0, σ2) is Gaussian noise, φ ∈ [−π, π] is a ran-
dom initial phase, and Ns is the waveform sampling rate. The
hyper-parameter α adjusts the amplitude of source waveforms,
while σ is the standard deviation of the Gaussian noise5. We
set σ = 0.003 and α = 0.1 in this study. Equation (13) treats
ft as an instantaneous frequency [42]. Thus, the phase of the
e<0> becomes continuous even if ft changes. Figure 5 plots 1:T
an example e<0> and the corresponding f1:T . 1:T -->
其中 $n_t\sim\mathcal{N}(0,\sigma^2)$ 是高斯噪声，$\phi\in[-\pi,\pi]$ 是随机初始相位，$N_s$ 是波形采样率。超参数 $\alpha$ 调整源波形的幅度，$\sigma$ 是高斯噪声的标准差。在本研究中，我们设置 $\sigma=0.003$ 和 $\alpha=0.1$。上式将 $f_t$ 视为瞬时频率，因此即使 $f_t$ 变化，$e^{<0>}_{1:T}$ 的相位也是连续的。下图绘制了一个示例 $e^{<0>}_{1:T}$ 和相应的 $f_{1:T}$：
![](image/Pasted%20image%2020240325110852.png)

<!-- The source module also generates harmonic overtones. For
the h-th harmonic overtone, which corresponds to the (h+1)-
th harmonic frequency, an excitation signal e<h> is generated 1:T
from Equation (13) with (h + 1)ft. The source module then
uses a trainable feedforward (FF) layer with a tanh activation
function to combine e<0> and e<h> into the final excitation 1:T 1:T
signal e1:T = {e1,··· ,eT}, where et ∈ R,∀t ∈ {1,··· ,T}. This combination can be written as -->
source module 还生成谐波。对于第 $h$ 个谐波，对应于 $(h+1)$-th 谐波频率，从而可以使用 $(h+1)f_t$ 生成激励信号 $e^{<h>}_{1:T}$。然后 source module 使用具有 tanh 激活函数的可训练前馈（FF）层将 $e^{<0>}_{1:T}$ 和 $e^{<h>}_{1:T}$ 组合成最终激励信号 $e_{1:T}=\{e_1,\cdots,e_T\}$，其中 $e_t\in\mathbb{R},\forall t\in\{1,\cdots,T\}$。这种组合可以写成：
$$\boldsymbol{e}_{1:T}=\tanh(\sum_{h=0}^Hw_h\boldsymbol{e}_{1:T}^{<h>}+w_b),$$
<!-- where {w0, · · · wH , wb} are the FF layer’s weights, and H is the total number of overtones. -->
其中 $\{w_0,\cdots,w_H,w_b\}$ 是 FF 层的权重，$H$ 是谐波的总数。
<!-- The value of H is not critical to the model’s performance because the model can re-create higher harmonic tones, as the experiments discussed in Section IV-D demonstrated. We set H = 7 based on a tentative rule (H +1)∗fmax < Ns/4, where Ns = 16 kHz is the sampling rate, and fmax ≈ 500 Hz is the largest F0 value observed in our data corpus. We used Ns/4 as the upper-bound so that there is at least four sampling points in each period of the sine waveform.
 -->
$H$ 的值对模型的性能并不重要，因为模型可以重新创建更高的谐波音调。这里基于一个临时规则设置 $H=7$，即 $(H+1)\times f_{\max}<N_s/4$，其中 $N_s=16\text{ kHz}$ 是采样率，$f_{\max}\approx500\text{ Hz}$ 是数据语料库中观察到的最大 F0 值。使用 $N_s/4$ 作为上界，以便每个正弦波形周期中至少有四个采样点。

<!-- Neural filter module: The filter module of the b-NSF model transforms the excitation e1:T into an output waveform ob1:T by using five baseline dilated-CONV filter blocks. Th -->
neural filter module：b-NSF 模型的 filter module 使用五个 baseline dilated-CONV filter block 将激励信号 $e_{1:T}$ 转换为输出波形 $\widehat{\boldsymbol{o}}_{1:T}$。filter block 模块如图：
![](image/Pasted%20image%2020240325112043.png)

<!-- Suppose the input to the block is vin , where vin ∈ R, 1:T t
∀t ∈ {1, · · · , T }. This input vin is expanded in dimension 1:T
through an FF layer as tanh(wvin+b) ∈ R64,∀t ∈ {1,··· ,T}, t
where w ∈ R64×1 is the transformation matrix and b ∈ R64 is the bias. The expanded signal is then processed by a dilated-CONV layer, summed with the condition feature c ̃1:T ,
processed by the gated activation unit based on tanh and
sigmoid [12], and transformed by two additional FF layers.
This procedure is repeated ten times within this filter block,
and the dilation size of the dilated convolution layer in the
k-th stage is set to 2k−1. The outputs from the ten stages
are summed and transformed into two signals a1:T and b1:T .
After that, vin is converted into an output signal vout by 1:T 1:T vout
1:T
= vin
1:T
⊙ b1:T + a1:T , where ⊙ denotes element-wise eul 200 av0F
0
multiplication. The output vout is further processed in the 1:T
following filter block, and the output of the last filter block is
the generated waveform ob1:T . -->
假设 block 的输入为 $\boldsymbol{v}^{\text{in}}_{1:T}$，其中 $v^{\text{in}}_t \in \mathbb{R}, \forall t\in\{1,\cdots,T\}$。通过 FF 层将输入 $\boldsymbol{v}^{\text{in}}_{1:T}$ 扩展到 $\tanh(\boldsymbol{w}\boldsymbol{v}^{\text{in}}_{t}+\boldsymbol{b})\in\mathbb{R}^{64},\forall t\in\{1,\cdots,T\}$，其中 $\boldsymbol{w}\in\mathbb{R}^{64\times1}$ 是变换矩阵，$\boldsymbol{b}\in\mathbb{R}^{64}$ 是偏置。扩展后的信号然后通过 dilated-CONV 层，与条件特征 $\tilde{\boldsymbol{c}}_{1:T}$ 相加，通过基于 tanh 和 sigmoid 的门控激活单元处理，然后通过两个额外的 FF 层变换。这个过程在 filter block 中重复十次，dilated convolution 层的 dilation 大小在第 $k$ 阶段设置为 $2^{k-1}$。十个阶段的输出相加并转换为两个信号 $\boldsymbol{a}_{1:T}$ 和 $\boldsymbol{b}_{1:T}$。然后，$\boldsymbol{v}^{\text{in}}_{1:T}$ 通过 $\boldsymbol{v}^{\text{out}}_{1:T}=\boldsymbol{v}^{\text{in}}_{1:T}\odot\boldsymbol{b}_{1:T}+\boldsymbol{a}_{1:T}$ 转换为输出信号 $\boldsymbol{v}^{\text{out}}_{1:T}$。输出 $\boldsymbol{v}^{\text{out}}_{1:T}$ 进一步在下一个 filter block 中处理，最后一个 filter block 的输出是生成的波形 $\widehat{\boldsymbol{o}}_{1:T}$。
<!-- Our implementation used a kernel size of 3 for the dilated- CONV layers, which is supposed to be necessary for non- AR waveform models [18]. Both the input and output feature
vectors of the dilated-CONV layer have 64 dimensions. Ac- cordingly, the residual connection that connects two adjacent dilated-CONV layers also has 64 dimensions. The feature vectors to the FF layer that produces a1:T and b1:T have 128 dimensions, i.e., skip-connection of 128 dimensions. The b1:T is parameterized as b1:T = exp(b ̃1:T ) to be positive [19]. -->
具体实现中，dilated-CONV 层的 kernel size 为 3，这对于非 AR 模型是必要的。dilated-CONV 层的输入和输出特征向量都有 64 维。因此，连接两个相邻 dilated-CONV 层的残差连接也有 64 维。产生 $\boldsymbol{a}_{1:T}$ 和 $\boldsymbol{b}_{1:T}$ 的 FF 层的特征向量有 128 维，即 128 维的 skip-connection。$\boldsymbol{b}_{1:T}$ 被参数化为 $\boldsymbol{b}_{1:T}=\exp(\tilde{\boldsymbol{b}}_{1:T})$ 以保证为正。
<!-- The baseline dilated-CONV filter block is similar to the student models in ClariNet and Parallel WaveNet because all use the stack of so-called “dilated residual blocks” in AR WaveNet [12]. However, because the b-NSF model does not use knowledge distilling, it is unnecessary to compute the distribution of the signal during forward propagation as ClariNet and Parallel WaveNet do. Neither is it necessary to make the filter blocks invertible as IAF does. Accordingly, the dilated convolution layers can be non-causal, even though we used causal ones to keep configurations of our NSF models consistent with our WaveNet-vocoder in the experiments.
C. Simplified NSF model -->
baseline dilated-CONV filter block 类似于 ClariNet 和 Parallel WaveNet 中的 student models，因为所有这些模型都使用 AR WaveNet 中的“dilated residual blocks”。然而，由于 b-NSF 模型不使用 knowledge distilling，因此前向传播时不需要像 ClariNet 和 Parallel WaveNet 那样计算信号的分布。也不需要像 IAF 那样使 filter block 可逆。因此，dilated convolution 层可以是非因果的。

### Simplified NSF 模型（略）

### Harmonic-plus-noise NSF 模型（略）

## 实验（略）
