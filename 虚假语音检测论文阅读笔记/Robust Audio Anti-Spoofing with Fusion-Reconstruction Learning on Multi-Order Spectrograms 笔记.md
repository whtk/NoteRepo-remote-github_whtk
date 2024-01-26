> InterSpeech 2023，The University of Sydney，City University of Macau

1. spectrograms 在 anti-spoofing 上效果很好，但是以 multi-order spectral patterns 出现的一些信息没人用
2. 提出 S2pecNet，使用 multi-order spectral pattern 实现鲁棒的 anti-spoofing
3. 具体来说，采用二阶的 spectral pattern 以 coarse-to-fine 的方式进行融合，然后用两个 branch，分别从 spectral 和 temporal context 进行 fine-level fusion，最后再从融合的 representation 进行 reconstruction 来进一步减少融合信息的损失
4. 在 ASVspoof2019 LA Challenge 实现了 SOTA，EER 为 0.77%

## Introduction

1. 之前的方法都是基于特定的 audio features，但是不同的 features 对不同的 attack 有不同的效果，如图
![](image/Pasted%20image%2020240126113742.png)
2. 于是提出使用不同 order 的 spectral pattern 以互补，比如 2nd-order spectrograms 对于 real-world speech 中的 noise patterns 更敏感
3. 提出 S2pecNet，使用 multi-order spectrogram pattern，包括 raw 和 power spectrograms，两者以 coarse-to-fine 的方式进行融合，然后用两个 branch，分别从 spectral 和 temporal context 进行 fine-level fusion，最后再从融合的 representation 进行 reconstruction 来进一步减少融合信息的损失
4. 贡献如下：
   1. 提出一个新的 deep learning based fusion architecture，用于 audio anti-spoofing，使用 multi-order spectrograms
   2. 提出了 coarse-to-fine fusion 机制，使用两个 branch，分别从 spectral 和 temporal context 进行 fine-level fusion
   3. 提出了 reconstruction strategy，用于保留 fused speech representations 中的信息

## 方法

S2pecNet 架构如下：
![](image/Pasted%20image%2020240126114047.png)

1st-order raw spectrogram 和 2nd-order power spectrogram 首先分别输入到 encoders，得到的 encoded features 进行 concat，然后输入到 temporal-spectral fusion 模块，得到 spoofing-sensitive representation。然后用 reconstruction 机制，迫使两个 decoders 将 fused features 重构回原始的 raw 和 power spectrograms。

### Raw Spectrogram and Power Spectrogram Encoding

S2pecNet 输入 waveform $\mathbf{X}$，得到 first-order patterns 为 raw spectrogram $\mathbf{X}_{1st}$，然后输入到 CNN based encoder $E_{1st}$，得到 feature map $\mathbf{H}_{1st} \in \mathbb{R}^{C \times F \times T}$，其中 $C$、$F$ 和 $T$ 分别表示 channels 数、spectral bins 数和序列长度。对于 second-order patterns，得到的是 audio 的 power spectrogram $\mathbf{X}_{2nd}$，然后输入到另一个 CNN based encoder $E_{2nd}$，得到 feature map $\mathbf{H}_{2nd} \in \mathbb{R}^{C \times F \times T}$，两者的维度相同。

### Temporal-Spectral Fusion

两个 encoder 的输出 feature maps $\mathbf{H}_{1st}$ 和 $\mathbf{H}_{2nd}$ 的 spectral orders 不同，因此需要一个 temporal-spectral fusion (TSF) 模块来 refine 和 fuse 这两个 feature maps。TSF 以 coarse-to-fine 的方式得到两个 spectral 之间的依赖关系，探索互补的 spoofing-related patterns：
+ 先进行 coarse fusion，通过 channel-wise 的方式将 $\mathbf{H}_{1st}$ 和 $\mathbf{H}_{2nd}$ 进行拼接，然后通过一组 convolution filters，得到 coarse fused representation $\mathbf{H}_{fuse}$
+ 为了从 $\mathbf{H}_{fuse}$ 中得到 finer spoofing-sensitive features，通过 long-term temporal dependencies 和 spectral patterns 来得到 attention map $\mathbf{A}$，来突出 $\mathbf{H}_{fuse}$ 中更容易受到 spoofing 影响的 patterns

为了得到 $\mathbf{A}$，从两个不同的 context 中得到两个 sub-attention maps $\mathbf{A}^{spectral}$ 和 $\mathbf{A}^{temporal}$，其中一个关注 temporal context，另一个关注 spectral context。具体来说，对 $\mathbf{H}_{fuse}$ 沿着 temporal 和 spectral 维度进行 pooling，得到：
$$\mathbf{H}_{\mathrm{fuse}}^{\mathrm{spectral}}=\max_t(|\mathbf{H}_{\mathrm{fused}}|),\mathbf{H}_{\mathrm{fuse}}^{\mathrm{temporal}}=\max_s(|\mathbf{H}_{\mathrm{fused}}|),$$
其中 $\mathbf{H}_{fuse}^{spectral} \in \mathbb{R}^{C \times F \times 1}$，$\mathbf{H}_{fuse}^{temporal} \in \mathbb{R}^{C \times 1 \times T}$，$| \cdot |$ 表示 element-wise absolute operator，$max_s$ 表示 global spectral pooling operator，$max_t$ 表示 global temporal pooling operator。这样，$\mathbf{H}_{fuse}^{spectral}$ 包含了 global temporal information，$\mathbf{H}_{fuse}^{temporal}$ 包含了 global spectral information。然后就可以得到两个 attention maps $\mathbf{A}^{spectral} \in \mathbb{R}^{C \times F \times 1}$ 和 $\mathbf{A}^{temporal} \in \mathbb{R}^{C \times 1 \times T}$：
$$\mathbf{A}^{\mathrm{spectral}}=\text{Conv}_{\mathrm{s}}(\mathbf{H}_{\mathrm{fuse}}^{\mathrm{spectral}}),\mathbf{A}^{\mathrm{temporal}}=\text{Conv}_{\mathrm{t}}(\mathbf{H}_{\mathrm{fuse}}^{\mathrm{temporal}}),$$
其中 $\text{Conv}_t$ 和 $\text{Conv}_s$ 分别表示用于得到两个 attention maps 的 convolution layers。这样，最终的 attention map $\mathbf{A}$ 为：$\mathbf{A} = \mathbf{A}^{spectral} \times \mathbf{A}^{temporal}$，最终的 fused representation 为：
$$\mathbf{H_{attentive}}=\mathbf{A}\times\mathbf{H_{fused}}.$$

### Raw Spectrogram and Power Spectrogram Decoding

为了防止在 encoding 和 feature fusion 过程中的信息丢失，采用 raw spectrogram decoder $D_{1st}$ 和 power spectrogram decoder $D_{2nd}$，分别用于重构 raw spectrograms 和 power spectrograms。$D_{1st}$ 包含一系列 deconvolution layers，将 fused feature $\mathbf{H}_{attentive}$ 重构为 raw spectrogram $\hat{\mathbf{X}}_{1st}$。同理 $D_{2nd}$ 也包含一系列 deconvolution layers，将 $\mathbf{H}_{attentive}$ 重构为 power spectrogram $\hat{\mathbf{X}}_{2nd}$。raw spectrogram 重构损失 $L_{1st}$ 和 power spectrogram 重构损失 $L_{2nd}$ 分别为：
$$\mathcal{L}_{1^{st}}=\|\hat{\mathbf{X}}_{1^{st}}-\mathbf{X}_{1^{st}}\|,\mathcal{L}_{2^{nd}}=\|\hat{\mathbf{X}}_{2^{nd}}-\mathbf{X}_{2^{nd}}\|.$$
最小化这两个损失，目的是使用 fused representation 最好地重构原始的 raw 和 power spectrograms。

### 虚假检测

分类器进一步将 fused representation $\mathbf{H}_{attentive}$ 作为输入，得到二分类预测 $\hat{y}$。分类损失 $L_{cls}$ 为 weighted binary cross-entropy (WACE)，用于计算预测 $\hat{y}$ 和 GT $y$ 之间的差异，可以表示为：
$$\begin{aligned}\mathcal{L}_{\text{cls}}=-\frac{1}{N}\sum_{i=1}^{N}y\cdot\log(\hat{y})+(1-y)\cdot\log(1-\hat{y}).\end{aligned}$$

### 模型训练

S2pecNet 的总损失 $L$ 包含上面提到的三个损失项，包括 raw spectrogram 重构损失 $L_{1st}$，power spectrogram 重构损失 $L_{2nd}$ 和分类损失 $L_{cls}$。总损失由超参数 $\alpha$ 控制：
$$\mathcal{L}=\alpha(\mathcal{L}_{1^{st}}+\mathcal{L}_{2^{nd}})+\mathcal{L}_{\mathrm{cls}}.$$

## 实验和讨论（略）
