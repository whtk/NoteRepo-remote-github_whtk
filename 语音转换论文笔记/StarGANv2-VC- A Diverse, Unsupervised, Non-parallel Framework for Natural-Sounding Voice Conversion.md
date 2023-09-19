> interspeech 2021，Columbia University

1. 组合使用对抗源分类损失和感知损失（adversarial source classifier loss and perceptual loss）
2. 模型训练的时候只选了20个说话人，但是可以泛化到多种任务：any-to-many, cross-lingual等
3. 效果接近于SOTA的TTS，但是不需要文本标签
4. 全卷积结构，在Parallel WaveGAN 加持下可以实现实时的VC。

### Introduction

1. 最近的VC可以分为三个方向：基于AE、基于TTS、基于GAN
2. 模型的效果优于当时的SOTA，AUTO-VC，接近TTS的方法，VTN
3. 本文贡献：
   1. 把 StarGANv2引入VC
   2. 提出了一个新型的 adversarial source classifier loss，能够极大的提高转换语音和目标语音的相似性
   3. 使用 ASR 网络和 F0 提取网络的感知损失perceptual loss

## 方法

### StarGANv2-VC

1. 采取和 StarGANv2 一样的结构，一个生成器和一个判别器，把每个说话人看成一个域domain
2. 添加了一个预训练的JDC（joint detection and classification ）-F0提取网络实现转换过程中的 F0 一致性
   模型结构如图：
   ![](image/Pasted%20image%2020230916222459.png)

+ 生成器：把输入 Mel 谱 $\mathbf{X}_{src}$ 转换成 $G(\mathbf{X}_{src}, h_{sty},h_{f0})$，其中，$h_{sty}$ 为风格编码（来自于映射网络或者风格编码网络），$h_{f0}$ 来自于 F0 extraction network $F$ 的卷积层。
+ F0 网络：F0 提取网络 $F$ 是一个预训练的 JDC 网络，能够从Mel谱中获取F0信息。
+ 映射网络：映射网络 $M$ 从随机隐变量 $\mathbf{z} \in \mathcal{Z}$ 和域 $y \in \mathcal{Y}$ 中提取风格编码向量 $h_{M}$：$h_{M} = M(\mathbf{z}, y)$。其中，$\mathbf{z}$ 从高斯分布中采样。
+ 风格编码器：给定参考Mel谱 $\mathbf{X}_{ref}$（其实就是target语音的Mel谱），风格编码器 $S$ 提取在域 $y$ 中的风格编码向量 $h_{sty}$：$h_{sty} = S(\mathbf{x_{ref}},y)$。
+ 判别器：$D$ 是一个多任务的判别器，有很多个输出分支。每个分支 $D_y$ 进行二值分类，确定Mel谱 $\mathbf{x}$ 是其域 $y$ 的 real 谱还是由 $G$ 生成的 fake 谱 $G(\mathbf{x},s)$；同时为了捕捉一些域特定的特征，还包含了一个额外的分类器 $C$（和 $D$ 结构一样），用于学习生成谱的原始域（简单来说，就是判断输出的Mel谱来自哪个说话人domain）。

#### 目标函数

StarGANv2-VC 的目标是学习从 $\mathcal{X}_{y_{src}}$ 到 $\mathcal{X}_{y_{trg}}$ 的映射，把来自于源说话人域 $y_{trg} \in \mathcal{Y}$ 的样本 $\mathbf{X} \in \mathcal{X}_{y_{src}}$ 转换到目标说话人域 $y_{trg} \in \mathcal{Y}$ 的样本 $\hat{\mathbf{X}} \in \mathcal{X}_{y_{trg}}$。

训练的时候，随机获取风格编码向量 $s$（来自 $F$ 和 $S$），基于以下损失函数进行模型的训练：
**对抗损失**，其中，$D(\cdot,y)$ 表示域 $y$ 下的真伪判别分类器：

$$
\begin{aligned}
\mathcal{L}_{a d v}=& \mathbb{E}_{\boldsymbol{X}, y_{s r c}}\left[\log D\left(\boldsymbol{X}, y_{s r c}\right)\right]+\\
& \mathbb{E}_{\boldsymbol{X}, y_{t r g}, s}\left[\log \left(1-D\left(G(\boldsymbol{X}, s), y_{t r g}\right)\right)\right]
\end{aligned}
$$

**对抗源分类损失**，$C(\cdot)$ 代表额外的分类器，$CE(\cdot)$ 代表交叉熵。

$$
\mathcal{L}_{a d v c l s}=\mathbb{E}_{\boldsymbol{X}, y_{t r g}, s}\left[\mathrm{CE}\left(C(G(\boldsymbol{X}, s)), y_{t r g}\right)\right]
$$

**风格重构损失**，用于确保前后两次的风格尽可能相同。

$$
\mathcal{L}_{s t y}=\mathbb{E}_{\boldsymbol{X}, y_{t r g}, s}\left[\left\|s-S\left(G(\boldsymbol{X}, s), y_{t r g}\right)\right\|_{1}\right]
$$

**风格多样性损失**，这个损失越大越好，表明在同一个域下，不同的风格编码向量 $s$ 可以产生多样性更强的谱。

$$
\begin{aligned}
&\left.\mathcal{L}_{d s}=\mathbb{E}_{\boldsymbol{X}, s_{1}, s_{2}, y_{t r g}}\left[\| G\left(\boldsymbol{X}, s_{1}\right)-G\left(\boldsymbol{X}, s_{2}\right)\right) \|_{1}\right]+ \\
&\left.\mathbb{E}_{\boldsymbol{X}, s_{1}, s_{2}, y_{t r g}}\left[\| F_{\text {conv }}\left(G\left(\boldsymbol{X}, s_{1}\right)\right)-F_{\text {conv }}\left(G\left(\boldsymbol{X}, s_{2}\right)\right)\right) \|_{1}\right]
\end{aligned}
$$

**F0一致性损失**，设 $F(\boldsymbol{X})$ 可以基于 F0 网络求出Mel谱的F0值，同时定义归一化F0值为，$\hat{F}(\boldsymbol{X})=\frac{F(\boldsymbol{X})}{\|F(\boldsymbol{X})\|_{1}}$，则 F0 一致性损失为：

$$
\mathcal{L}_{f0}=\mathbb{E}_{\boldsymbol{X}, s}\left[\|\hat{F}(\boldsymbol{X})-\hat{F}(G(\boldsymbol{X},s))\|_{1}\right]
$$

**语音一致性损失**，为了确保转换的语音具有相同的语音内容，从预训练的联合 CTC-attention VGG-BLSTM 网络计算其内容编码 $h_{asr}(\cdot)$，则语音一致性损失为：

$$
\mathcal{L}_{asr}=\mathbb{E}_{\boldsymbol{X}, s}\left[\|h_{asr}(\boldsymbol{X})-h_{asr}(G(\boldsymbol{X},s))\|_{1}\right]
$$

**Norm consistency loss**：

$$
\mathcal{L}_{n o r m}=\mathbb{E}_{\boldsymbol{X}, s}\left[\frac{1}{T} \sum_{t=1}^{T}|\|\boldsymbol{X} \cdot, t\|-\| G(\boldsymbol{X}, s) \cdot, t\||\right]
$$

**循环一致性损失（循环重构损失）**，确保Mel谱在两次转换后完全一致：

$$
\left.\mathcal{L}_{c y c}=\mathbb{E}_{\boldsymbol{X}, y_{s r c}, y_{t r g}, s}[\| \boldsymbol{X}-G(G(\boldsymbol{X}, s), \tilde{s})) \|_{1}\right]
$$

总生成器损失为：

$$
\begin{aligned}
\min _{G, S, M} & \mathcal{L}_{a d v}+\lambda_{a d v c l s} \mathcal{L}_{a d v c l s}+\lambda_{s t y} \mathcal{L}_{s t y} \\
&-\lambda_{d s} \mathcal{L}_{d s}+\lambda_{f 0} \mathcal{L}_{f 0}+\lambda_{a s r} \mathcal{L}_{a s r} \\
&+\lambda_{n o r m} \mathcal{L}_{n o r m}+\lambda_{c y c} \mathcal{L}_{c y c}
\end{aligned}
$$

总判别器损失为：

$$
\min _{C, D}-\mathcal{L}_{a d v}+\lambda_{c l s} \mathcal{L}_{c l s}
$$

其中，$\mathcal{L}_{c l s}=\mathbb{E}_{\boldsymbol{X}, y_{s r c}, s}\left[\mathbf{C E}\left(C(G(\boldsymbol{X}, s)), y_{s r c}\right)\right]$ 表示将

### 其他

1. 在训练判别器时，$G$ 的权重固定，分类器 $C$ 训练用来判断转换后的样本的原始域，而不管其目标域是什么。
2. 在训练生成器时，分类器 $C$ 的权值固定，通过调整 $G$ 参数使得其转换后的样本（生成样本）尽可能的被判为目标域，而不管输入样本的原始域是什么。
3. 由上分析可知，这里的分类器和传统的分类器并不一样，其作用更像是一个对抗性的判别器；因为生成器的目的是尽可能的骗过分类器（即训练生成器，使得其输出任意图片，分类器都能判成目标域），而分类器的目的是尽可能不被生成器欺骗（即训练分类器，使得不管生成器给出任何图片，分类器需要判其原始域——尽管生成器输出的图片是目标域的，而传统的分类器在这个时候会被要求判成目标域）。
4. 论文中的 ASR模型、F0 LDC模型、声码器（Parallel WaveGAN）都是使用已有的固定模型，且不在训练过程中更新参数。
