> IJCNN 2022，厦门大学

1. 提出 FCH-TTS，快速、可控、 通用的 TTS，可以生成高质量的 spectrogram
2. 基本结构和 FastSpeech 很相似，但是采用了更简单且更有效的 attention-based soft alignment 机制来替换 FastSpeech 中的 teacher model
3. 为了控制速度和韵律，采用 fusion model 来建模说话人特征从而更好地获得 timbre
4. 还用了一些特别的损失函数来确保mel 谱的质量
5. 相比于所有的 baseline 都可以实现最快的推理速度和最好的合成质量

## Introduction

1. FastSpeech 系列还是需要 alignment 工具，对于低资源的 TTS 还是不太友好
2. 本文 follow FastSpeech 来实现 expressive TTS，也采用了 length regulator
3. 提出 FCH-TTS 可以快速合成可控、高质量的 mel 谱，设计 attention-based soft alignment，可以在低资源的语音数据中获取对齐；采用 multi-scale residual fusion 网络融合多尺度说话人特征

## 相关工作（略）

## 方法

### 模型架构

用 [FastSpeech- Fast, Robust and Controllable Text to Speech 笔记](../FastSpeech-%20Fast,%20Robust%20and%20Controllable%20Text%20to%20Speech%20笔记.md) 作为 backbone。

如图：
![](image/Pasted%20image%2020240118212349.png)

包含四个组成部分：
+ encoder，包含 text、spectrogram 和 speaker encoder
+ aligner，包含 attention, soft-alignment, duration predictor and length regulator
+ FusionNet
+ decoder

encoder 的输入分别是 文本 $T{=}[t_1,t_2,...,t_M]$、spectrogram 和 spectrogram $\boldsymbol{S=}[s_1,s_2,...,s_N]$，计算表征：
$$\begin{gathered}
\begin{aligned}(K,H)=TextEncoder(T)\end{aligned} \\
\begin{aligned}Q=SpecEncoder(S),P=SpeakerEncoder(S)\end{aligned} 
\end{gathered}$$
其中 $P=[p_1,p_2,...,p_N]$ 为 speaker embedding，$H\:=\:[h_1,h_2,...,h_M]$ 包含丰富的上下文信息，可以用于预测 duration $D$：
$$D=DurationPredictor(H)$$
然后 $K=[k_1,k_2,...,k_M],\:Q=[q_1,q_2,...,q_N],V=K$ 用于计算文本和 spectrogram 之间的 attention，然后做 soft alignment 得到 $A$，做 length regulation 得到 拓展后的表征 $E$，length regulator 用的就是 FastSpeech 中的。上述过程公式表示如下：
$$\begin{gathered}
A=Attention(K,Q,V) \\
E=LengthRegulator(H,SoftAlignment(A),\alpha) 
\end{gathered}$$
然后 speaker embedding 通过 FusionNet 进行融合：
$$F=FusionNet(H,P)$$
decoder 输入 $E,F$ 生成低精度的 spectrogram，然后通过 super-resolution
reconstruction module 恢复到 全精度：
$$Y=Super\text{-}resolution(Decoder(E,F))$$
最终通过 MelGAN 生成波形。

### soft alignment

给定文本序列 $T$ 和 spectrogram $S$，可以得到两个对应的 hidden representation，然后通过点积计算 attention 矩阵 $A\in R^{M\times N}$，然后就可以通过 $A$ 知道每个 character 最优对应哪些帧（也就是 duration），且此 duration 满足以下条件：
+ 边界性：$s_1\leftrightarrow t_1,s_N\leftrightarrow t_M$
+ 单调性和局部最大性：如果 $s_j(1\leq j\leq N-1)\to t_i(1\leq i\leq M-1)$，则 $s_{j+1}\leftrightarrow\max(t_i,\:t_{i+1})$
+ smoothing complementarity

但由于目标只是获得 duration 而非具体的对齐关系，采用 smoothing 来将那些没有对应 frame 的文本强制对齐到其相邻文本的 frame 中。

例如，对于下图的矩阵：
![](image/Pasted%20image%2020240119103101.png)
其得到的 duration 为 $D_1=[1,1,2,2,1]$。

### FusionNet

在自回归模型中，直接把 speaker features 进行拼接可以得到较好的效果，这种融合方式在非自回归模型中效果较差，提出 multi-scale residual fusion network (FusionNet)  用于说话人特征的融合。

网络结构如下：
![](image/Pasted%20image%2020240119103342.png)
网络类似于 U-Net，包含上采样和下采样两部分：
+ 先通过 $k$ 个下采用得到不同的 scale 的内部特征
+ 基于 $k$ 个内部特征，采用上采样逐步恢复丢失的数据
+ 特征融合模块不仅融合下采样的信息，也会把 speaker  encoder 得到的特征进行融合，包含两个部分：
	+ multi-end fusion：融合不同 scale 的特征
	+ residual fusion：进行残差连接

### 损失函数

需要考虑以下三个损失：
+ duration prediction 损失
+ guided attention 损失
+ mel 谱 损失

mel 谱损失用的是图像领域的 SSIM loss，比较两张图片在 luminance, contrast and structure 三个维度的差异：
$$SSIM(x,y)=\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}$$
然后还有一个经典的 MAE loss 和一个 perceptual loss。perceptual loss通过从预训练的网络提取 high-level hidden feature 然后比较两者的不同，这里用的是预训练的 VGG-16，计算如下：
$$L_{perc}=\frac{1}{N}\sum|f_{vgg}(Y)-f_{vgg}(Y^{'})|$$
总的 mel 谱 loss 为：
$$L_{Spec}=L_{SSIM}+L_{MAE}+L_{perc}$$

对于 guided attention loss，用的是 [DCTTS- Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention 笔记](DCTTS-%20Efficiently%20Trainable%20Text-to-Speech%20System%20Based%20on%20Deep%20Convolutional%20Networks%20with%20Guided%20Attention%20笔记.md) 中的，记为 $L_{Attn}$。

duration loss 用的是 smooth mean absolute error：
$$L_{Dura}=SMAE(D,D^{'})$$

总的 loss 为：
$$L=L_{Spec}+\lambda_1L_{Attn}+\lambda_2L_{Dura}$$
实验时，$L_1=L_2=0.1$。

## 实验（略）