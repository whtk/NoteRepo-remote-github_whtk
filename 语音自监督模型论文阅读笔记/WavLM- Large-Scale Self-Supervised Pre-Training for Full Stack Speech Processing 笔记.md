> Microsoft，2022

1. SSL 在 ASR 中很大成功，但是在其他任务上却很少使用
2. 提出 WavLM，全栈式解决所有的下游语音任务
3. 通过在预训练时学习进行 掩蔽语音预测和去噪，前者任务可以保留语音内容的建模，而后者任务通过语音降噪任务可以提高在非 ASR 任务中的潜能

## Introduction

1. SSL 在 NLP 领域取得了巨大成功，对于语音，主要专注于 ASR
2. 对于非 ASR 任务，需要一个全栈的通用的预训练模型
3. 但是语音中不同的任务关注语音信号的不同方面，ASV 学习说话人特征而不关注内容，ASR 则只关注内容而不关注说话人特征，SD 和 SS 则涉及多个说话人，SUPERB 证明，使用来自不同层的 embedding 的加权和，预训练模型在全栈语音任务上有很大潜力（不同层有不同任务的信息）
4. 现有预训练模型存在一些缺点：
	1. 多说话人任务效果不行
	2. 训练数据和真实场景相差较大
5. 本文提出 WavLM，提出 masked speech denoising 和 prediction 框架，输入是被 mask 的模拟带噪或者重叠语音，目标是预测伪标签，有点类似于 HuBERT，由于同时预测 mask 语音和降噪，模型不仅可以学 ASR 的信息，也可以学非 ASR 的信息
6. 将 门控相对位置偏移（grep）添加到Transformer结构，提高了ASR的模型性能，与wav2vec 2.0和HuBERT中使用的卷积相对位置相比， gate 允许通过调节当前语音内容来自适应地调整相对位置偏差，训练采用 94k小时的公开音频
7. 在19个任务上评估模型，效果都很好，而且代码开源！！！

## 相关工作（略）

## 背景：HuBERT

HuBERT 的 backbone 是 $L$ 层的 Transformer ，训练时，输入是 masked 的声学特征 $\mathbf{u}$，输出是 hidden state $\mathbf{h}^{L}$，通过预测离散的目标 $\mathbf{z}$ 来优化模型，这里每个 $z_t\in[C]$ 都是 $C$ 类的 categorical 变量。

HuBERT 采用 masked speech prediction 任务，损失仅在 masked 的区域进行计算。

迭代进行 re-clustering 和 re-training，第一次迭代，目标是 MFCC 聚类，第二次迭代，通过对第一次迭代得到的模型所生成的表征来进行聚类。

## WavLM

提出 masked speech denoising 和 prediction 框架。

### 模型架构
![](image/Pasted%20image%2020230525223344.png)
采用 Transformer 作为 backbone，包含
+ 卷积特征编码，包含 七个  temporal convolution+layer norm+GELU，一帧大概 25 ms，输出为 $\mathbf{x}$
+ Transformer Encoder，采用 gated relative position bias，令 $\left\{\mathbf{h}_i\right\}_{i=1}^T$ 表示 attention 的输入，每个 $\mathbf{h}_i$ 都投影到 Q、K、V ：$$\mathbf{q}_i, \mathbf{k}_i, \mathbf{v}_i=\mathbf{h}_i \mathbf{W}^Q, \mathbf{h}_i \mathbf{W}^K, \mathbf{h}_i \mathbf{W}^V$$
然后 attention 计算为：$$\begin{aligned}
a_{i j} & \propto \exp \left\{\frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_k}}+r_{i-j}\right\} \\
\tilde{\mathbf{h}}_i & =\sum_{j=1}^T a_{i j} \mathbf{v}_j
\end{aligned}$$
这里的 $r_{i-j}$ 就是 gated relative position bias，其计算为：$$\begin{aligned}
& g_i^{\text {(update })}, g_i^{(\text {reset) }}=\sigma\left(\mathbf{q}_i \cdot \mathbf{u}\right), \sigma\left(\mathbf{q}_i \cdot \mathbf{w}\right) \\
& \tilde{r}_{i-j}=w g_i^{(\text {reset) }} d_{i-j} \\
& r_{i-j}=d_{i-j}+g_i^{\text {(update })} d_{i-j}+\left(1-g_i^{\text {(update })}\right) \tilde{r}_{i-j}
\end{aligned}$$其中 $d_{i-j}$ 为可学习的 scalar relative position bias，$\mathbf{u},\mathbf{w},w$ 都是可学习的参数

本文的 $d_{i-j}$ 是 bucket relative position embedding，embedding 的参数在所有层中共享，采用 $n=320$ 个embedding：$$d_{|i-j|}= \begin{cases}|i-j|, & |i-j|<\frac{n}{4} \\ \left\lfloor\frac{n}{4}\left(\frac{\log (|i-j|)-\log \left(\frac{n}{4}\right)}{\log (m)-\log \left(\frac{n}{4}\right)}+1\right)\right\rfloor, & \frac{n}{4} \leq|i-j|<m \\ \frac{n}{2}-1, & |i-j| \geq m\end{cases}$$
$m=800$ 表示 maximum offset。

与 wav2vec 2.0 和 HuBERT 中的 卷积相对位置嵌入 相比，gates考虑了内容，并通过调节当前语音内容自适应地调整 relative position bias。直观地说，如果一帧是无声的，而另一帧属于语音片段，那么两帧之间相同的距离偏移往往会起到不同的作用。

### Masked Speech Denoising 和 Prediction

手动生成具有噪声和重叠的语音作为输入，目标是预测**原始语音**（也就是不加噪或者不重叠前的）在masked的区域上的伪标签。

具体来说，从每个 batch 的语音中随机选几条，然后在随机区域将它和随机噪声或者另一条随机语音混合。
> 注：这里的噪声音频和混合音频是从同一个 batch 中的其他语音选的，然后随机裁剪，根据一个随机的能量比进行缩放

通过这一操作，模型开源从噪声或者重叠的语音中识别出主要的说话人，同时预测其说话的内容！

采用 mask prediction loss  优化网络，给定语音 $\mathbf{u}$ 和处理后的（加噪重叠）语音 $\mathbf{u}^{\prime}$，目标是生成伪标签 $\mathbf{z}$，和 HuBERT 一样，采用 k-mean 对 MFCC  或者 latent representation 特征进行聚类作为伪标签，最后的木目标函数为：$$\mathcal{L}=-\sum_{l \in K} \sum_{t \in M} \log p\left(z_t \mid \mathbf{h}_t^L\right)$$
### 预训练数据

先前的模型的数据集都是 audiobook 里的，但是和真实场景相差较大。于是采用两个数据集拓展数据：
+ 10k hours GigaSpeech ，来自 audiobooks 和 YouTube
+ VoxPopuli，多语言无标签数据集，4000K 小时，23 种语言，但是只用了英语的部分，有 24K 小时
总的数据集有 94K 小时，LibriLight, VoxPopuli 和 GigaSpeech.

### 训练稳定性

attention 训练过程种，使用 fp16 训练会出现 overflow，也就是计算 attention score 的时候 $\frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d}}$ 会大于 fp16 的上界。

采用了一个简单的方法来提高其上界，softmax 满足
$$\operatorname{softmax}(\mathbf{x}+\alpha)_k=\operatorname{softmax}(\mathbf{x})_k$$
其中，$\alpha$ 为常数，则计算 score 的时候变为：$$\begin{aligned}
\alpha_{i, j} & \propto \exp \left\{\frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d}}+r_{i-j}\right\} \\
& =\exp \left\{\left(\frac{\mathbf{q}_i}{c \sqrt{d}} \cdot \mathbf{k}_j-\max _{j^{\prime} \leq T}\left(\frac{\mathbf{q}_i}{c \sqrt{d}} \cdot \mathbf{k}_{j^{\prime}}\right)\right) \times c+r_{i-j}\right\} .
\end{aligned}$$
标量参数 $c=32$。

## 实验

### 预训练

WavLM Base 和 WavLM Base+ 有 12 层 Transformer encoder，hidden state 维度 768，attention head 8，94.70M 参数量。

WavLM Large 有 24 层 Transformer encoder，hidden state 维度 1024，attention head 12，316.62M 参数量。

不同层对不同任务的关注度：![](image/Pasted%20image%2020230526215831.png)
颜色越深，表示任务越看重这一层的特征。

分析：
+ bottom layers 对 说话人相关的任务 贡献更大；top layers 对 ASR 相关任务更重要

