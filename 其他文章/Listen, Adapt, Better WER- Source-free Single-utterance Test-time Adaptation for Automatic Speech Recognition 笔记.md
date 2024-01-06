> Interspeech 2022

1. Test-time Adaptation（也被称为 Source-free Domain Adaptation） 最开始用在 CV 中，通过调整在源域训练的模型来实现对测试域样本更好的预测（通常是 OOD）
2. 本文提出用于语音识别的 Single-Utterance Test-time Adaptation (SUTA) 框架，首次研究ASR中的 TTA 
3. 经验表明，SUTA 可以提高域外目标语料库和域内测试样本上 ASR 模型的性能

## Introduction

1. iid 条件下，基于深度学习 AS R 效果很好，但是当测试数据和训练数据的 分布不同时，性能会验证下降
2. UDA 是一种常用的方法，但是 UDA 也需要源域数据和目标域数据，而这通常会首先限制：
	1. 出于隐私原因，源数据有时候可能没有
	2. 需要收集和处理目标数据，耗时
3. TTA 可以在很少的目标域数据（一个batch或者就一个 instance）且不需要获得源域数据的情况下，有效地适应模型，在 CV 中用处很大
4. 大部分的 TTA 都限制在 batch level ，而 SITA 提出一种 single instance TTA 方法，不需要预先收集 batch 的 test samples，但是仍专注于 cv ，且依赖于数据增强和 BN 层
5. 本文提出了Single-Utterance Test-time Adaptation，SUTA 框架，可以用于任何基于 CTC 的 ASR 模型，只需要一条语音就可以以无监督的方式进行 TTA

## 方法

设 ASR 模型为 $g(y \mid x ; \theta)$，参数为 $\theta$，输入语音 $x$，输出没有归一化的词类别 $y$。其中，$\theta$ 可以分为两部分，$\theta_{\mathbf{f}}$ 在 adaptation 阶段参数被冻结，$\theta_{\mathbf{a}}$ 在 inference 的时候更新。测试数据集 $D_{test}$ 包含 $n$ 条语音 $\{x_1,x_2,\dots,x_n\}$，本文重点在于，使用一条语音 $x_i$ 无监督地适应 $\theta_{\mathbf{a}}$。

使用带有 CTC loss 的 Transformer Encoder 作为 ASR 模型，但其实可以用在各种模型上。

设 字符+ CTC blank token 的数量为 $C$，CTC 输出的时间帧为 $L$，则输出 $\mathbf{O}\in\mathbb{R}^{L\times C}$。

### SUTA

![](./image/Pasted%20image%2020230414211224.png)

#### Entropy minimization

由于测试的时候 label 未知，自适应时采用无监督的、基于熵的损失函数。Entropy Minimization 用于在大量目标数据的情况下进行域自适应。

本文使用 EM 目标函数，基于单条语音，对模型参数进行无监督的 TTA，损失计算为：$$\mathcal{L}_{e m}=\frac{1}{L} \sum_{i=1}^L \mathcal{H}_i=-\frac{1}{L} \sum_{i=1}^L \sum_{j=1}^C \mathbf{P}_{\mathrm{ij}} \log \mathbf{P}_{\mathbf{i j}}$$
> 计算的时候忽略了 blank token 来避免类不平衡问题。

#### Minimum class confusion

在 EM 目标函数中，Minimum Class Confusion（MCC）目标函数作为一种替代的模型参数调整方法，主要通过减少不同类别之间的相关性来实现。

MCC 损失计算为：$$\mathcal{L}_{m c c}=\sum_{j=1}^C \sum_{j^{\prime} \neq j}^C \mathbf{P}_{\cdot \mathbf{j}}^{\top} \mathbf{P}_{\cdot \mathbf{j}^{\prime}}$$
其中，$\mathbf{P}_{\cdot \mathbf{j}} \in \mathbb{R}^L$ 表示第 $j$ 类在长为 $L$ 帧下的概率向量。通过对混淆矩阵非对角元素添加惩罚来最小化不同类之间的相关性。
> 公式其实就是在求所有非对角元素的值的和。最小化这个值，可以减少类之间的相关性，因为如果类相关性比较大，那么输出的时候这两个类的概率值会比较接近，体现在 confusion 上就是值比较大，通过最小化 confusion 就可以减少类相关性。

#### Temperature smoothing

基于 EM 损失或者 MCC 损失的原始的 TTA 效果提升不大，原因可能是因为 entropy loss 对于一些预测信心比较大的帧，值很小，从而导致不能从这些帧中获得指导，甚至还可能存在梯度消失的问题。从而使得 $\mathcal{L}_{em}$ 主要受那些不确定的帧的影响，导致更新的方向不可靠。
> 也就是说，我们既要用上这些确定性较大的帧，又要让这些帧的影响比较大（因为对于确定性大的帧，entropy 小，导致其对loss的影响很小，我们的目的是增大这些影响。）

本文使用 temperature scaling 方法来平滑输出的概率分布，从而保留高置信帧的影响，平滑后的输出分布为：$$\mathbf{P}_{\cdot \mathbf{j}}=\frac{\exp \left(\mathbf{O}_{\cdot \mathbf{j}} / T\right)}{\sum_{j^{\prime}=1}^C \exp \left(\mathbf{O}_{\cdot \mathbf{j}^{\prime}} / T\right)}$$
其中，$\mathbf{O}_{\cdot \mathbf{j}}$ 为第 $j$ 类在所有帧的输出 logits，$T$ 大于 $1$。

#### 总训练目标函数

为了防止过拟合，在两个损失中进行了优化：$$\mathcal{L}=\alpha \mathcal{L}_{e m}+(1-\alpha) \mathcal{L}_{m c c}$$
其中，$\alpha$ 为超参数。

算法的总体流程如下：![](./image/Pasted%20image%2020230415110135.png)
从语音 $x$ 开始，计算得到模型的输出 $\mathbf{O}$，然后进行 temperature smoothing，通过最小化训练损失来更新参数 $\theta_{\mathbf{a}}$，迭代 $N$ 次之后得到的模型 $g\left(y \mid x ; \theta_{\mathbf{f}}, \theta_{\mathbf{a}}^{\mathbf{N}}\right)$ 用于最后测试集的推理。

## 实验

### 源 ASR 模型

采用 Wav2vec 2.0-base CTC model 模型，模型在 Librispeech 上预训练，所以 Librispeech 就是源域。

### 数据集

目标域在以下数据集进行测试：
+ Librispeech，用于验证模型是否可以在源域中性能不下降，同时还引入了高斯噪声
+ CHiME-3，真实环境下带噪数据
+ Common voice
+ TEDLIUM-v3

### Baseline TTA 模型

之前没有 single utterance 的TTA，于是用 Single-utterance dynamic pseudo labeling 作为 baseline，通过最小化 CTC loss 来使用 ASR 模型预测伪标签。伪标签通过贪婪算法在每个迭代步骤动态更新。

### 实现细节

设 layer normalization 的参数为 LN，特征提取的参数为 feat，整个模型的参数为 all。

采用 AdamW，学习率搜索发现，对上面的三个部分，最佳的学习率分别是 2e-4, 2e-5, 1e-6（要学习的参数越多， lr 越小）。

所有的实验中，$\alpha=0.3$，迭代次数为 10，$\theta_{\mathbf{a}}$ 为 LN+feat，在 3090 GPU 上训练。
 
### 结果

![](./image/Pasted%20image%2020230415151207.png)
说明：
1. SOTA 结果最好是肯定的，因为是在目标域下训练和测试的
2. SUTA 在所有的数据集中都优于 SDPL 的 baseline
3. 看右边第一列，TTA 好像也可以提高域内样本的性能

![](./image/Pasted%20image%2020230415151918.png)
左边的表给出了消融实验的结果：
1. EM 和 MCC 损失都可以提高性能，$\alpha=0.3$ 是最好的
2. temperature smoothing 非常重要
3. 只 fine tune LN的效果不如 fine tune LN+feat，但是fine tune all 反而导致轻微的性能下降
4. 迭代步数超过10性能会变差
5. 音频长度也有影响，小于 2s 的音频WER 相比于长于 2s 的音频的 WER 更低

