> InterSpeech 2022，Kakao Enterprise Corporation，韩国

1. TTS 中的 two-stage 效果很好，但是训练流程过于复杂
2. 提出端到端的 TTS，联合训练  FastSpeech2 和 HiFi-GAN，通过在联合训练框架中引入一个对齐学习目标函数，移除了外部的 speech-text 对齐工具
3. MOS 在 LJSpeech 上超过了 ESPNet2-TTS 

> 缝合怪，FastSpeech 2 + HiFi-GAN + one TTS alignment。

## Introduction

1. 两阶段的 TTS 通常会导致合成质量的降低
2. 端到端的 TTS 可以直接从文本生成波形，不存在 acoustic feature mismatch 的问题
3. 提出 E2E-TTS，和 ESPNet2 工具包很类似，但是不存在中间的 mel 谱特征，同时还引入了一个对齐学习目标函数

## 相关工作（略）

## 方法

### FastSpeech 2

见 [FastSpeech 2- Fast and High-Quality End-to-End Text to Speech 笔记](../FastSpeech%202-%20Fast%20and%20High-Quality%20End-to-End%20Text%20to%20Speech%20笔记.md) 。结构如图：
![](image/Pasted%20image%2020240107095521.png)

训练的时候移除了 mel 谱 loss，此时的 variance loss 变为：
$$L_{var}=||\mathbf{d}-\mathbf{\hat{d}}||_2+||\mathbf{p}-\mathbf{\hat{p}}||_2+||\mathbf{e}-\mathbf{\hat{e}}||_2$$
其中的 $\mathbf{d},\mathbf{p},\mathbf{e}$ 分别代表 duration、pitch 和 energy。

### HiFi-GAN

见 [HiFi-GAN- Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis 笔记](../HiFi-GAN-%20Generative%20Adversarial%20Networks%20for%20Efficient%20and%20High%20Fidelity%20Speech%20Synthesis%20笔记.md) 。

其中 generator 的 loss 为：
$$L_g=L_{g,adv}+\lambda_{fm}L_{fm}+\lambda_{mel}L_{mel}$$

### 对齐学习框架

用的是 [One TTS Alignment To Rule Them All 笔记](../对齐/One%20TTS%20Alignment%20To%20Rule%20Them%20All%20笔记.md) 的对齐学习框架，在训练的时候 on the fly 地获取 duration。

用一个对齐模块，将 text embedding 和 mel 谱 编码到 $\mathbf{h}^{\boldsymbol{e}nc},\mathbf{m}^{\boldsymbol{e}nc}$，然后计算 soft alignment $\mathcal{A}_{soft}$ 如下：
$$\begin{aligned}D_{i,j}&=dist_{L2}(\mathbf{h}_i^{enc},\mathbf{m}_j^{enc})\\\mathcal{A}_{soft}&=\text{softmax}(-D,dim=0)\end{aligned}$$
然后再计算所有的单调对齐的似然的和，作为对齐学习的目标函数：
$$P(S(\mathbf{h})|\mathbf{m})=\sum_{\mathbf{s}\in S(\mathbf{h})}\prod_{t=1}^TP(s_t|m_t)$$
这里的 $\mathbf{s}$ 为一个特定的对齐，$S(\mathbf{h})$ 为所有的单调对齐的集合。用的是 forward-sum 算法。
> 在实际使用时，可以直接通过 CTC-loss 实现。

最后，通过 MAS 将 soft alignment 转为 hard alignment $\mathcal{A}_{hrad}$，然后 $\sum_{j=1}^T\mathcal{A}_{hard,i,j}$ 即可得到 duration。然后有一个额外的 binarization loss 定义为两个 alignment 之间的 KL 散度：
$$\begin{aligned}L_{bin}&=-\mathcal{A}_{hard}\odot\log\mathcal{A}_{soft}\\L_{align}&=L_{forward\_sum}+L_{bin}\end{aligned}$$

### 最终损失

模型直接从文本合成波形，中间不经过 mel 谱 特征 及相关损失，总的损失为：
$$L=L_g+\lambda_{var}L_{var}+\lambda_{align}L_{align}$$
其中，$\lambda_{var}=1,\lambda_{align}=2$。