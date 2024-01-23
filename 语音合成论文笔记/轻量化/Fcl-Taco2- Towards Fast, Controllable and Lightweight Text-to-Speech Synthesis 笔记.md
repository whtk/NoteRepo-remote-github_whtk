> ICASSP 2021，CUHK，华为诺亚方舟实验室

1. 提出 FCL-taco2，快速、可控、轻量化的 TTS 模型
2. 基于 Tacotron2，采用  semi-autoregressive，基于 prosody feature 来实现 phoneme-level 的并行 mel 谱 生成
3. 采用 KD 将 FCL-taco2 大模型压缩为小模型
4. 中英文数据集实验表明，小的 FCL-taco2 相比于 Tacotron2 可以实现 comparable 的性能，但是快了 18.5 倍

## Introduction

1. 之前的工作主要基于 vocoder 的优化，而很少关注 acoustic model
2. 提出 FCL-taco2，是 Tacotron2 的 semi-autoregressive 版本，采用显式的韵律建模，从三个方面对 Tacotron2 进行提升：
	1. 以 SAR 的方式预测 mel 谱，即 AR 方式生成独立的 phoneme，NAR 的方式生成不同的 phoneme
	2. 提出 prosody injector 来推理 phoneme duration、pitch 和 energy，从而实现灵活的韵律控制
	3. 采用 知识蒸馏（KD） 进行知识迁移

## 相关工作（略）

## Tacotron2

见 [Tacotron 2- Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions 笔记](../Tacotron%202-%20Natural%20TTS%20Synthesis%20by%20Conditioning%20WaveNet%20on%20Mel%20Spectrogram%20Predictions%20笔记.md)，包含 encoder 和一个带有 location-sensitive attention 的 decoder。

## 方法

如图：
![](image/Pasted%20image%2020240123102526.png)
包含三个关键组成：
+ encoder
+ prosody injector
+ decoder

encoder 结构类似 Tacotron2 中的，输入 phoneme 序列 $X{=}\{x_n\}_{1\leq n\leq N}$，得到 hidden representation：
$$\mathbf{H=Encoder(X)}$$

prosody injector 包含三个 prosody predictor，用于预测 duration, pitch 和energy，其结构相同（2 Conv + 1 FC Layer），输入为 $\mathbf{H}$，预测：
$$\begin{aligned}\mathbf{D}&=\text{Duration-Predictor}(\mathbf{H})\\\mathbf{F}&=\text{Pitch-Predictor}(\mathbf{H})\\\mathbf{E}&=\text{Energy-Predictor}(\mathbf{H})\end{aligned}$$
然后通过对应的 embedding 层投影到相同的维度进行相加：
$$\mathbf{G=H+Pitch-Embed}(\mathbf{F})+\operatorname{Energy-Embed}(\mathbf{E})$$

decoder 输入为 $\mathbf{G}$，以 SAR 的方式生成 mel 谱，具体来说，对应于第 $n$ 个 phoneme 的 mel 谱 $\mathbf{Y}_n=\{\mathbf{y}_{n,m}\}_{1\leq m\leq d_n}$ 以自回归的方式生成：
$$\mathbf{y}_{n,m}=\text{Decoder}\left(\mathbf{g}_n,p_{n,m},\mathbf{y}_{n,m-1}\right)$$
也就是说，第 $n$ 和 phoneme 的第 $m$ 个 frame 基于 $\mathbf{g}_n,p_{n,m}$ 和前一个 frame $\mathbf{y}_{n,m-1}$ 生成，其中的 $p_{n,m}$ 是一个 0-1 之间的值表示 frame 的相对位置来辅助自回归生成，生成的长度取决于第 $n$ 个 frame 的 duration $d_n$。在所有的 phonemes 中 decoder 是共享的，从而不同 phoneme 的 mel 谱可以并行生成。最后拼接起来得到 $\mathbf{Y=}\left\{\mathbf{Y}_n\right\}_{1\leq n\leq N}$，然后通过 post-net 得到最终的 mel 谱 输出：
$$\mathbf{Z}=\mathbf{Y}+\text{Post-net}\mathbf{\left(Y\right)}$$

### 两阶段学习策略

包含：
+ 训练大小和 Tacotron2 类似的 teacher 模型 FCL-taco2-T
+ 用 KD 压缩得到 student 模型

FCL-taco2-T 的 encoder, decoder and Post-net 结构和 Tacotron2 相同，但是没用 attention 模块和 stop token 预测。其损失为：
$$L^{GT}=\lambda_1L_m^{GT}+\lambda_2L_d^{GT}+\lambda_3L_f^{GT}+\lambda_4L_e^{GT}$$
其中，$L_m^{GT}$ 是 GT mel 谱 和预测的 mel 谱的 L1 和 L2-Norm 组合损失，$L_d^{GT},L_f^{GT},L_e^{GT}$ 分别是 duration, pitch 和 energy 的 L2-Norm 损失，$\lambda_1,\lambda_2,\lambda_3,\lambda_4$ 是对应的权重。GT duration 通过 MFA 提取，GT pitch 和 energy 通过对根据 duration 对每个 phoneme 取平均得到。

而对于 student 模型，其层数和 FCL-taco2-T 相同，但是维度减少，采用高效的蒸馏策略：mel-spectrogram distillation (MSD), hidden representation distillation (HRD) and prosody distillation (PD)：
+ MSD 为 sequence-level 蒸馏，使得输出的 mel 谱 尽可能相似，从而 $L_{MSD}$ 为 teacher 和 student 的 mel 谱的 L1 和 L2-Norm 组合损失之和。
+ HRD 为 hidden representation 蒸馏，使得 teacher 和 student 的 hidden representation 尽可能相似，其中 hidden representations 包含 encoder 层、decoder 层和 post-net 层的输出，此时损失为 $\begin{aligned}L_{HRD}=\sum_{i\in I}\left\|\mathbf{K}_{S}^{i}\mathbf{W}^{i}-\mathbf{K}_{T}^{i}\right\|_{2}\end{aligned}$，其中 $\mathbf{W}$ 为可学习的映射矩阵来确保维度相似
+ PD 用于蒸馏两个模型的 prosody 和 embedding，损失为 $\begin{aligned}L_{PD}=L_d^{ST}+L_f^{ST}+L_e^{ST}+\left\|\mathbf{K}_S^f\mathbf{W}^f-\mathbf{K}_T^f\right\|_2+\left\|\mathbf{K}_S^e\mathbf{W}^e-\mathbf{K}_T^e\right\|_2\end{aligned}$，其中 $L_d^{ST},L_f^{ST},L_e^{ST}$ 分别是 student 和 teacher 的 duration, pitch 和 energy 的 L2-Norm 损失，$\mathbf{K}_S^f,\mathbf{K}_S^e$ 分别是 student 的 pitch 和 energy 的 embedding，$\mathbf{K}_T^f,\mathbf{K}_T^e$ 分别是 teacher 的 pitch 和 energy 的 embedding，$\mathbf{W}^f,\mathbf{W}^e$ 分别是可学习的映射矩阵来确保维度相似

训练 student 模型时，总的损失为：
$$L_{student}=\alpha_1L^{GT}+\alpha_2L_{MSD}+\alpha_3L_{HRD}+\alpha_4L_{PD}$$

## 实验和分析（略）