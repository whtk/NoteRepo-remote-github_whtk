> ICASSP 2021，NVIDIA

1. 提出 FastPitch，基于 FastSpeech，完全并行的基于基频曲线的 TTS 模型，
2. 在推理的时候预测 pitch contours 来生成更 expressive 的语音

> 和 FastSpeech 2 很相似，在非自回归模型中引入 duration 和 pitch，但是这两篇论文是同一时间发出的。

## Introduction

1. 提出 FastPitch，基于 FastSpeech，相比于 WaveGlow，可以合成 mel 谱 60× faster than real-time，且对于每一个 symbol 都预测一个 pitch value，从而可以很容易地交互式地调整 pitch

## 模型架构

主要由包含两层 FFT 层的 FastSpeech 组成：
![](image/Pasted%20image%2020240117094141.png)

第一层的 FFT 输入为 token，第二层的 FFT 输出为 frame。令 $(x_1,\ldots,x_n)$ 为 输入的 lexical units，$\begin{aligned}\boldsymbol{y}=(y_1,\ldots,y_t)\end{aligned}$ 为 target mel 谱帧，第一个 FFT 计算表征 $\boldsymbol{h}=\operatorname{FFTr}(\boldsymbol{x})$，然后表征 $\boldsymbol{h}$ 采用 1D CNN 来预测每个 character 的 duration 和 average pitch：
$$\hat{d}=\text{DurationPredictor}(h),\quad\hat{p}=\text{PitchPredictor}(h)$$
其中 $\hat{\boldsymbol{d}}\in\mathbb{N}^n\mathrm{~and~}\hat{\boldsymbol{p}}\in\mathbb{R}^n$，然后将 pitch 投影到 $\boldsymbol{h}$ 的维度，然后和 $\boldsymbol{h}$ 相加，得到的结果 $\boldsymbol{g}$ 直接上采样然后通过第二层的 FFT 产生 mel 谱序列：
$$\begin{aligned}
&g=h+\text{PitchEmbedding}(p) \\
&\hat{\boldsymbol{y}}=\text{FFTr}([\underbrace{g_1,\ldots,g_1}_{d_1},\ldots\underbrace{g_n,\ldots,g_n}_{d_n}]).
\end{aligned}$$
损失采用的是 MSE：
$$\mathcal{L}=\|\hat{\boldsymbol{y}}-\boldsymbol{y}\|_2^2+\alpha\|\hat{\boldsymbol{p}}-\boldsymbol{p}\|_2^2+\gamma\|\hat{\boldsymbol{d}}-\boldsymbol{d}\|_2^2$$

### 输入序列的 duration

输入序列的 duration 通过 Tacotron 2 估计，设 $A\in\mathbb{R}^{n\times t}$ 为 attention 矩阵，第 $i$ 个序列的 duration 为 $\begin{aligned}d_i=\sum_{c=1}^t[\arg\max_rA_{r,c}=i]\end{aligned}$，FastPitch 对于对齐的质量很鲁棒，不同的 Tacotron 2 模型提取的 duration 可能不同，但是合成的质量相似。

### 输入序列的 pitch

用 accurate autocorrelation 方法通过 acoustic periodicity detection 得到 GT pitch。设置窗口的大小使其匹配 mel 谱的 resolution，从而每帧都会得到一个 $F_0$。然后根据 duration $d$ 在每个 symbol 上求 $F_0$ 的平均值。

## 相关工作（略）

## 实验（略）