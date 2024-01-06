> ICASSP 论文

1. 提出 RawBoost 数据增强方法，直接对原始波形进行操作
2. 不需要额外的数据源，用于电话场景
3. RawBoost 基于线性和非线性卷积噪声、脉冲信号相关加性噪声和平稳信号无关加性噪声的组合，可以对各种干扰进行建模
4. 使用 RawBoost 之后可以使 baseline 性能提高 27%

## Introduction

1. 现有的增强技术主要是在谱上操作的，不能用于原始的波形输入
2. 引入 RawBoost，可以直接对原始音频进行数据增强，主要是通过音调修改、带阻滤波、添加混响或噪声等方法实现，且不需要额外的数据源（而且是数据、模型无关的）

## 数据增强

ASVspoof 2021 LA 中，很多人用了 speed perturbation、SpecAugment 、codec augmentation 等数据增强方法。

## RawBoost

RawBoost 采用已有的 线性和非线性信号处理方法 对语言进行增强。

### 线性和非线性卷积噪声

任何涉及到 编码、（解）压缩和传输的信道都会引入静态卷积失真，同时也会有非线性的扰动。为了提高对这种扰动的鲁棒性，采用 多频带滤波器和 Hammerstein systems 。Hammerstein systems 模型是用于估计非线性系统的多频带滤波器响应的，但是这里用相同的想法来生成信号失真。

多频带滤波器使用时域陷波滤波来产生卷积噪声。使用一组数量为 $N_{\text{notch}}$ 的陷波滤波器，每个陷波滤波器随机中心频率 $f_c$ 和滤波器宽度 $\Delta f$，然后使用基于窗口的滤波器设计方法定义随机增益为 $g_j^{\text{cn}}$ 的 FIR 滤波器，生成阶数 $N_{\text{fir}}$ 随机的、特征频率响应的滤波器。阶数越高，频率响应就越突然；阶数越低，通带纹波或失真越明显，$N_{\text{notch}}=3$ 的一个例子如下：
![](image/Pasted%20image%2020230513102558.png)

Hammerstein systems 产生高阶谐波引入非线性谐波失真。每个高阶谐波的频率和幅度取决于原始分量的频率和振幅以及非线性系统的特性：![](image/Pasted%20image%2020230513102823.png)
上图中的卷积噪声计算为：$$y_{\mathrm{cn}}[n]=\sum_{j=1}^{N_{\mathrm{f}}} g_j^{\mathrm{cn}} \sum_{i=0}^{N_{\mathrm{fir}_j}} b_{i_j} \cdot x^j[n-i]$$
其中，$x\in[-1,1]^{l\times 1}$ 表示长为 $l$ 的原始波形，$j\in[1,N_\mathrm{f}]$ 为线性或者非线性的阶数，$b_{i_j}$ 为多频带滤波器的系数。

### 脉冲 信号相关 加性噪声

这种噪声通常会在数据采集中引入，将这种扰动建模为非平稳脉冲扰动，由瞬时或脉冲状振幅变化组成：$$y_{\mathrm{sd}}[n]=x[n]+z_{\mathrm{sd}}[n]$$
其中，$$z_{\mathrm{sd}}[n]= \begin{cases}g^{\mathrm{sd}} \cdot D_R\{-1,1\}[n] \cdot x[n], & \text { if } n=\left\{p_1, p_2, \ldots, p_P\right\} \\ 0, & \text { otherwise }\end{cases}$$
是一个信号相关的量。

### 平稳 信号无关 加性噪声

与信号无关的附加噪声可能由松散或连接不良的电缆连接、传输通道效应、电磁干扰或热噪声引起。与脉冲噪声相反，平稳白噪声 $w$ 通过 FIR 滤波器变成色噪声，然后再加入到语言信号中：$$\begin{gathered}
y_{\mathrm{si}}[n]=x[n]+g_{s n r}^{\mathrm{si}} \cdot z_{\mathrm{si}}[n] \\
g_{s n r}^{\mathrm{si}}=\frac{10^{\frac{S N R}{20}}}{\left\|z_{\mathrm{si}}\right\|^2 \cdot\|x\|^2}
\end{gathered}$$
其中，$g$ 是增益因子，$z_{\mathrm{si}}$ 是色噪声。

## 实验和结果

ASVspoof 2021 LA 结果：
![](image/Pasted%20image%2020230513111703.png)
