> Speaker Recognition from Raw Waveform with SincNet 笔记，Bengio 大佬2019年的文章

1. 提出一种新的 CNN 架构——SincNet，在第一层卷积中探索更有意义的滤波器
2. SincNet 基于 parametrized sinc functions ，实现了 带通滤波器
3. CNN 学习每个滤波器的所有系数，而 SincNet 只学习带通滤波器的 low 和 high 的截止频率
4. 在说话人验证和说话人识别任务上表明，提出的架构在原始波形作为输入时比标准的 CNN 收敛快，性能好

## Introduction

基于波形的 CNN 的关键部分是第一层卷积层，这一层不仅处理高维输入（也就是输入波形，因为声纹可以看作是高维语音波形的低维流形——lin），而且受梯度消失问题的影响。

CNN 学习的滤波器通常是不连续的、有噪声的多波段形状，不符合人类的直觉，从而影响语音信号的表征能力。

本文则对滤波器的形状添加一些约束，相比于 CNN，SincNet 将输入波形和一组参数化的带通滤波器 sinc 函数进行卷积（标准的滤波器频率特性取决于一系列的滤波器系数），而 SincNet 中只需要低频和高频截止频率两个参数。

同时这种方法有着很高的灵活性，迫使网络将重点放在滤波器的带宽上，其泛化性更强。

最终效果优于基于 i-vector 的说话人识别系统。

## SincNet 架构

SincNet 架构如图：![](./image/Pasted%20image%2020221128213927.png)

对于标准的 CNN（一维），卷积过程为：$$y[n]=x[n] * h[n]=\sum_{l=0}^{L-1} x[l] \cdot h[n-l]$$
其中，$x[n]$ 是输入语音信号，$h[n]$ 为长 $L$ 的滤波器，标准的 CNN 中，长为 $L$ 的滤波器系数为待学习的参数。相反，提出的 SincNet 使用预先定义好的函数 $g$ 进行卷积，其参数为 $\theta$，输出计算为：$$y[n]=x[n] * g[n, \theta]$$
选择 $g$ 由矩形带通滤波器组成，频域中，带通滤波器可以写成两个低通滤波器的差：$$G\left[f, f_1, f_2\right]=\operatorname{rect}\left(\frac{f}{2 f_2}\right)-\operatorname{rect}\left(\frac{f}{2 f_1}\right)$$
$f_1,f_2$ 分别代表待学习的低频和高频截止频率，$\operatorname{rect}(\cdot)$ 为矩形函数，其对应的时域表示可以写成：$$g\left[n, f_1, f_2\right]=2 f_2 \operatorname{sinc}\left(2 \pi f_2 n\right)-2 f_1 \operatorname{sinc}\left(2 \pi f_1 n\right)$$
其中，sinc 函数定义为 $\operatorname{sinc}(x)=\sin (x) / x$ 。

使用过程中，截止频率可以在 $\left[0, f_s / 2\right]$ 随机初始化，$f_s$ 表示信号的采样频率，也可以使用 Mel 频率下的滤波器组进行初始化。为了确保 $f_1 \geq 0,f_2 \geq f_1$，实际上是通过以下参数得到：$$\begin{aligned}
&f_1^{a b s}=\left|f_1\right| \\
&f_2^{a b s}=f_1+\left|f_2-f_1\right|
\end{aligned}$$
作者发现，训练过程中，模型自然会满足约束，所以没有施加限制来使得 $f_2$ 小于奈奎斯特频率，同时不会学习滤波器的参数（也就是默认幅度为 1），因为幅度可以在后续的层中自动调整。

但是， 理想的带通滤波器长度 $L$ 是无限的，如果直接进行截断都使得滤波器变成近似的而不是和理论完全一致，因此通过加窗来解决：$$g_w\left[n, f_1, f_2\right]=g\left[n, f_1, f_2\right] \cdot w[n]$$
论文使用了 Hamming 窗。Hamming 窗特别适合实现高频选择性，但是实际上使用其他窗口，性能没有显著差异。

SincNet 的所有操作都是可微的，滤波器的截止频率也可以通过 SGD 来进行优化。

通过一个 Sinc 层之后，就可以接上标准的 CNN 网络得到最终的模型。

## SincNet 的优点

1. 快速收敛：SincNet 使得网络只关注对性能影响较大的滤波器参数，同时保持了适应数据的灵活性，利用了 滤波器的先验知识（矩形滤波器），有助于 SincNet 更好的收敛
2. 参数更少：显然减少了第一层卷积的参数，同时适用于 few sample 的训练场景
3. 可解释性：滤波器的参数更具有物理意义

## 相关工作（略）

## 实验和结果
标准 CNN 滤波器和 使用 SincNet 得到的滤波器的对比：![](./image/Pasted%20image%2020221128214630.png)

其他略。