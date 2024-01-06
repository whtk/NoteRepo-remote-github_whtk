> NIPS 2022，清华、人大

1. diffusion 的采样很慢，而采样过程可以看成是求解 diffusion 过程对应的 ODE 方程
2. 本文提出一个求解 diffusion ODE 的确定的公式，可以解析地求 解 的线性部分，而非直接把所有的项都用黑盒 ODE solver 来解
3. 采用 hange-of-variable，解可以简化为神经网络的指数加权平均
4. 提出 DPM-solver，一种快速的高阶 diffusion ODE solver，同时适用于离散和连续时间的 ODE

## Introduction

DPM 可以定义为离散随机过程或者连续时间的 SDE，DPM 不仅可以实现确切的似然计算，还可以得到更好的采样质量。但是需要上千步的采样，而这也成为了 DPM 的瓶颈。

现有的 DPM 快速采样方法可以分为两类：
+ 知识蒸馏、noise level 或 sample trajectory learning，但是在使用之前需要复杂的训练过程，灵活性也受限
+ training-free sampler，包括 implicit 或者 analytical 的生成过程、advanced differential equation (DE) solvers、动态编程 等，但是还是需要大改 50 次的生成过程

本文拓展 training-free samplers，可以实现 few-step sampling，只需 10 步左右的迭代。将 DPM 的采样看成是求解对应的 ODE 方程。diffusion ODE 有 semi-linear structure，包含数据变量的线性函数和一个由神经网络参数化的非线性函数，而之前的 training-free samplers 都忽略这种结构，直接使用 black-box DE solvers。于是本文通过分析计算线性部分的解来得到其确切的公式解，从而避免了量化误差。同时，通过采用 change-of-variable，解可以简化为神经网络的指数加权积分。

最后提出 DPM-solver，一种快速的 用于 diffusion ODE 的 solver，也提出对应使用的 adaptive step size schedule，可以使 DPM-solver 同时适用于连续和离散时间的 DPM，也可用于 classifier guidance 的条件采样。

## DPM

### 前向过程和 diffusion SDE

设 $D$ 维随机变量 $\boldsymbol{x}_0$，其分布为 $q_0(\boldsymbol{x}_0)$，DPM 定义 前向过程 $\{\boldsymbol{x}_t\}_{t\in[0,T]}$ 从 $\boldsymbol{x}_0$ 开始，使得对任意的 $t\in[0,T]$，$\boldsymbol{x}_t$ 基于条件 $\boldsymbol{x}_0$ 的分布满足：
$$q_{0t}(\boldsymbol{x}_t|\boldsymbol{x}_0)=\mathcal{N}(\boldsymbol{x}_t|\alpha(t)\boldsymbol{x}_0,\sigma^2(t)\boldsymbol{I})$$
其中，$\alpha(t),\sigma(t)\in\mathbb{R}^+$ 是关于 $t$ 的可微函数，简写为 $\alpha_t,\sigma_t$，这两个参数的选择对应于 noise schedule。同时定义 $q_{t}(\boldsymbol{x}_{t})$ 为边缘分布，则选择 schedule 的时候要满足：$q_T(\boldsymbol{x}_T)\approx\mathcal{N}(\boldsymbol{x}_T|\boldsymbol{0},\widetilde{\sigma}^2\boldsymbol{I})\text{ for some }\widetilde{\sigma}>0$。

而对于任意的 $t\in[0,T]$，下述 SDE 有着相同的转移分布 $q_{0t}(\boldsymbol{x}_t|\boldsymbol{x}_0)$：
$$\mathrm{d}\boldsymbol{x}_t=f(t)\boldsymbol{x}_t\mathrm{d}t+g(t)\mathrm{d}\boldsymbol{w}_t,\quad\boldsymbol{x}_0\sim q_0(\boldsymbol{x}_0)$$
其中，$\boldsymbol{w}_t\in\mathbb{R}^D$ 为标准的维纳过程，且：
$$f(t)=\frac{\mathrm{d}\log\alpha_t}{\mathrm{d}t},\quad g^2(t)=\frac{\mathrm{d}\sigma_t^2}{\mathrm{d}t}-2\frac{\mathrm{d}\log\alpha_t}{\mathrm{d}t}\sigma_t^2$$
在某些条件下，上式的前向过程有等效的从 $T$ 到 $0$ 的反向过程，此时从边缘分布 $q_T(\boldsymbol{x}_T)$ 开始：
$$\mathrm{d}x_t=[f(t)\boldsymbol{x}_t-g^2(t)\nabla_{\boldsymbol{x}}\log q_t(\boldsymbol{x}_t)]\mathrm{d}t+g(t)\mathrm{d}\bar{\boldsymbol{w}}_t,\quad x_T\sim q_T(\boldsymbol{x}_T)$$
其中，$\bar{\boldsymbol{w}}_t$ 为反向时间的标准维纳过程。上式中唯一的位置项就是 score function $\nabla_{\boldsymbol{x}}\log q_t(\boldsymbol{x}_t)$，实际使用时，DPM 采用神经网络 $\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t,t)$ 来估计缩放后的 score function $-\sigma_t\nabla_{\boldsymbol{x}}\log q_t(\boldsymbol{x}_t)$，而参数 $\theta$ 通过下述目标函数进行优化：
$$\begin{gathered}
\mathcal{L}(\theta;\omega(t)) :=\frac12\int_0^T\omega(t)\mathbb{E}_{q_t(\boldsymbol{x}_t)}\left[\|\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t,t)+\sigma_t\nabla_{\boldsymbol{x}}\log q_t(\boldsymbol{x}_t)\|_2^2\right]\mathrm{d}t \\
=\frac12\int_0^T\omega(t)\mathbb{E}_{q_0(\boldsymbol{x}_0)}\mathbb{E}_{q(\boldsymbol{\epsilon})}\left[\|\epsilon_\theta(\boldsymbol{x}_t,t)-\boldsymbol{\epsilon}\|_2^2\right]\mathrm{d}t+C, 
\end{gathered}$$
其中的 $\omega(t)$ 为加权函数，而 $\boldsymbol{\epsilon}\sim q(\boldsymbol{\epsilon})=\mathcal{N}(\boldsymbol{\epsilon}|\boldsymbol{0},\boldsymbol{I}),\boldsymbol{x}_t=\alpha_t\boldsymbol{x}_0+\sigma_t\boldsymbol{\epsilon}$ ，$C$ 为常数。此时参数化的反向过程可以从 $T$ 到 $0$，开始分布为 $\boldsymbol{x}_T\sim\mathcal{N}(\boldsymbol{0},\tilde{\sigma}^2\boldsymbol{I})$ ，此过程写为：
$$\mathrm{d}\boldsymbol{x}_t=\left[f(t)\boldsymbol{x}_t+\frac{g^2(t)}{\sigma_t}\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t,t)\right]\mathrm{d}t+g(t)\mathrm{d}\bar{\boldsymbol{w}}_t,\quad\boldsymbol{x}_T\sim\mathcal{N}(\mathbf{0},\tilde{\sigma}^2I)$$
通过用数值求解器求解上述 SDE 方程即可采样生成样本，这个过程将 SDE 从 $T$ 到 $0$ 离散化。

传统的采样过程可以看成是一阶 SDE solver，但是需要成百上千次迭代。

### diffusion ODE

离散 SDE 时，step size 受限于维纳过程的随机性，step size 过大通常导致无法收敛，尤其是在高维空间。为了实现快速采样，一种方法是考虑相关的 概率流 ODE，其在时刻 $t$ 有着和 SDE 相同的边缘分布。具体来说，反向过程对应的概率流 ODE 为：
$$\frac{\mathrm{d}x_t}{\mathrm{d}t}=f(t)\boldsymbol{x}_t-\frac12g^2(t)\nabla_{\boldsymbol{x}}\log q_t(\boldsymbol{x}_t),\quad\boldsymbol{x}_T\sim q_T(\boldsymbol{x}_T)$$
其中边缘分布也为 $q_T(\boldsymbol{x}_T)$。通过将神经网络模型替换其中的 score function，定义参数化的 ODE 为：
$$\frac{\mathrm{d}\boldsymbol{x}_t}{\mathrm{d}t}=\boldsymbol{h}_\theta(\boldsymbol{x}_t,t):=f(t)\boldsymbol{x}_t+\frac{g^2(t)}{2\sigma_t}\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t,t),\quad\boldsymbol{x}_T\sim\mathcal{N}(\mathbf{0},\tilde{\sigma}^2\boldsymbol{I})$$
通过从 $T$ 到 $0$ 求解这个 ODE，即可进行采样。

和 SDE 相比，ODE 可以实现更大的 step size，因为其没有随机性了。同时也可以采用一些高效的数值 ODE solver 来加速采样。例如，采用  RK45 ODE solver 可以在大概 60 次迭代下生成样本，其质量可和 SDE 的 1000 次迭代相比。

但是要实现 10 步左右的采样，现有的 ODE 还是无法生成满意的样本。

## 用于 diffusion ODE 的 Customized Fast Solvers

### diffusion ODE 的确切解的简化公式

给定 时间 $s>0$ 的初始值 $\boldsymbol{x}_s$，ODE 方程的每个时间 $t<s$ 的解 $\boldsymbol{x}_t$ 能够简化为非常确切的公式，从而可以被有效地近似。

 diffusion ODE 方程等式右边包含两部分：
 + $f(t)\boldsymbol{x}_t$ 是一个关于 $\boldsymbol{x}_t$ 的线性函数
 + $\frac{g^2(t)}{2\sigma_t}\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t,t)$ 是一个关于 $\boldsymbol{x}_t$ 的非线性函数

这种 ODE 被称为 semi-linear ODE。之前的 solver 忽略了这种结构，然后把 $\boldsymbol{h}_\theta(\boldsymbol{x}_t,t)$ 整体当作输入，从而导致线性和非线性部分都会存在离散误差。

但是，时刻 $t$ 的解可以确定地写为：
$$\boldsymbol{x}_t=e^{\int_s^tf(\tau)\mathrm{d}\tau}\boldsymbol{x}_s+\int_s^t\left(e^{\int_\tau^tf(r)\mathrm{d}r}\frac{g^2(\tau)}{2\sigma_\tau}\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_\tau,\tau)\right)\mathrm{d}\tau$$
这个公式解耦了线性和非线性部分，和  black-box ODE solver 相反，线性部分可以被精确计算。但是非线性部分的积分还是很复杂，很难估计。

不过，通过引入一个特别的变量 $\lambda_{t}:=\log(\alpha_{t}/\sigma_{t})$ (其实就是 log-SNR 的二分之一)，非线性部分的积分项可以被极大地简化。可以将公式：
$$f(t)=\frac{\mathrm{d}\log\alpha_t}{\mathrm{d}t},\quad g^2(t)=\frac{\mathrm{d}\sigma_t^2}{\mathrm{d}t}-2\frac{\mathrm{d}\log\alpha_t}{\mathrm{d}t}\sigma_t^2$$
中的 $g(t)$ 从写为：
$$g^2(t)=\frac{\mathrm{d}\sigma_t^2}{\mathrm{d}t}-2\frac{\mathrm{d}\log\alpha_t}{\mathrm{d}t}\sigma_t^2=2\sigma_t^2\left(\frac{\mathrm{d}\log\sigma_t}{\mathrm{d}t}-\frac{\mathrm{d}\log\alpha_t}{\mathrm{d}t}\right)=-2\sigma_t^2\frac{\mathrm{d}\lambda_t}{\mathrm{d}t}$$
结合 $f(t)=\frac{\mathrm{d}\log\alpha_t}{\mathrm{d}t}$，解可以进一步写为：
$$\boldsymbol{x}_t=\frac{\alpha_t}{\alpha_s}\boldsymbol{x}_s-\alpha_t\int_s^t\left(\frac{\mathrm{d}\lambda_\tau}{\mathrm{d}\tau}\right)\frac{\sigma_\tau}{\alpha_\tau}\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_\tau,\tau)\mathrm{d}\tau$$
由于 $\lambda(t)=\lambda_{t}$ 是一个关于 $t$ 的严格递减函数（实际上只要满足一对一映射即可），其逆函数 $t_{\lambda}(\cdot)$ 满足 $t=t_{\lambda}(\lambda(t))$，此时将 $\boldsymbol{x}_t$ 和 $\boldsymbol{\epsilon}_\theta$ 的下标进行替换，即 定义：$$\hat{\boldsymbol{x}}_\lambda:=\boldsymbol{x}_{t_\lambda(\lambda)},\hat{\boldsymbol{\epsilon}}_\theta(\hat{\boldsymbol{x}}_\lambda,\lambda):=\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_{t_\lambda(\lambda)},t_\lambda(\lambda))$$
通过 change-of-variable，有：
给定 初始值 $\boldsymbol{x}_s,s>0$，ODE 方程的每个时间 $t<s$ 的解 $\boldsymbol{x}_t$ 为：
$$\boldsymbol{x}_t=\frac{\alpha_t}{\alpha_s}\boldsymbol{x}_s-\alpha_t\int_{\lambda_s}^{\lambda_t}e^{-\lambda}\hat{\boldsymbol{\epsilon}}_\theta(\hat{\boldsymbol{x}}_\lambda,\lambda)\mathrm{d}\lambda$$
称积分项 $\int e^{-\lambda}\hat{\boldsymbol{\epsilon}}_\theta(\hat{\boldsymbol{x}}_\lambda,\lambda)\mathrm{d}\lambda$ 为 $\hat{\boldsymbol{\epsilon}}_\theta$ 的指数加权积分，和传统的 ODE solver 中的指数积分很相似。

上式提供了一种全新的视角来近似 diffusion ODE 的解。即给定时刻 $s$ 的 $\boldsymbol{x}_s$，近似时刻 $t$ 的解 $\boldsymbol{x}_t$ 可以直接通过近似 $\hat{\boldsymbol{\epsilon}}_\theta$ 从 $\lambda_s$ 到 $\lambda_t$ 的指数加权积分得到。

基于上面的发现，下面提出 用于 diffusion ODE 的 fast solver。

### 用于 diffusion ODE 的 high-order solver

给定时刻 $T$ 的初始值 $\boldsymbol{x}_t$ 和  $M+1$ 个 time step $\{t_i\}_{i=0}^{M}$ 逐步从 $T$ 到 $0$ 递减。令 $\tilde{\boldsymbol{x}}_{t_0}=\boldsymbol{x}_T$ 为初始值，则提出的 solver 用 $M$ 步迭代计算序列 $\{\tilde{\boldsymbol{x}}_{t_i}\}_{i=0}^M$ 来近似真实的解 $\{t_i\}_{i=0}^{M}$，特别地，最后一次迭代得到 $\tilde{x}_{t_{M}}$ 近似时刻 $0$ 的解。

为了减少每一步的近似误差，从 前一个 位于时间 $t_{i-1}$ 的 $\tilde{\boldsymbol{x}}_{t_{i-1}}$ 开始，根据上一节的公式，时间 $t_i$ 的确切解 $\boldsymbol{x}_{t_{i-1}\rightarrow t_{i}}$ 为：
$$x_{t_{i-1}\to t_i}=\frac{\alpha_{t_i}}{\alpha_{t_{i-1}}}\tilde{x}_{t_{i-1}}-\alpha_{t_i}\int_{\lambda_{t_{i-1}}}^{\lambda_{t_i}}e^{-\lambda}\hat{\boldsymbol{\epsilon}}_\theta(\hat{\boldsymbol{x}}_\lambda,\lambda)\mathrm{d}\lambda$$
也就是需要近似 $\hat{\boldsymbol{\epsilon}}_\theta$ 从 $\lambda_{t_{i-1}}$ 到 $\lambda_{t_i}$ 的指数加权积分。定义 $h_i:=\lambda_{t_i}-\lambda_{t_{i-1}}$，$\hat{\boldsymbol{\epsilon}}_\theta^{(n)}(\hat{\boldsymbol{x}}_\lambda,\lambda):=\frac{\mathrm{d}^n\hat{\boldsymbol{\epsilon}}_\theta(\hat{\boldsymbol{x}}_\lambda,\lambda)}{\mathrm{d}\lambda^n}$ 为对 $\hat{\boldsymbol{\epsilon}}_\theta(\hat{\boldsymbol{x}}_\lambda,\lambda)$ 关于 $\lambda$ 的 $n$ 阶微分，那么，对于 $k\ge 1$，$\hat{\boldsymbol{\epsilon}}_\theta(\hat{\boldsymbol{x}}_\lambda,\lambda)$ 关于 $\lambda_{t_{i-1}}$ 的 $k-1$ 阶泰勒展开为：
$$\hat{\boldsymbol{\epsilon}}_\theta(\hat{\boldsymbol{x}}_\lambda,\lambda)=\sum_{n=0}^{k-1}\frac{(\lambda-\lambda_{t_{i-1}})^n}{n!}\hat{\boldsymbol{\epsilon}}_\theta^{(n)}(\hat{\boldsymbol{x}}_{\lambda_{t_{i-1}}},\lambda_{t_{i-1}})+\mathcal{O}((\lambda-\lambda_{t_{i-1}})^k)$$
将其带入 $\boldsymbol{x}_{t_{i-1}\rightarrow t_{i}}$ 的计算，有：
$$\boldsymbol{x}_{t_{i-1}\to t_i}=\frac{\alpha_{t_i}}{\alpha_{t_{i-1}}}\boldsymbol{\tilde{x}}_{t_{i-1}}-\alpha_{t_i}\sum_{n=0}^{k-1}\hat{\boldsymbol{\epsilon}}_\theta^{(n)}(\hat{\boldsymbol{x}}_{\lambda_{i-1}},\lambda_{t_{i-1}})\int_{\lambda_{t_{i-1}}}^{\lambda_{t_i}}e^{-\lambda}\frac{(\lambda-\lambda_{t_{i-1}})^n}{n!}\mathrm{d}\lambda+\mathcal{O}(h_i^{k+1})$$
其中 $\int e^{-\lambda}\frac{(\lambda-\lambda_{t_{i-1}})^n}{n!}\mathrm{d}\lambda$ 可以连用 $n$ 次分部积分计算。 因此，为了近似 $\boldsymbol{x}_{t_{i-1}\rightarrow t_{i}}$，我们只需计算 $n$ 阶微分 $\hat{\epsilon}_\theta^{(n)}(\hat{x}_\lambda,\lambda)$ ，而这个问题已经在 ODE 的相关文献中得到很好的研究了。丢掉误差项 $\mathcal{O}(h_i^{k+1})$ 来近似前 $k-1$ 阶总微分，可以得到 diffusion ODE 的 $k$ 阶 ODE solver，称之为 DPM-solver。

当 $k=1$ 时，上式为：
$$\begin{gathered}
\boldsymbol{x}_{t_{i-1}\rightarrow t_{i}} =\frac{\alpha_{t_i}}{\alpha_{t_{i-1}}}\tilde{\boldsymbol{x}}_{t_{i-1}}-\alpha_{t_i}\boldsymbol{\epsilon}_\theta(\tilde{\boldsymbol{x}}_{t_{i-1}},t_{i-1})\int_{\lambda_{t_{i-1}}}^{\lambda_{t_i}}e^{-\lambda}\mathrm{d}\lambda+\mathcal{O}(h_i^2) \\
=\frac{\alpha_{t_i}}{\alpha_{t_{i-1}}}\widetilde{\boldsymbol{x}}_{t_{i-1}}-\sigma_{t_i}(e^{h_i}-1)\boldsymbol{\epsilon}_\theta(\widetilde{\boldsymbol{x}}_{t_{i-1}},t_{i-1})+\mathcal{O}(h_i^2). 
\end{gathered}$$
丢掉 $\mathcal{O}(h_i^2)$ 之后即可得到最终的近似解。

DPM-solver-1：给定初始值 $\boldsymbol{x}_T$ 和 $M+1$ time step $\{t_i\}_{i=0}^M$ 从 $t_0=T$ 到 $t_M=0$，那么，从 $\tilde{\boldsymbol{x}}_{t_0}=\boldsymbol{x}_T$ 开始，序列 $\{\tilde{\boldsymbol{x}}_{t_i}\}_{i=0}^M$ 迭代计算如下：
$$\tilde{\boldsymbol{x}}_{t_i}=\frac{\alpha_{t_i}}{\alpha_{t_{i-1}}}\tilde{\boldsymbol{x}}_{t_{i-1}}-\sigma_{t_i}(e^{h_i}-1)\boldsymbol{\epsilon}_{\theta}(\tilde{\boldsymbol{x}}_{t_{i-1}},t_{i-1}),\quad\mathrm{~where~}h_i=\lambda_{t_i}-\lambda_{t_{i-1}}$$

### step size schedule

### 从离散 DPM 中采样

## 与现有的快速采样方法比较

### DDIM vs DPM-Solver-1

### 和传统的 Runge-Kutta 方法比较

### Training-based Fast Sampling Methods for DPMs

## 实验（略）