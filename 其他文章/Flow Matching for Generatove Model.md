> preprint，Meta AI

1. 提出一种建立在 Continuous Normalizing Flows (CNFs) 上的新的生成模型范式，即 Flow Matching
2. Flow Matching 兼容通用的高斯概率路径，用于在噪声和数据样本之间进行转换
3. 同时 为使用其他非扩散概率路径训练 CNF 打开了大门

## Introduction

1. 深度生成模型是旨在估计和采样未知数据分布，diffusion-based models 是一种可扩展且相对稳定的训练方法，但是限制了采样概率路径的空间，导致训练时间很长，需要采用专门的方法来进行采样

2. 本文考虑 Continuous Normalizing Flows 的一般性和确定性框架。提出 Flow Matching，是一种训练 CNF 模型 simulation-free 的方法，采用 通用的概率路径进行训练。且 FM 避免了对 diffusion 过程进行推理，而是直接使用概率路径

3. 提出 Flow Matching objective，一种简单直观的训练目标函数，用于回归到 生成所需的概率路径 的 目标向量场（target vector field）

4. 在 ImageNet 上验证了 Flow Matching 和 Optimal Transport paths 的构造，发现可以轻松训练模型，在似然估计和样本质量之间取得较好的性能。且可以在计算成本和样本质量之间实现 trade-off

## 预备知识：Continuous Normalizing Flows

定义 $\mathbb{R}^d$ 为数据空间，$x = (x_1, \dots, x_d) \in \mathbb{R}^d$ 为数据点。概率密度路径 $p: [0, 1] \times \mathbb{R}^d \rightarrow \mathbb{R}_{>0}$ 是一个时间相关的概率密度函数，即 $\int_{\mathbb{R}^d} p_t(x)dx = 1$。时间相关的向量场为 $v: [0, 1] \times \mathbb{R}^d \rightarrow \mathbb{R}^d$。向量场 $v_t$ 可以用来构造时间相关的微分同胚映射（diffeomorphic map），称为流（flow），$\phi: [0, 1] \times \mathbb{R}^d \rightarrow \mathbb{R}^d$，定义为常微分方程（ODE）：
$$\begin{aligned}\frac d{dt}\phi_t(x)&=v_t(\phi_t(x))\\\phi_0(x)&=x\end{aligned}$$

[Neural Ordinary Differential Equations 笔记](Neural%20Ordinary%20Differential%20Equations%20笔记.md) 已经表明，可以用神经网络来建模向量场 $v_t(x; \theta)$，其中 $\theta \in \mathbb{R}^p$ 是可学习的参数，其实就是流 $\phi_t$ 的深度参数模型，称为 Continuous Normalizing Flow（CNF）。CNF 用于将简单的先验概率密度 $p_0$（例如纯噪声）reshape 为更复杂的概率密度 $p_1$，通过 push-forward equation 实现：
$$p_t=[\phi_t]_*p_0$$
其中 push-forward（即 change of variables）算子 $\ast$ 定义为：
$$[\phi_t]_*p_0(x)=p_0(\phi_t^{-1}(x))\det\left[\frac{\partial\phi_t^{-1}}{\partial x}(x)\right]$$
如果向量场 $v_t$ 可以生成概率密度路径 $p_t$，则其流 $\phi_t$ 满足上式。一般通过连续性方程（continuity equation）来检验向量场是否生成概率路径。

## Flow Matching

令 $x_1$ 为随机变量，其分布 $q(x_1)$ 未知（假设只知道对应分布 $q(x_1)$ 的数据样本，但不知道密度函数）。设 $p_t$ 是一个概率路径，其中 $p_0 = p$ 是一个简单的分布（如标准正态分布 $p(x) = N(x|0, I)$），$p_1$ 与 $q$ 的分布近似相等。Flow Matching 的目标函数是，匹配此目标概率路径，从而可以从 $p_0$ 流向 $p_1$。

给定目标概率密度路径 $p_t(x)$ 和其对应的向量场 $u_t(x)$，生成 $p_t(x)$，定义 Flow Matching（FM）目标函数为：
$$\mathcal{L}_\mathrm{FM}(\theta)=\mathbb{E}_{t,p_t(x)}\|v_t(x)-u_t(x)\|^2$$
其中 $\theta$ 表示 CNF 向量场 $v_t$ 的可学习的参数，$t \sim U[0, 1]$（均匀分布），$x \sim p_t(x)$。简单来说，FM loss 用神经网络 $v_t$ 回归（regresses）向量场 $u_t$。当 loss 为 0 时，学习到的 CNF 模型将生成 $p_t(x)$。

Flow Matching 无法单独使用，因为不知道合适的 $p_t$ 和 $u_t$ 是什么。有很多选择的概率路径可以满足 $p_1(x) \approx q(x)$，且通常无法获得生成所需 $p_t$ 的 $u_t$ 的 closed form。但是，可以使用仅针对 per sample 定义的概率路径和向量场来构造 $p_t$ 和 $u_t$，且适当的聚合方法可以得到 $p_t$ 和 $u_t$。

### 从条件概率路径和向量场构造 $p_t$ 和 $u_t$

构造目标概率路径的一种简单方法是通过混合更简单的概率路径：给定特定的数据样本 $x_1$，记 $p_t(x|x_1)$ 为条件概率路径，满足 $t=0$ 时 $p_0(x|x_1) = p(x)$，$t=1$ 时 $p_1(x|x_1)$ 是一个集中在 $x=x_1$ 的分布，例如 $p_1(x|x_1) = N(x|x_1, \sigma^2 I)$，均值为 $x_1$，标准差为 $\sigma > 0$ 的正态分布。将条件概率路径在 $q(x_1)$ 上边缘化，得到边缘概率路径（marginal probability path）：
$$\begin{aligned}p_t(x)=\int p_t(x|x_1)q(x_1)dx_1,\end{aligned}$$
当 $t=1$ 时，边缘概率 $p_1$ 为混合分布，近似于数据分布 $q$：
$$\begin{aligned}p_1(x)=\int p_1(x|x_1)q(x_1)dx_1\approx q(x).\end{aligned}$$
也可以定义 marginal vector field 如下：
$$\begin{aligned}u_t(x)&=\int u_t(x|x_1)\frac{p_t(x|x_1)q(x_1)}{p_t(x)}dx_1,\end{aligned}$$
其中 $u_t(\cdot|x_1): \mathbb{R}^d \rightarrow \mathbb{R}^d$ 是一个条件向量场（conditional vector field），生成 $p_t(\cdot|x_1)$。这种聚合条件向量场的方式可以得到正确的向量场，用于建模边缘概率路径。

可以观察到：**边缘向量场生成边缘概率路径。**

这提供了条件向量场（生成条件概率路径）和边缘向量场（生成边缘概率路径）之间的联系。从而可以将未知的边缘向量场分解为更简单的条件向量场，这些条件向量场仅依赖于单个数据样本，从而更简单定义。即：

定理 1： 给定生成条件概率路径 $p_t(x|x_1)$ 的向量场 $u_t(x|x_1)$，对于任何分布 $q(x_1)$，边缘向量场 $u_t$ 可以生成边缘概率路径 $p_t$，即 $u_t$ 和 $p_t$ 满足连续性方程。

> 检验向量场 $v_t$ 是否生成概率路径 $p_t$ 的一种方法是连续性方程（continuity equation），其是一个偏微分方程（PDE），提供了确保向量场 $v_t$ 生成 $p_t$ 的充要条件：
> $$\frac d{dt}p_t(x)+\mathrm{div}(p_t(x)v_t(x))=0$$

### Conditional Flow Matching

由于边缘概率路径和向量场的定义中存在不可计算的积分，无法计算 $u_t$ ，从而原始 Flow Matching 目标函数的无偏估计量也是不可计算的。因此提出一个更简单的目标函数，且其最优解与原目标函数相同。具体来说，考虑 Conditional Flow Matching（CFM）目标函数：
$$\mathcal{L}_{\mathrm{cFM}}(\theta)=\mathbb{E}_{t,q(x_1),p_t(x|x_1)}\begin{Vmatrix}v_t(x)-u_t(x|x_1)\end{Vmatrix}^2$$
其中 $t \sim U[0, 1]$，$x_1 \sim q(x_1)$，$x \sim p_t(x|x_1)$。与 FM 目标函数不同，CFM 目标函数可以实现无偏估计量的采样，只要能够有效地从 $p_t(x|x_1)$ 采样且可以计算 $u_t(x|x_1)$。而当它们是基于 per-sample 定义的，这两个条件都可以满足。因此：**FM 目标函数和 CFM 目标函数对 $\theta$ 的梯度相同。**

即：优化 CFM 目标函数等价于优化 FM 目标函数。因此，可以训练 CNF 生成边缘概率路径 $p_t$。只需要设计合适的条件概率路径和向量场。

定理 2：假设对于所有 $x \in \mathbb{R}^d$ 和 $t \in [0, 1]$，$p_t(x) > 0$，直到与 $\theta$ 无关的常数，$\mathcal{L}_\mathrm{CFM}$ 和 $\mathcal{L}_\mathrm{FM}$ 相等。因此，$\nabla_\theta \mathcal{L}_\mathrm{FM}(\theta) = \nabla_\theta \mathcal{L}_\mathrm{CFM}(\theta)$。

## 条件概率路径和向量场

Conditional Flow Matching 目标函数适用于任何条件概率路径和条件向量场。这里讨论一般高斯条件概率路径的构造。即，考虑形式为：
$$\begin{aligned}p_t(x|x_1)=\mathcal{N}(x\mid\mu_t(x_1),\sigma_t(x_1)^2I)\end{aligned}$$
其中 $\mu: [0, 1] \times \mathbb{R}^d \rightarrow \mathbb{R}^d$ 是高斯分布的时间相关均值，$\sigma: [0, 1] \times \mathbb{R} \rightarrow \mathbb{R}_{>0}$ 是时间相关标准差。设 $\mu_0(x_1) = 0$，$\sigma_0(x_1) = 1$，使得所有条件概率路径在 $t=0$ 时收敛到相同的标准高斯噪声分布 $p(x) = N(x|0, I)$。设 $\mu_1(x_1) = x_1$，$\sigma_1(x_1) = \sigma_\mathrm{min}$，其中 $\sigma_\mathrm{min}$ 足够小，使得 $p_1(x|x_1)$ 是以 $x_1$ 为中心的集中高斯分布。

有无限多个向量场可以生成任何特定的概率路径，但是大多数是由于存在 使基础分布不变的 components 而导致的，例如，当分布是旋转不变时的旋转分量会导致额外的计算。这里使用最简单的向量场。具体来说，考虑流（以 $x_1$ 为 condition）：
$$\psi_t(x)=\sigma_t(x_1)x+\mu_t(x_1)$$
当 $x$ 服从标准高斯分布时，$\psi_t(x)$ 是仿射变换，将其映射到均值为 $\mu_t(x_1)$，标准差为 $\sigma_t(x_1)$ 的正态分布随机变量。根据 change of variables，$\psi_t$ 将噪声分布 $p_0(x|x_1) = p(x)$ 推向 $p_t(x|x_1)$，即：
$$[\psi_t]_*p(x)=p_t(x|x_1)$$
这个流提供了生成条件概率路径的向量场：
$$\begin{aligned}\frac d{dt}\psi_t(x)=u_t(\psi_t(x)|x_1)\end{aligned}$$
将 $p_t(x|x_1)$ 重新参数化为 $x_0$，此时 CFM loss 为：
$$\mathcal{L}_\text{сғм}{ ( \theta ) }=\mathbb{E}_{t,q(x_1),p(x_0)}\begin{Vmatrix}v_t(\psi_t(x_0))-\frac d{dt}\psi_t(x_0)\end{Vmatrix}^2$$

由于 $\psi_t$ 是一个简单的（可逆的）仿射映射，可以闭式求解 $u_t$。对于时间相关函数 $f$，设 $f'$ 表示关于时间的导数，即 $f' = \frac {df}{dt}$。

定理 3：设 $p_t(x|x_1)$ 是一个高斯概率路，$\psi_t$ 是其对应的流映射。则定义 $\psi_t$ 的唯一向量场形式为：
$$u_t(x|x_1)=\frac{\sigma_t^{\prime}(x_1)}{\sigma_t(x_1)}\left(x-\mu_t(x_1)\right)+\mu_t^{\prime}(x_1)$$

因此，$u_t(x|x_1)$ 生成高斯路径 $p_t(x|x_1)$。

### 高斯条件概率路径的一些特殊例子

上面的公式对任意函数 $\mu_t(x_1)$ 和 $\sigma_t(x_1)$ 都是通用的，可以将它们设置为任何满足所需边界条件的可微函数。

#### Diffusion conditional VFs

Diffusion 从数据点开始，逐渐添加噪声，直到近似纯噪声。可以将其表示为随机过程，为了在任意时间 $t$ 获得闭式表示，需要满足严格的要求，从而得到具有特定均值 $\mu_t(x_1)$ 和标准差 $\sigma_t(x_1)$ 的高斯条件概率路径 $p_t(x|x_1)$。例如，反向（noise→data）Variance Exploding（VE）路径的形式为：
$$\begin{aligned}p_t(x)=\mathcal{N}(x|x_1,\sigma_{1-t}^2I)\end{aligned}$$
其中 $\sigma_t$ 是一个增函数，$\sigma_0 = 0$，$\sigma_1 \gg 1$。将 $\mu_t(x_1) = x_1$ 和 $\sigma_t(x_1) = \sigma_{1-t}$ 代入定理 3 得到：
$$u_t(x|x_1)=-\frac{\sigma_{1-t}^{\prime}}{\sigma_{1-t}}(x-x_1)$$
此时反向（noise→data）Variance Preserving（VP）diffusion 路径的形式为：
$$p_t(x|x_1)=\mathcal{N}(x\mid\alpha_{1-t}x_1,\left(1-\alpha_{1-t}^2\right)I),\mathrm{where~}\alpha_t=e^{-\frac12T(t)},T(t)=\int_0^t\beta(s)ds
$$
其中 $\beta$ 是噪声比例函数。将 $\mu_t(x_1) = \alpha_{1-t}x_1$ 和 $\sigma_t(x_1) = \sqrt{1-\alpha_{1-t}^2}$ 代入定理 3 得到：
$$u_t(x|x_1)=\frac{\alpha_{1-t}^{\prime}}{1-\alpha_{1-t}^2}\left(\alpha_{1-t}x-x_1\right)=-\frac{T^{\prime}(1-t)}2\left[\frac{e^{-T(1-t)}x-e^{-\frac12T(1-t)}x_1}{1-e^{-T(1-t)}}\right]$$
构造的条件向量场 $u_t(x|x_1)$ 与 score-based generative modeling 中的确定性概率流在这些条件扩散过程中是一致的。

#### Optimal Transport conditional VFs

条件概率路径的另一个选择是，均值和标准差随时间线性变化：
$$\mu_t(x)=tx_1,\mathrm{~and~}\sigma_t(x)=1-(1-\sigma_{\min})t$$
根据定理 3，该路径由向量场生成：
$$u_t(x|x_1)=\frac{x_1-(1-\sigma_{\min})x}{1-(1-\sigma_{\min})t}$$
与扩散条件向量场不同，该向量场对所有 $t \in [0, 1]$ 都是确定的。对应于 $u_t(x|x_1)$ 的条件流为：
$$\psi_t(x)=(1-(1-\sigma_{\min})t)x+tx_1$$
此时 CFM loss 为：
$$\mathcal{L}_\text{CFM}{ ( \theta ) }=\mathbb{E}_{t,q(x_1),p(x_0)}\begin{Vmatrix}v_t(\psi_t(x_0))-\left(x_1-(1-\sigma_{\min})x_0\right)\end{Vmatrix}^2$$

均值和标准差线性变化不仅使得路径简单直观，而且在以下意义上也是最优的。即，条件流 $\psi_t(x)$ 实际上是两个高斯分布 $p_0(x|x_1)$ 和 $p_1(x|x_1)$ 之间的最优传输（Optimal Transport，OT）位移映射。OT 插值（interpolant），即概率路径，定义为：
$$p_t=[(1-t)\text{id}+t\psi]_\star p_0$$
其中 $\psi: \mathbb{R}^d \rightarrow \mathbb{R}^d$ 是将 $p_0$ 推向 $p_1$ 的 OT 映射，id 表示恒等映射，即 $\text{id}(x) = x$，$(1-t)\mathrm{id}+t\psi$ 称为 OT 位移映射。在两个都是高斯分布的情况下（其中第一个是标准高斯分布），OT 位移映射的形式为：
$$\psi_t(x)=(1-(1-\sigma_{\min})t)x+tx_1$$

直观上，OT 位移映射下的粒子总是沿直线轨迹以恒定速度运动。下图为扩散和 OT 条件向量场的采样路径，扩散路径的采样轨迹可能会“超过”最终样本，导致不必要的回溯，而 OT 路径保证保持直线：
![](image/Pasted%20image%2020240124202945.png)

下图比较了扩散条件 score function $\nabla \log p_t(x|x_1)$（$p_t$ 如之前公式中定义）与 OT 条件向量场。两个例子中的起始（$p_0$）和结束（$p_1$）高斯分布相同。可以发现，OT 向量场在时间上具有恒定的方向，从而使得回归任务更简单。公式上来看，因为向量场可以写成 $u_t(x|x_1) = g(t)h(x|x_1)$ 的形式。
> 注意，尽管条件流是最优的，但不意味着边缘向量场是最优的传输解。但是还是希望边缘向量场保持相对简单。

![](image/Pasted%20image%2020240124203027.png)

## 相关工作（略）

## 实验（略）
