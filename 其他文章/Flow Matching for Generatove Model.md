> preprint，Meta AI

1. 提出一种建立在 Continuous Normalizing Flows (CNFs) 上的新的生成模型范式，即 Flow Matching
2. Flow Matching 兼容通用的高斯概率路径，用于在噪声和数据样本之间进行转换
3. 同时 为使用其他非扩散概率路径训练 CNF 打开了大门

## Introduction
<!-- Deep generative models are a class of deep learning algorithms aimed at estimating and sampling from an unknown data distribution. The recent influx of amazing advances in generative modeling, e.g., for image generation Ramesh et al. (2022); Rombach et al. (2022), is mostly facilitated by the scalable and relatively stable training of diffusion-based models Ho et al. (2020); Song et al. (2020b). However, the restriction to simple diffusion processes leads to a rather confined space of sampling probability paths, resulting in very long training times and the need to adopt specialized methods (e.g., Song et al. (2020a); Zhang & Chen (2022)) for efficient sampling -->
1. 深度生成模型是旨在估计和采样未知数据分布，diffusion-based models 是一种可扩展且相对稳定的训练方法，但是限制了采样概率路径的空间，导致训练时间很长，需要采用专门的方法来进行采样
<!-- In this work we consider the general and deterministic framework of Continuous Normalizing Flows (CNFs; Chen et al. (2018)). CNFs are capable of modeling arbitrary probability path and are in particular known to encompass the prob-
ability paths modeled by diffusion processes (Song
et al., 2021). However, aside from diffusion that can be trained efficiently via, e.g., denoising score matching (Vincent, 2011), no scalable CNF train- ing algorithms are known. Indeed, maximum like- lihood training (e.g., Grathwohl et al. (2018)) re- quire expensive numerical ODE simulations, while existing simulation-free methods either involve in- tractable integrals (Rozen et al., 2021) or biased gra- dients (Ben-Hamu et al., 2022). 
The goal of this work is to propose Flow Matching (FM), an efficient simulation-free approach to train- ing CNF models, allowing the adoption of general probability paths to supervise CNF training. Impor- tantly, FM breaks the barriers for scalable CNF train- ing beyond diffusion, and sidesteps the need to rea- son about diffusion processes to directly work with probability paths.-->
2. 本文考虑 Continuous Normalizing Flows 的一般性和确定性框架。提出 Flow Matching，是一种训练 CNF 模型 simulation-free 的方法，采用 通用的概率路径进行训练。且 FM 避免了对 diffusion 过程进行推理，而是直接使用概率路径
<!-- In particular, we propose the Flow Matching objective (Section 3), a simple and intuitive training objective to regress onto a target vector field that generates a desired probability path. We first show that we can construct such target vector fields through per-example (i.e., conditional) formu- lations. Then, inspired by denoising score matching, we show that a per-example training objective, termed Conditional Flow Matching (CFM), provides equivalent gradients and does not require ex- plicit knowledge of the intractable target vector field. Furthermore, we discuss a general family of per-example probability paths (Section 4) that can be used for Flow Matching, which subsumes ex- isting diffusion paths as special instances. Even on diffusion paths, we find that using FM provides more robust and stable training, and achieves superior performance compared to score matching. Furthermore, this family of probability paths also includes a particularly interesting case: the vector field that corresponds to an Optimal Transport (OT) displacement interpolant (McCann, 1997). We find that conditional OT paths are simpler than diffusion paths, forming straight line trajectories whereas diffusion paths result in curved paths. These properties seem to empirically translate to faster training, faster generation, and better performance. -->
3. 提出 Flow Matching objective，一种简单直观的训练目标函数，用于回归到 生成所需的概率路径 的 目标向量场（target vector field）
<!-- We empirically validate Flow Matching and the construction via Optimal Transport paths on Im- ageNet, a large and highly diverse image dataset. We find that we can easily train models to achieve favorable performance in both likelihood estimation and sample quality amongst competing diffusion-based methods. Furthermore, we find that our models produce better trade-offs between computational cost and sample quality compared to prior methods. Figure 1 depicts selected uncon- ditional ImageNet 128×128 samples from our model. -->
4. 在 ImageNet 上验证了 Flow Matching 和 Optimal Transport paths 的构造，发现可以轻松训练模型，在似然估计和样本质量之间取得较好的性能。且可以在计算成本和样本质量之间实现 trade-off

## 预备知识：Continuous Normalizing Flows

<!-- Let Rd denote the data space with data points x = (x1, . . . , xd) ∈ Rd. Two important objects we use in this paper are: the probability density path p : [0,1] × Rd → R>0, which is a time dependent1 probability density function, i.e., R pt(x)dx = 1, and a time-dependent vector field, v : [0, 1] × Rd → Rd . A vector field vt can be used to construct a time-dependent diffeomorphic map, called a flow, φ : [0, 1] × Rd → Rd , defined via the ordinary differential equation (ODE): -->
定义 $\mathbb{R}^d$ 为数据空间，$x = (x_1, \dots, x_d) \in \mathbb{R}^d$ 为数据点。概率密度路径 $p: [0, 1] \times \mathbb{R}^d \rightarrow \mathbb{R}_{>0}$ 是一个时间相关的概率密度函数，即 $\int_{\mathbb{R}^d} p_t(x)dx = 1$。时间相关的向量场为 $v: [0, 1] \times \mathbb{R}^d \rightarrow \mathbb{R}^d$。向量场 $v_t$ 可以用来构造时间相关的微分同胚映射（diffeomorphic map），称为流（flow），$\phi: [0, 1] \times \mathbb{R}^d \rightarrow \mathbb{R}^d$，定义为常微分方程（ODE）：
$$\begin{aligned}\frac d{dt}\phi_t(x)&=v_t(\phi_t(x))\\\phi_0(x)&=x\end{aligned}$$
<!-- Previously, Chen et al. (2018) suggested modeling the vector field vt with a neural network, vt(x; θ), where θ ∈ Rp are its learnable parameters, which in turn leads to a deep parametric model of the flow φt, called a Continuous Normalizing Flow (CNF). A CNF is used to reshape a simple prior density p0 (e.g., pure noise) to a more complicated one, p1, via the push-forward equation -->
[Neural Ordinary Differential Equations 笔记](Neural%20Ordinary%20Differential%20Equations%20笔记.md) 已经表明，可以用神经网络来建模向量场 $v_t(x; \theta)$，其中 $\theta \in \mathbb{R}^p$ 是可学习的参数，其实就是流 $\phi_t$ 的深度参数模型，称为 Continuous Normalizing Flow（CNF）。CNF 用于将简单的先验概率密度 $p_0$（例如纯噪声）reshape 为更复杂的概率密度 $p_1$，通过 push-forward equation 实现：
$$p_t=[\phi_t]_*p_0$$
<!-- where the push-forward (or change of variables) operator ∗ is defined by -->
其中 push-forward（即 change of variables）算子 $\ast$ 定义为：
$$[\phi_t]_*p_0(x)=p_0(\phi_t^{-1}(x))\det\left[\frac{\partial\phi_t^{-1}}{\partial x}(x)\right]$$
<!-- a vector field vt is said to generate a probability density path pt if its flow φt satisfies equation 3. One practical way to test if a vector field generates a probability path is using the continuity equation, which is a key component in our proofs, see Appendix B. We recap more information on CNFs, in particular how to compute the probability p1(x) at an arbitrary point x ∈ Rd in Appendix C. -->
如果向量场 $v_t$ 可以生成概率密度路径 $p_t$，则其流 $\phi_t$ 满足上式。一般通过连续性方程（continuity equation）来检验向量场是否生成概率路径。

## Flow Matching

<!-- Let x1 denote a random variable distributed according to some unknown data distribution q(x1). We assume we only have access to data samples from q(x1) but have no access to the density function itself. Furthermore, we let pt be a probability path such that p0 = p is a simple distribution, e.g., the standard normal distribution p(x) = N (x|0, I ), and let p1 be approximately equal in distribution to q. We will later discuss how to construct such a path. The Flow Matching objective is then designed to match this target probability path, which will allow us to flow from p0 to p1. -->
令 $x_1$ 为随机变量，其分布 $q(x_1)$ 未知（假设只知道对应分布 $q(x_1)$ 的数据样本，但不知道密度函数）。设 $p_t$ 是一个概率路径，其中 $p_0 = p$ 是一个简单的分布（如标准正态分布 $p(x) = N(x|0, I)$），$p_1$ 与 $q$ 的分布近似相等。Flow Matching 的目标函数是，匹配此目标概率路径，从而可以从 $p_0$ 流向 $p_1$。
<!-- Given a target probability density path pt(x) and a corresponding vector field ut(x), which generates pt(x), we define the Flow Matching (FM) objective as -->
给定目标概率密度路径 $p_t(x)$ 和其对应的向量场 $u_t(x)$，生成 $p_t(x)$，定义 Flow Matching（FM）目标函数为：
$$\mathcal{L}_\mathrm{FM}(\theta)=\mathbb{E}_{t,p_t(x)}\|v_t(x)-u_t(x)\|^2$$
<!-- where θ denotes the learnable parameters of the CNF vector field vt (as defined in Section 2), t ∼ U [0, 1] (uniform distribution), and x ∼ pt (x). Simply put, the FM loss regresses the vector field ut with a neural network vt. Upon reaching zero loss, the learned CNF model will generate pt(x). -->
其中 $\theta$ 表示 CNF 向量场 $v_t$ 的可学习的参数，$t \sim U[0, 1]$（均匀分布），$x \sim p_t(x)$。简单来说，FM loss 用神经网络 $v_t$ 回归（regresses）向量场 $u_t$。当 loss 为 0 时，学习到的 CNF 模型将生成 $p_t(x)$。
<!-- Flow Matching is a simple and attractive objective, but na ̈ıvely on its own, it is intractable to use in practice since we have no prior knowledge for what an appropriate pt and ut are. There are many choices of probability paths that can satisfy p1(x) ≈ q(x), and more importantly, we generally don’t have access to a closed form ut that generates the desired pt. In this section, we show that we can construct both pt and ut using probability paths and vector fields that are only defined per sample, and an appropriate method of aggregation provides the desired pt and ut. Furthermore, this construction allows us to create a much more tractable objective for Flow Matching. -->
Flow Matching 无法单独使用，因为不知道合适的 $p_t$ 和 $u_t$ 是什么。有很多选择的概率路径可以满足 $p_1(x) \approx q(x)$，且通常无法获得生成所需 $p_t$ 的 $u_t$ 的 closed form。但是，可以使用仅针对 per sample 定义的概率路径和向量场来构造 $p_t$ 和 $u_t$，且适当的聚合方法可以得到 $p_t$ 和 $u_t$。
<!-- CONSTRUCTING pt,ut FROM CONDITIONAL PROBABILITY PATHS AND VECTOR FIELDS -->
### 从条件概率路径和向量场构造 $p_t$ 和 $u_t$

<!-- A simple way to construct a target probability path is via a mixture of simpler probability paths: Given a particular data sample x1 we denote by pt(x|x1) a conditional probability path such that it satisfies p0(x|x1) = p(x) at time t = 0, and we design p1(x|x1) at t = 1 to be a distribution concentrated around x = x1, e.g., p1(x|x1) = N (x|x1, σ2I), a normal distribution with x1 mean and a sufficiently small standard deviation σ > 0. Marginalizing the conditional probability paths over q(x1) give rise to the marginal probability path -->
构造目标概率路径的一种简单方法是通过混合更简单的概率路径：给定特定的数据样本 $x_1$，记 $p_t(x|x_1)$ 为条件概率路径，满足 $t=0$ 时 $p_0(x|x_1) = p(x)$，$t=1$ 时 $p_1(x|x_1)$ 是一个集中在 $x=x_1$ 的分布，例如 $p_1(x|x_1) = N(x|x_1, \sigma^2 I)$，均值为 $x_1$，标准差为 $\sigma > 0$ 的正态分布。将条件概率路径在 $q(x_1)$ 上边缘化，得到边缘概率路径（marginal probability path）：
$$\begin{aligned}p_t(x)=\int p_t(x|x_1)q(x_1)dx_1,\end{aligned}$$
<!-- where in particular at time t = 1, the marginal probability p1 is a mixture distribution that closely approximates the data distribution q,
 -->
当 $t=1$ 时，边缘概率 $p_1$ 为混合分布，近似于数据分布 $q$：
$$\begin{aligned}p_1(x)=\int p_1(x|x_1)q(x_1)dx_1\approx q(x).\end{aligned}$$

也可以定义 marginal vector field 如下：
$$\begin{aligned}u_t(x)&=\int u_t(x|x_1)\frac{p_t(x|x_1)q(x_1)}{p_t(x)}dx_1,\end{aligned}$$
<!-- where ut(·|x1) : Rd → Rd is a conditional vector field that generates pt(·|x1). It may not seem apparent, but this way of aggregating the conditional vector fields actually results in the correct vector field for modeling the marginal probability path. -->
其中 $u_t(\cdot|x_1): \mathbb{R}^d \rightarrow \mathbb{R}^d$ 是一个条件向量场（conditional vector field），生成 $p_t(\cdot|x_1)$。这种聚合条件向量场的方式可以得到正确的向量场，用于建模边缘概率路径。
<!-- Our first key observation is this:
The marginal vector field (equation 8) generates the marginal probability path (equation 6). -->
可以观察到：**边缘向量场生成边缘概率路径。**
<!-- This provides a surprising connection between the conditional VFs (those that generate conditional probability paths) and the marginal VF (those that generate the marginal probability path). This con- nection allows us to break down the unknown and intractable marginal VF into simpler conditional VFs, which are much simpler to define as these only depend on a single data sample. We formalize this in the following theorem.
 -->
这提供了条件向量场（生成条件概率路径）和边缘向量场（生成边缘概率路径）之间的联系。从而可以将未知的边缘向量场分解为更简单的条件向量场，这些条件向量场仅依赖于单个数据样本，从而更简单定义。即：
<!-- Theorem 1. Given vector fields ut(x|x1) that generate conditional probability paths pt(x|x1), for any distribution q(x1), the marginal vector field ut in equation 8 generates the marginal probability path pt in equation 6, i.e., ut and pt satisfy the continuity equation (equation 26). -->
定理 1： 给定生成条件概率路径 $p_t(x|x_1)$ 的向量场 $u_t(x|x_1)$，对于任何分布 $q(x_1)$，边缘向量场 $u_t$ 可以生成边缘概率路径 $p_t$，即 $u_t$ 和 $p_t$ 满足连续性方程。
<!-- One method of testing if a vector field vt generates a probability path pt is the continuity equation (Villani, 2009). It is a Partial Differential Equation (PDE) providing a necessary and sufficient condition to ensuring that a vector field vt generates pt, -->
> 检验向量场 $v_t$ 是否生成概率路径 $p_t$ 的一种方法是连续性方程（continuity equation），其是一个偏微分方程（PDE），提供了确保向量场 $v_t$ 生成 $p_t$ 的充要条件：$$\frac d{dt}p_t(x)+\mathrm{div}(p_t(x)v_t(x))=0$$

### Conditional Flow Matching
<!-- Unfortunately, due to the intractable integrals in the definitions of the marginal probability path and VF (equations 6 and 8), it is still intractable to compute ut, and consequently, intractable to na ̈ıvely compute an unbiased estimator of the original Flow Matching objective. Instead, we propose a simpler objective, which surprisingly will result in the same optima as the original objective. Specifically, we consider the Conditional Flow Matching (CFM) objective, -->
由于边缘概率路径和向量场的定义中存在不可计算的积分，无法计算 $u_t$ ，从而原始 Flow Matching 目标函数的无偏估计量也是不可计算的。因此提出一个更简单的目标函数，且其最优解与原目标函数相同。具体来说，考虑 Conditional Flow Matching（CFM）目标函数：
$$\mathcal{L}_{\mathrm{cFM}}(\theta)=\mathbb{E}_{t,q(x_1),p_t(x|x_1)}\begin{Vmatrix}v_t(x)-u_t(x|x_1)\end{Vmatrix}^2$$
<!-- where t ∼ U[0, 1], x1 ∼ q(x1), and now x ∼ pt(x|x1). Unlike the FM objective, the CFM objective allows us to easily sample unbiased estimates as long as we can efficiently sample from pt(x|x1) and compute ut(x|x1), both of which can be easily done as they are defined on a per-sample basis. Our second key observation is therefore: -->
其中 $t \sim U[0, 1]$，$x_1 \sim q(x_1)$，$x \sim p_t(x|x_1)$。与 FM 目标函数不同，CFM 目标函数可以实现无偏估计量的采样，只要能够有效地从 $p_t(x|x_1)$ 采样且可以计算 $u_t(x|x_1)$。而当它们是基于 per-sample 定义的，这两个条件都可以满足。因此：**FM 目标函数和 CFM 目标函数对 $\theta$ 的梯度相同。**
<!-- The FM (equation 5) and CFM (equation 9) objectives have identical gradients w.r.t. θ. -->

<!-- That is, optimizing the CFM objective is equivalent (in expectation) to optimizing the FM objective. Consequently, this allows us to train a CNF to generate the marginal probability path pt—which in particular, approximates the unknown data distribution q at t=1— without ever needing access to either the marginal probability path or the marginal vector field. We simply need to design suitable conditional probability paths and vector fields. We formalize this property in the following theorem. -->
即：优化 CFM 目标函数等价于优化 FM 目标函数。因此，可以训练 CNF 生成边缘概率路径 $p_t$。只需要设计合适的条件概率路径和向量场。

<!-- Theorem 2. Assuming that pt(x) > 0 for all x ∈ Rd and t ∈ [0,1], then, up to a constant independent of θ, LCFM and LFM are equal. Hence, ∇θLFM(θ) = ∇θLCFM(θ). -->
定理 2：假设对于所有 $x \in \mathbb{R}^d$ 和 $t \in [0, 1]$，$p_t(x) > 0$，则 LCFM 和 LFM 相等，即 $\nabla_\theta LFM(\theta) = \nabla_\theta LCFM(\theta)$。