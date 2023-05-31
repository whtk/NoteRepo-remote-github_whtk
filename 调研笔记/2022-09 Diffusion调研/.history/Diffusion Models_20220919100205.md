<!--
 * @Author: error: git config user.name && git config user.email & please set dead value or install git
 * @Date: 2022-09-16 10:31:22
 * @LastEditors: 郭印林
 * @LastEditTime: 2022-09-19 10:00:42
 * @FilePath: \2022-09 Diffusion调研\Diffusion Models.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->

# Diffusion 模型的方法和应用调研

> 绝大部分来自于论文 [Diffusion Models: A Comprehensive Survey of Methods and Applications](https://paperswithcode.com/paper/diffusion-models-a-comprehensive-survey-of)

1. Diffusion 可以看成是 VAE 的级联，对应于VAE的编码和解码；Diffusion 也可以看成是 随机微分方程的离散化。
2. 可以将 Diffusion 模型分成三类：
    + sampling-acceleration enhancement
    + likelihood-maximization enhancement
    + generalization-ability enhancement 
3. 其他五种生成模型：
    + VAE
    + GAN
    + normalizing flow
    + autoregressive
    + energy-based models
通过组合这些模型和 Diffusion 模型，可以得到更强大的模型
4. Diffusion 模型应用：
    + cv
    + nlp
    + waveform signal processing
    + multi-modal modeling
    + molecular graph generation（分子图生成）
    + time series modeling（时间序列建模）
    + dversarial purification（对抗降噪）

![](overview.png)

## 预备知识
生成模型一个中心问题就是，模型概率分布的灵活度和复杂度之间的权衡。


Diffusion 模型的基本思想是通过正向过程扰乱数据分布中的结构，然后通过学习反向扩散过程来恢复结构，从而产生高度灵活和易于处理的生成模型。

一共有两种 Diffusion 模型，Denoising Diffusion Probabilistic 模型 和 Score-based Generative 模型。
![](2022-09-16-11-24-15.png)

### Denoising Diffusion Probabilistic 模型（DDPM）
1. 包含两个马尔科夫链，采用变分推理来生成指定分布的样本
2. forward：从已有的数据分布开始，逐步增加噪声直到分布收敛到某个先验分布（如高斯分布）
3. reverse：从某个先验分布开始，采用带参数的高斯转化核，逐渐恢复出原始数据。
4. 数学角度看，给定目标数据分布 $\mathbf{x}_0 \sim q\left(\mathbf{x}_0\right)$，**forward** 过程为：
    $$\begin{aligned} q\left(\mathbf{x}_1, \ldots, \mathbf{x}_T \mid \mathbf{x}_0\right) &=\prod_{t=1}^T q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right) \\ q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right) &=\mathcal{N}\left(\mathbf{x}_t ; \sqrt{1-\beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I}\right) \end{aligned}$$
其中，$\beta_{t} \in (0,1)$ 为方差系数。当 $\alpha_t=1-\beta_t$ 且 $\bar{\alpha}_t=\prod_{s=0}^t \alpha_s$ 时，有：
    $$\begin{aligned} q\left(\mathbf{x}_t \mid \mathbf{x}_0\right) &=\mathcal{N}\left(\mathbf{x}_t ; \sqrt{\bar{\alpha}_t} \mathbf{x}_0,\left(1-\bar{\alpha}_t\right) \mathbf{I}\right) \\ \mathbf{x}_t &=\sqrt{\bar{\alpha}_t} \mathbf{x}_0+\sqrt{1-\bar{\alpha}_t} \epsilon \end{aligned}$$
最终当 $\bar{\alpha}_t$ 接近 $0$ 时，$p(\mathbf{x}_T)$ 和高斯分布无二。
**reverse** 过程为，从 $p\left(\mathbf{x}_T\right)=\mathcal{N}\left(\mathbf{x}_T ; \mathbf{0}, \mathbf{I}\right)$ 开始，参数化反向过程如下：
    $$\begin{aligned} p_\theta\left(\mathbf{x}_{0: T}\right) &=p\left(\mathbf{x}_T\right) \prod_{t=1}^T p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right) \\ p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right) &=\mathcal{N}\left(\mathbf{x}_{t-1} ; \mu_\theta\left(\mathbf{x}_t, t\right), \Sigma_\theta\left(\mathbf{x}_t, t\right)\right) \end{aligned}$$
训练过程可以通过优化负的变分下界来实现，通过一系列的推到，最终的优化目标函数可以写为：
    $$\mathbb{E}_{t \sim \mathcal{U}(0, T), \mathbf{x}_0 \sim q\left(\mathbf{x}_0\right), \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathrm{I})}\left[\lambda(t)\left\|\boldsymbol{\epsilon}-\boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t, t\right)\right\|^2\right]$$
简单来说，就是优化预测噪声和真实噪声之间的L2距离。

### Score-based Generative 模型 
另一方面，可以把 Diffusion 模型看成 score-based generative model 的离散化。这种模型通过构造 stochastic differential equation（随机差分方程，SDE）将数据打乱到一个已知的先验分布。同时一个对应的反向SDE可以反转这个过程。**forward** 过程为以下SDE的解：
    $$d\mathbf{x} = \mathbf{f}(x,t)dt + g(t)d\mathbf{w}$$
其中，$\mathbf{x}_T$ 为打乱后的分布。那么 DDPM 的 forward 过程能够看成SDE前向过程的离散化：
    $$d\mathbf{x} = -\frac 12 \beta(t)\mathbf{x}dt + \sqrt{\beta{(t)}}d\mathbf{w}$$
最后，为了从已知先验生成数据，**reverse** 过程为：
$$d \mathbf{x}=\mathbf{f}(\mathbf{x}, t)-g(t)^2 \nabla_{\mathbf{x}} \log q_t(\mathbf{x}) d t+g(t) d \overline{\mathbf{w}}$$

经典的 Diffusion 模型有三个主要的缺点：
+ 采样过程效率低
+ 似然估计次优
+ 数据泛化能力差
 
增强提高后的 Diffusion 模型可以分为三类：
1. 采样加速
2. 似然最大化
3. 数据泛化


### 采样加速增强（SAMPLING-ACCELERATION ENHANCEMENT）的 Diffusion 模型
现有的方法采样耗时高，以下是一些加速采样方法。

#### Discretization Optimization 离散优化
通过减少采样步是一个可行的方法，但是会影响模型的性能；可以通过优化离散化，如通过求解随机微分方程或常微分方程来减少采样步骤。
1. SGM：以forward SDE相同的方式离散化反向时间SDE。
2. ODE：不需要进行高斯采样，从而提高采样效率

#### Non-Markovian Process 非马尔可夫过程
马尔可夫仅依赖于最近的一个step进行预测，这限制了之前信息的利用。

DDIM 拓展了原始的DDPM来实现非马尔可夫过程。其forward 过程为：
$$q\left(\mathbf{x}_1, \ldots, \mathbf{x}_T \mid \mathbf{x}_0\right)=\prod_{t=1}^T q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}, \mathbf{x}_0\right)$$
$$q_\sigma\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right)=\mathcal{N}\left(\mathbf{x}_{t-1} \mid \tilde{\mu}_t\left(\mathbf{x}_t, \mathbf{x}_0\right), \sigma_t^2 \mathbf{I}\right)$$
$$\tilde{\mu}_t\left(\mathbf{x}_t, \mathbf{x}_0\right)=\sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0+\sqrt{\bar{\beta}_{n-1}-\sigma_t^2} \cdot \frac{\mathbf{x}_n-\sqrt{\alpha_t} \mathbf{x}_0}{\sqrt{\bar{\beta}_t}}$$

DDPM可以看成是DDIM的一个特殊形式。

#### Partial Sampling 部分采样
通过仅使用reverse过程中的部分step来生成样本，以质量换取采样速度。


### 最大似然增强（LIKELIHOOD-MAXIMIZATION ENHANCEMENT）的 Diffusion 模型
DDPM 相比于其他似然模型没有什么优势，通过设计和分析变分下界来增强最大似然，主要有以下三种方法。

#### Noise Schedule Optimization 噪声调度优化
通过优化噪声引入来实现增强的对数似然。如采取以下调度器：
    $$\bar{\alpha}_t=\frac{h(t)}{h(0)}, h(t)=\cos \left(\frac{t / T+m}{1+m} \cdot \frac{\pi}{2}\right)^2$$
同时使用 $\beta_t=1-\frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}$ 来计算 forward 方差。由于 $\bar{\alpha}_t$ 平滑改变，reverse 过程能够更好的恢复原始的数据结构。

VDM [表明数据 $\mathbf{x}$ 的变分下界可以简化为 Diffusion 数据的信噪比的简短表达，同时表明，在无限深的设置中，VLB 完全由端点的噪声调度决定，而与端点之间的噪声调度无关，因此可以通过优化端点噪声调度以减少方差。

#### Learnable Reverse Variance 可学习的反向方差

通过优化 reverse 过程的方差的估计可以减少拟合误差，从而最大化VLB和对数似然。通过设置 reverse 方差为：
$$\Sigma_\theta\left(\mathbf{x}_t, t\right)=\exp \left(v \cdot \log \beta_t+(1-v) \cdot \log \tilde{\beta}_t\right)$$
可以减少采样的时间步。

 Analytic-DPM 