> ICML 2021，OpenAI

1. 表明只需要一点点的修改，DDPM 可以实现 competitive 的对数似然同时提高采样质量
2. 学习 reverse diffusion process 的方差，可以减少一个数量级的采样次数，同时保持采样的质量几乎不变

> 总的来说，本文的改进是，将 DDPM scale 到一个更大的数据集，然后调参效果能变好；DDPM 中的方差也变得可学习，而不是一开始就指定好的；改进了 noise schedule，使用余弦而非线性的 schedule。

## Introduction

1. DDPM 有几个未知的问题：
	1. 能否捕获一个分布的所有的 mode
	2. 能够 scale 到更多样化的数据集上
	3. DDPM 用于音频生成只需要几个 采样 step，但是图像还不知道
2. 本文表明：即使在更多样化的数据集中，DDPM 也可以实现 competitive 的对数似然计算；为了使 VLB 更 tight，使用重参数技巧学习反向过程的方差来得到一个混合的目标函数
3. 可以实现 50 次的 forward pass 得到好的样本
4. 也发现，随着模型大小和计算量的增加，性能也会增加

## DDPM（略）

## 对数似然的提升

在生成模型中，对数似然是一个很重要的指标，通过优化对数似然可以迫使生成模型捕获数据分布的所有的 mode。而且在对数似然上的小的改变会极大地影响采样的质量。

但是 DDPM 并不能得到较好的对数似然。下面给出一些 DDPM 算法的修改以实现更好的对数似然。

在 DDPM 中，参数 $\sigma_t^2=\beta_{t},T=1000$ 时，对数似然为 3.99 (bits/dim) 在 ImageNet 64 × 64 数据集中，且训练了 200K。随着 $T$ 增大到 $T=4000$，可以提升到 3.77。

### 学习 $\Sigma_\theta(x_t,t)$ 

DDPM 设置 $\Sigma_\theta(x_t,t)=\sigma_t^2\mathbf{I}$ ，其中 $\sigma_t$ 不是可学习的，且发现固定 $\sigma_t^2$ 为 $\beta_t$ 或 $\tilde{\beta}_t$ 性能差不多。但是实际上这两个值的两个极端为啥效果会差不多呢？

其实计算就可以发现，除了在 0 附近，两个值的大小其实是差不多的（如下图）。这说明，如果 diffusion 的 step 是无限的，$\sigma_t$ 的选择压根不重要。也就是说，step 越多，模型的均值 $\mu_\theta\left(x_t, t\right)$ 比 $\Sigma_\theta\left(x_t, t\right)$ 更能决定分布。
![](image/Pasted%20image%2020230817114803.png)

但是，在前几个 step 中，diffusion process 对变分下界的贡献才是最大的，因此选择一个好的 $\Sigma_\theta\left(x_t, t\right)$ 可以提高对数似然的计算。

具体来说，模型输出向量 $v$，然后将这个向量通过下式插值转换为方差：
$$\Sigma_\theta\left(x_t, t\right)=\exp \left(v \log \beta_t+(1-v) \log \tilde{\beta}_t\right)$$
这里的 $v$ 没有任何限制（不限于 0-1 之间），但实际上网络没有超出范围。从而定义一个新的混合损失为：
$$L_{\text {hybrid }}=L_{\text {simple }}+\lambda L_{\mathrm{vlb}}$$
实验中，设置 $\lambda=0.001$，防止 $L_{\mathrm{vlb}}$ 超过原来的损失。并且 $L_{\mathrm{vlb}}$ 不对均值计算梯度，从而使得这一项损失只用来更新方差项。

### 改进 noise schedule

DDPM 中使用的 Linear noise schedule 是 sub-optimal 的，因为 forward 过程的最后的部分的噪声太大了，导致其对样本质量没有贡献，如图：
![](image/Pasted%20image%2020230817195840.png)

于是基于 $\bar{\alpha}_t$ 构造了一个新的 noise schedule：
$$\bar{\alpha}_t=\frac{f(t)}{f(0)}, \quad f(t)=\cos \left(\frac{t / T+s}{1+s} \cdot \frac{\pi}{2}\right)^2$$
这个函数使得 $\bar{\alpha}_t$ 在中间的时候是线性的，而在靠近 $t=0$ 和 $t=T$ 的位置几乎不变。如图：![](image/Pasted%20image%2020230817200431.png)

同时，使用一个小的 offset $s$ 来避免 $\beta_t$ 在 $t=0$ 的位置太小，因为如果一开始的噪声很小的话网络会很难预测噪声 $\epsilon$ 。

### 减少梯度噪声

我们希望通过直接优化 $L_{\text {vlb }}$ 而不是 $L_{\text {hybrid }}$ 来获得最佳的对数似然。但是发现 $L_{\text {vlb }}$ 很难优化，尤其是在多样性数据集上。

下图显示两个损失的学习曲线。两条曲线都有噪声，但在训练时间相同的情况下，混合目标函数显然能在训练集上获得更好的对数似然：
![](image/Pasted%20image%2020230817201237.png)

作者假设，$L_{\text {vlb }}$ 的噪声比 $L_{\text {hybrid }}$ 大：
![](image/Pasted%20image%2020230817201048.png)
从而需要找到一种方法来减少 $L_{\text {vlb }}$ 的方差从而可以直接优化对数似然。作者的假设是，对 $t$ 的均匀采样导致了这种不必要的噪声，于是使用重要性采样：$$L_{\mathrm{vlb}}=E_{t \sim p_t}\left[\frac{L_t}{p_t}\right], \text { where } p_t \propto \sqrt{E\left[L_t^2\right]} \text { and } \sum p_t=1$$
由于 $E\left[L_t^2\right]$ 事先未知，而且在整个训练过程中可能会发生变化，因此为每个损失项保留了前 10 个值的历史记录，并在训练过程中进行动态更新。

但是对于 less noisy 的 $L_{\text {hybrid }}$ 没效果。

### 结果和消融实验
（略）

## 采样速度的提升

所有模型都是用 4000 个 step 训练的，在 GPU 上生成一个样本需要几分钟。

作者发现，预训练好的 Lhybrid 模型可以用比训练时少得多的 step（没有任何微调）生成高质量的样本，从而可以在数秒完成采样。

具体来说，其实可以从时间序列 $(1,2,\dots,T)$ 的一个子序列 $S$ 中进行采样（ stride 采样）。此时的 noise schedule 记为 $\bar{\alpha}_{S_t}$：
$$\beta_{S_t}=1-\frac{\bar{\alpha}_{S_t}}{\bar{\alpha}_{S_{t-1}}}, \quad \tilde{\beta}_{S_t}=\frac{1-\bar{\alpha}_{S_{t-1}}}{1-\bar{\alpha}_{S_t}} \beta_{S_t}$$
下图是不同的采样步和不同损失函数对应的样本的 FID 值：
![](image/Pasted%20image%2020230817203111.png)
结果表明，$L_{\text {hybrid }}$ 和 可学习的 sigma 可以得到最好的采样质量，此配置下，只需 100 steps 即可实现接近于 fully trained 的模型。

而且发现，使用 DDIM 在 step 小于 50 时可以产生更好的样本，而大于 50 之后更差。

## 和 GAN 的对比
（略）

## scaling

结果表明，随着计算量的增加，DDPM 的性能会提升。如下表：
![](image/Pasted%20image%2020230817204024.png)

