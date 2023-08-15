## 1. AE

AE的目标：输入 $x$，首先生成中间变量 $z$（低维），此过程相当于encoder，然后根据 $z$ 进行decoder输出 $x^\prime$ ，且使得输出的 $x^\prime$ 和 输入的 $x$ 尽可能相似。其对应的优化函数为：

$$L_{AE}(\theta, \phi) = \frac{1}{n} \Sigma_{i=1}^{n}(x^{(i)}-f_{\theta}(g_{\phi}(x^{(i)})))^2$$

其中，$g_{\phi}(\cdot)$ 代表encoder，$f_{\theta}(\cdot)$ 代表decoder，且 $x^\prime = f_{\theta}(g_{\phi}(x))$。

## 2. VAE
相较于AE，VAE将所有的特征都以概率分布建模。假设此分布以 $\theta$ 为参数，则VAE的目标为最大化分布 $p_{\theta}(x)$ 的概率，同时隐变量（latent variable）也以 $\theta$ 为参数进行建模 为$p_{\theta}(z)$。
### 2.0 生成过程
假设已经求出了最优的参数 $\theta^*$，通过隐变量生成x的过程如下：
  +  从 $p_{\theta^*}(x|z=z^{(i)})$ 中采样出隐变量 $z^{(i)}$
  +  计算此时的后验概率 $p_{\theta^*}(x|z=z^{(i)})$，则 $x^{(i)}$ 可以从此条件分布中生成得到。

### 2.1 优化
当关于 $x^{(i)}$ 的似然最大时，此时输出的 $x^\prime$ 越接近真实的 $x$ 的分布，对应的 $\theta$ 为最优参数：
$$\theta^{*} = \text{argmax}_\theta \prod_{i=1}^{n}p_\theta(x^{(i)})$$
为了方便计算，通过取最大对数似然，上式变为：
$$\theta^{*} = \text{argmax}_\theta \sum_{i=1}^{n}\text{log }p_\theta(x^{(i)})$$
其中，$p_\theta(x^{(i)})$ 又可以写成：
$$p_\theta(x^{(i)}) = \int p_{\theta}(x^{(i)}|z)p_{\theta}(z) \text{d}z$$
但是，上式通常不好计算，因此引入另一个关于隐变量、以 $\phi$ 为参数的分布 $q_{\phi}(z|x)$，这个分布相当于AE中的encoder（因为是关于隐变量 $z$ 的分布），那么，整个VAE的过程可以描述如下：
  + 从分布 $p_\theta(z)$ 中采样获得 $z$
  + 优化关于 $x$ 的似然函数，并使得encoder和decoder关于 $z$ 的后验分布 $q_{\phi}(z|x)、p_{\theta}(z|x)$ 尽可能的相似，也即最小化两者之间的KL散度

### 2.2 KL散度和ELBO
$q_{\phi}(z|x)、p_{\theta}(z|x)$ 之间的KL散度计算如下：

$$
\begin{aligned}
& D_{KL}(q_{\phi}(z|x)||p_{\theta}(z|x)) \\
& = \int q_{\phi}(z|x) \text{log } \frac{q_{\phi}(z|x)}{p_{\theta}(z|x)} \text{d} z \\
& = \int q_{\phi}(z|x) \text{log } \frac{q_{\phi}(z|x)p_\theta(x)}{p_{\theta}(z, x)} \text{d} z \\
& = \int q_{\phi}(z|x) (\text{log }p_\theta(x) +\text{log } \frac{q_{\phi}(z|x)}{p_{\theta}(z,x)} )\text{d} z \\
& = \text{log }p_\theta(x) + \int q_{\phi}(z|x) \text{log } \frac{q_{\phi}(z|x)}{p_{\theta}(x|z)p_{\theta}(z)} \text{d} z \\
& = \text{log }p_\theta(x) + \mathbin{E}_{z\sim q_{\phi}(z|x)}[\text{log } \frac{q_{\phi}(z|x)}{p_{\theta}(z)} - \text{log }p_\theta(x|z)] \\
& = \text{log }p_\theta(x) + D_{KL}(q_\phi(z|x)||p_\theta(z)) - \mathbin{E}_{z\sim q_{\phi}(z|x)}[\text{log }p_\theta(x|z)]
\end{aligned} 
$$
对上式进行移项，有：
$$\text{log }p_\theta(x) - D_{KL}(q_{\phi}(z|x)||p_{\theta}(z|x)) = \mathbin{E}_{z\sim q_{\phi}(z|x)}[\text{log }p_\theta(x|z)] - D_{KL}(q_\phi(z|x)||p_\theta(z))$$
上式左边的部分即为需要最大化的部分：最大似然且最小KL散度。因此定义VAE的损失函数为：
$$L_{VAE}(\phi, \theta) = -\mathbin{E}_{z\sim q_{\phi}(z|x)}[\text{log }p_\theta(x|z)] + D_{KL}(q_\phi(z|x)||p_\theta(z))$$
该损失函数也被称为证据下界（ELBO），其右侧第一项称为重建损失，第二项称为KL损失。

### 2.3 参数化技巧
损失函数中存在期望项，需要根据分布 $q_\phi(z|x)$ 对 $z$ 进行采样，该过程是一个随机过程且无法被用于梯度的反向传播计算，因此通常将 $z$ 进行参数化为一个确定的变量 $z = \tau_\phi(x,\epsilon)$，其中 $\epsilon$ 服从均值为 $0$，协方差矩阵为单位阵的多元高斯分布，且 $z = \mu + \sigma * \epsilon$，此时在encoder部分，需要学习从输入变量 $x$ 到 $\mu、\sigma$ 的变换 $\mu = f_{\phi_1}(x)、\sigma = f_{\phi_2}(x)$，且 $\sigma$ 通常为对角阵，此变换通常用神经网络来建模。
  +  $q_\phi(z|x) \sim \mathcal{N}(\mu,\sigma)$
  +  $p_\theta(z) \sim \mathcal{N}(0, I)$

此时，损失函数右侧的KL散度项可以写成：
$$D_{KL}(q_\phi(z|x)||p_\theta(z)) = \frac12 (\text{tr}(\sigma)+\mu\mu^T - k - \text{log det}(\sigma))$$
对于第一项，通常选取一个 $z$ 的样本，并把该样本对应的 $\text{log }p_\theta(x|z)$ 作为第一项的近似。在训练过程中，其实就是所有的输入样本的重构损失之和（这里的样本就相当于采样过程了）。