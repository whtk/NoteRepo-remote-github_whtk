## Voice Conversion from Non-parallel Corpora Using Variational Auto-encoder 笔记（2016年10月发表）
1. 本文贡献：提出了一种对非对其语音数据下的谱转换（Spectral Conversion）方法
2. 第一篇将VAE应用于VC中的论文

### Introduction
1. 传统的方法需要对源语音和目标语音应用DTW方法进行对齐——并行数据
2. 对于非并行数据
   1. 进行语音识别然后显式对齐，但是对跨语言不成立
   2. INCA：帧级别的对齐
   3. 分别建立源和目标语音的帧簇，然后建立两者之间的映射

### 原理
将VAE用于语音转换的灵感来源于手写数字生成，在数字中，identity是具体的数字，variation是手写风格，在语音中，identity是源说话人，variation是语音内容。
#### 符号设置
定义源说话者为 $s$，目标说话者为 $t$，集合 $X_s = \{x_{s, n}\}_{n=1}^{N_s}$ 表示所有源语音中的帧，集合 $X_t = \{x_{t, n}\}_{n=1}^{N_t}$ 表示所有目标语音中的帧，$N_s、N_t$ 分别代表目标和源语音中的总帧数。

#### AE for SC
传统的SC寻求转换函数使得：
$$ \hat{x}_{t,m} = f(x_{s,n}) $$
其中 $f(\cdot)$ 是转换函数。

本文显式地加入一个说话人表征向量 $y_n$ 到SC的公式中。并认为，在编码器端 $f_\phi(\cdot)$ 中是说话人独立的，也即忽略了输入数据到底来自源还是目标说话人，因此可以将 $x_{s,n}$ 和 $x_{t,m}$ 统一表示为 $x_n$，并且将观察到的帧转换为说话人独立的隐变量 $z_n$：
$$ z_n = f_\phi({x_n}) $$
$z_n$ 包含了和说话人无关的信息，如语音内容。
下一步需要一个解码器 $f_\theta(\cdot)$ 重构和说话人相关的帧。这里则将说话人特征向量当成另一个隐变量作为解码器的输入，和 $z_n$ 拼接后输入解码器来重构 $\hat{x}_n$：
$$ \hat{x}_n = f_\theta(z_n, y_n) $$
以上过程无需对齐数据，只需输入 $x_n$ 和 $y_n$ 即可，他们分别对应语音内容和说话人的特征，两者可以被解耦。

#### VAE for SC
将SC看做是VAE的生成过程，尝试最大化每个独立帧的联合概率分布：
$$\text{log }p_\theta (X) = \sum_{n=1}^N \text{log } p_\theta(x_n)$$
而 $\text{log } p_\theta(x_n)$ 可以写成：
$$\text{log } p_\theta(x_n) = D_{KL}(q_\phi(z_n|x_n)||p_\theta(z_n|x_n))+\mathcal{L}(\theta,\phi; x_n)$$
其中 $\mathcal{L}(\theta,\phi; x_n)$ 为变分下界（ELBO）：
$$\mathcal{L}(\theta,\phi; x_n) = -D_{KL}(q_\phi(z_n|x_n)||p(z_n))+E_{z\sim q_\phi(z_n|x_n)}[\text{log }p_\theta(x_n|z_n)]$$
目标就是优化上式，或者说求最优的 $\phi$ 和 $\theta$。
应用重参数技巧，期望项计算如下：
$$ E_{z\sim q_\phi(z_n|x_n)}[\text{log }p_\theta(x_n|z_n)] \approx \sum_{l=1}^L \text{log }p_\theta(x_n|\hat{z}_n, y_n) $$
其中，$\hat{z}_n = \mu+\epsilon * \sigma $。
同时，在高斯分布的假设下，KL散度计算如下（$D$ 为隐变量的维度）：
$$-D_{KL}(q_\phi(z_n|x_n)||p(z_n)) = \frac12 \sum_{d=1}^D (1+\text{log }\sigma_{z_{n,d}}^2 - \mu_{z_{n,d}}^2 - \sigma_{z_{n,d}}^2)$$

##### 可见空间建模
本文同时假设了输出对数谱遵循方差为对角阵的高斯分布：
$$
\begin{aligned}
\hat{ {x}}_{n} & \sim \mathcal{N}\left( {x}_{n} ;  {\mu}_{ {x}_{n}},  {\sigma}_{ {x}_{n}}\right) \\
 {\mu}_{ {x}_{n}} &=f_{ {\theta}_{1}}\left( {z}_{n},  {y}_{n}\right) \\
\log  {\sigma}_{ {x}_{n}} &=f_{ {\theta}_{2}}\left( {z}_{n},  {y}_{n}\right)
\end{aligned}
$$
其中 $f_{\theta_1}$ 和 $f_{\theta_2}$ 是用神经网络拟合的非线性函数。则期望项中的对数似然可以表示为（$D$ 为可见特征空间的维度）：
$$
\begin{aligned}
& \log p_{\theta}\left( {x}_{n} \mid \hat{ {z}}_{n},  {y}_{n}\right)=\log \mathcal{N}\left( {x}_{n} ;  {\mu}_{ {x}_{n}}, \operatorname{diag}\left( {\sigma}_{ {x}_{n}}\right)\right) \\
=&-\frac{1}{2} \sum_{d=1}^{D}\left(\log \left(2 \pi \sigma_{ {x}_{n, d}}^{2}\right)+\frac{\left(x_{d}-\mu_{ {x}_{n, d}}\right)^{2}}{\sigma_{ {x}_{n, d}}^{2}}\right)
\end{aligned}
$$
即不是直接用神经网根据 $z$ 来计算 $\hat{x}$，而是同样假设输出 $\hat{x}$ 也服从高斯分布，利用神经网络来拟合从 $z$ 到 $\hat{x}$ 的均值和方差的映射。
##### 训练过程
1. $y_n$ 直接用 one-hot 向量
2. 不区分源和目标语音，统一为 $x_n$，在训练过程中每帧都加入了说话人特征组成训练集：$(X,Y) = \left\{\left({x}_{n}, {y}_{n}\right)\right\}_{n=1}^{N=N_{s}+N_{t}}$