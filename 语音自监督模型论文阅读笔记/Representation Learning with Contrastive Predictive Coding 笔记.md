
1. 提出一种通用的无监督学习方法来从高维数据中提取表征，即对比预测编码（CPC）
2. 使用概率对比损失使潜在空间捕获对预测未来样本最有用的信息，模型还使用负样本。
3. 能够在语音、图像、文本和3D环境强化学习等领域实现强大的性能

## Introduction  

无监督学习最常见的策略之一是预测未来的、缺失的或上下文的信息。

本文假设这些方法是有效的，因为预测相关值通常有条件地依赖于共享高级潜在信息。

本文贡献：
1. 将高维数据压缩到一个更紧凑的潜在嵌入空间中，更容易建模条件预测
2. 在潜在空间中使用自回归模型来预测未来的 time step。
3. 使用 Noise-Contrastive Estimation 中相似的损失函数来进行端到端的模型训练

## CPC

### Motivation and Intuitions

主要的目的是是，学习 编码高维信号不同部分之间的潜在共享表征，丢弃低级信息和局部噪声。

时间序列建模在预测未来数据时，有一个 slow features 的特性。

预测高维数据的挑战之一是 unimodal losses ，通常需要强大的生成模型来进行细节的建模；但是这也导致了大量的计算，同时忽略了上下文的信息。这表明，直接建模 $p(x\mid c)$ 或许不是提取 $x$ 和 $c$ 共享信息的最佳方法。

相反，当预测未来信息时，编码未来的 $x$ 和当前的 $c$ 到一个紧凑分布的向量中，这个过程需要最大程度的保留互信息。
> 互信息定义为：$$I(x ; c)=\sum_{x, c} p(x, c) \log \frac{p(x \mid c)}{p(x)}$$

通过最大化 编码后的表征 的互信息，可以提取输入中公有的潜在变量。

### CPC

CPC 模型如图：![[image/Pasted image 20221204102606.png]]
首先，非线性 encoder $g_{\text {enc }}$ 将输入序列 $x_t$ 映射到潜在表征 $z=g_{\text{enc}}(x_t)$  ，然后使用 自回归模型 $g_{\text{ar}}$ 将所有的 $z _{\leq t}$  summarize 以产生上下文潜在表征 $c_t=g_{\mathrm{ar}}\left(z_{\leq t}\right)$ 。

正如前一节所述，没有使用生成模型 $p_k\left(x_{t+k} \mid c_t\right)$ 来预测 $x_{t+1}$，而是建模了一个 density ratio 来保留 $x_{t+k}$ 和 $c_t$ 之间的互信息，即：$$f_k\left(x_{t+k}, c_t\right) \propto \frac{p\left(x_{t+k} \mid c_t\right)}{p\left(x_{t+k}\right)}$$
具体到实现，使用 log-bilinear 模型来建模这个 ratio，即： $$f_k\left(x_{t+k}, c_t\right)=\exp \left(z_{t+k}^T W_k c_t\right)$$
这里需要注意，每个 time step 的 $W_k$ 都不一样。
> 其实也可以使用其他的非线性网络或者RNN来建模

通过使用 density ratio $f\left(x_{t+k}, c_t\right)$ 和 encoder 来推理 $z_{t+k}$，模型不再对高维分布 $x_{t_k}$ 的建模。虽然不能直接评估 $p(x)$ 或 $p(x \mid c)$，但可以使用这些分布的样本，从而可以使用噪声对比估计和重要性采样等技术，来将 target value 和 随机采样的 negative value 进行对比。

模型中的 $z_t$ 和 $c_t$ 都可以用于下游任务的表征，如果是在需要上下文信息的任务中（如语音识别），$c_t$ 可能更好，在不需要上下文的任务中 $z_t$ 可能更好。

模型没有指定特定的 encoder 或自回归模型，满足条件的模型都可以用。

### InfoNCE Loss 和 Mutual Information Estimation

encoder 和 自回归模型联合训练来优化基于 NCE 的损失，也就是所谓的 InfoNCE。

给定样本集 $X=\left\{x_1, \ldots x_N\right\}$，其中包含一个从 $p\left(x_{t+k} \mid c_t\right)$ 中采样的 positive 样本和 $N-1$ 个从$p\left(x_{t+k}\right)$中采样的 negative 样本，通过优化 InfoNCE 损失：$$\mathcal{L}_{\mathrm{N}}=-\underset{X}{\mathbb{E}}\left[\log \frac{f_k\left(x_{t+k}, c_t\right)}{\sum_{x_j \in X} f_k\left(x_j, c_t\right)}\right]$$
来使得 $f_k\left(x_{t+k}, c_t\right)$ 估计 density ratio  。

上式中的损失是正确分类 positive 样本的 categorical cross-entropy，其中 $\frac{f_k}{\sum_X f_k}$ 是模型的预测。将此损失的最佳概率写为 $p\left(d=i \mid X, c_t\right)$，其中 $[d=i]$ 表示 $x_i$ 是 positive 样本。样本 $x_i$ 的概率是从条件分布 $p\left(x_{t+k} \mid c_t\right)$而非 proposal 分布 $p(x_{t+k})$ 中得到的，可推导如下：$$\begin{aligned}
p\left(d=i \mid X, c_t\right) & =\frac{p\left(x_i \mid c_t\right) \prod_{l \neq i} p\left(x_l\right)}{\sum_{j=1}^N p\left(x_j \mid c_t\right) \prod_{l \neq j} p\left(x_l\right)} \\
& =\frac{\frac{p\left(x_i \mid c_t\right)}{p\left(x_i\right)}}{\sum_{j=1}^N \frac{p\left(x_j \mid c_t\right)}{p\left(x_j\right)}} .
\end{aligned}$$最大化上面这个式子，即最大化模型“成功分辨出每个正负样本的能力”，也就是最大化定义的密度比，也就是最大化 $c_t$ 与 $x_{t+k}$ 的互信息。
此时，$c_t$ 和 $x_{t+k}$ 之间的互信息满足：$$I\left(x_{t+k}, c_t\right) \geq \log (N)-\mathcal{L}_{\mathrm{N}}$$
随着 $N$ 的变大，互信息的下界也增加。

### 相关工作（略）

## 实验

### 音频
使用 100h 的LS 数据集，模型的信息见论文。
![[image/Pasted image 20221204153140.png]]
结果如图。

使用提取的表征进行音素分类和说话人分类：
![[image/Pasted image 20221204153429.png]]
1. 效果甚至接近有监督的学习！！
2. CPC 可以同时捕获语音的内容和说话人的特征！！

### 其他（略）