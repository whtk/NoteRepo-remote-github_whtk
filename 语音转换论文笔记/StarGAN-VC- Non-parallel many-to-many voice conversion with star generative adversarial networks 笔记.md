> SLT 2018，Nippon Telegraph and Telephone Corporation

1. 非并行、many-to-many
2. 可以产生实时的语音
3. 只需要几分钟的训练样本即可产生真实的声音，效果优于[VAE-GAN](https://arxiv.org/abs/1704.00849)（本文的baseline）


StarGAN可以看成是CycGAN的改进，因此先介绍CycGAN的基本原理。
### CycleGAN-VC 原理
定义 $\mathbf{x} \in \mathbb{R} ^{ Q \times N}$ 和 $\mathbf{y} \in \mathbb{R} ^ {Q \times M}$ 为输入输出的声学特征域（如Mel谱）$X,Y$，其中 $Q$ 为特征维度，$N,M$ 为输入输出序列长度。则CycGAN的目标为，学习从输入 $\mathbf{x}$ 到域 $\mathbf{y}$ 的映射 $G$ 和逆映射 $F$，定义判别器 $D_X,D_Y$ 用来判别其输入的特征是否 属于域 $X,Y$ 的 真实语音 的声学特征。同时定义：
$$
\begin{aligned}
\mathcal{L}_{\mathrm{adv}}^{D_{Y}}\left(D_{Y}\right)=&-\mathbb{E}_{\mathbf{y} \sim p_{Y}(\mathbf{y})}\left[\log D_{Y}(\mathbf{y})\right] \\
&-\mathbb{E}_{\mathbf{x} \sim p_{X}(\mathbf{x})}\left[\log \left(1-D_{Y}(G(\mathbf{x}))\right)\right], \\
\mathcal{L}_{\mathrm{adv}}^{G}(G)=& \mathbb{E}_{\mathbf{x} \sim p_{X}(\mathbf{x})}\left[\log \left(1-D_{Y}(G(\mathbf{x}))\right)\right] \\
\mathcal{L}_{\mathrm{adv}}^{D_{X}}\left(D_{X}\right)=&-\mathbb{E}_{\mathbf{x} \sim p_{X}(\mathbf{x})}\left[\log D_{X}(\mathbf{x})\right] \\
&-\mathbb{E}_{\mathbf{y} \sim p_{Y}(\mathbf{y})}\left[\log \left(1-D_{X}(F(\mathbf{y}))\right)\right], \\
\mathcal{L}_{\mathrm{adv}}^{F}(F)=& \mathbb{E}_{\mathbf{y} \sim p_{Y}(\mathbf{y})}\left[\log \left(1-D_{X}(F(\mathbf{y}))\right],\right.
\end{aligned}
$$
但是仅训练 $G,F$ 并不能确保生成器能够保留语音的语言信息，因此引入循环一致损失：
$$
\begin{aligned}
\mathcal{L}_{\mathrm{cyc}}(G, F) &=\mathbb{E}_{\mathbf{x} \sim p_{X}(\mathbf{x})}\left[\|F(G(\mathbf{x}))-\mathbf{x}\|_{1}\right] \\
&+\mathbb{E}_{\mathbf{y} \sim p_{Y}(\mathbf{y})}\left[\|G(F(\mathbf{y}))-\mathbf{y}\|_{1}\right]
\end{aligned}
$$
直观的来看，就是使序列依次通过两个生成器之后得到的结果和原序列尽可能的相同 $F(G(\mathbf{x})) \simeq \mathbf{x} \text { and } G(F(\mathbf{y})) \simeq \mathbf{y}$ ，以及所谓的恒等映射损失：
$$
\begin{aligned}
\mathcal{L}_{\mathrm{id}}(G, F) &=\mathbb{E}_{\mathbf{x} \sim p_{X}(\mathbf{x})}\left[\|F(\mathbf{x})-\mathbf{x}\|_{1}\right] \\
&+\mathbb{E}_{\mathbf{y} \sim p_{Y}(\mathbf{y})}\left[\|G(\mathbf{y})-\mathbf{y}\|_{1}\right]
\end{aligned}
$$
来确保生成器 $G$ 输入 $\mathbf{x}$ 之后输出尽可能是 $\mathbf{x}$，$F$ 输入 $\mathbf{y}$ 之后输出尽可能是 $\mathbf{y}$ 。最终整个系统的目标函数为：
$$
\begin{aligned}
\mathcal{I}_{G, F}(G, F)=& \mathcal{L}_{\mathrm{adv}}^{G}(G)+\mathcal{L}_{\mathrm{adv}}^{F}(F) \\
&+\lambda_{\mathrm{cyc}} \mathcal{L}_{\mathrm{cyc}}(G, F)+\lambda_{\mathrm{id}} \mathcal{L}_{\mathrm{id}}(G, F) \\
\mathcal{I}_{D}\left(D_{X}, D_{Y}\right)=& \mathcal{L}_{\mathrm{adv}}^{D_{X}}\left(D_{X}\right)+\mathcal{L}_{\mathrm{adv}}^{D_{Y}}\left(D_{Y}\right)
\end{aligned}
$$

### StarGAN-VC 原理

CycleGAN的一个最大的缺点在于，只能生成 one-to-one 的映射，StarGAN的提出解决了这个问题，可以实现 many-to-many 的映射。

设目标（target）的 attribute 为向量 $c$，则生成器模型为，$\hat{\mathbf{y}}=G(\mathbf{x},c)$，其中 $c$ 可以认为是一个拼接的one-hot向量，为 1 表明对应的输出需要该特征，反之则不需要。同时还有一个判别真伪（real/fake）的判别器 $D$ 和一个域分类器 $C$，$C$ 用来产生 $\mathbf{y}$ 的分类概率 $P_C(c|\mathbf{y})$（用于分类 attribute），$D$ 可以产生概率 $D(\mathbf{y},c)$，$\mathbf{y}$ 是真实的语音特征(Groud Truth)，对比到图片中，$\mathbf{y}$ 就是 real image，$\hat{\mathbf{y}}$ 为 fake image。
那么，定义**对抗损失**为：
$$
\begin{aligned}
\mathcal{L}_{\mathrm{adv}}^{D}(D)=&-\mathbb{E}_{c \sim p(c), \mathbf{y} \sim p(\mathbf{y} \mid c)}[\log D(\mathbf{y}, c)] \\
&-\mathbb{E}_{\mathbf{x} \sim p(\mathbf{x}), c \sim p(c)}[\log (1-D(G(\mathbf{x}, c), c))] \\
\mathcal{L}_{\mathrm{adv}}^{G}(G)=&-\mathbb{E}_{\mathbf{x} \sim p(\mathbf{x}), c \sim p(c)}[\log D(G(\mathbf{x}, c), c)]
\end{aligned}
$$
同时定义**域分类损失**：
$$
\begin{aligned}
&\mathcal{L}_{\mathrm{cls}}^{C}(C)=-\mathbb{E}_{c \sim p(c), \mathbf{y} \sim p(\mathbf{y} \mid c)}\left[\log p_{C}(c \mid \mathbf{y})\right] \\
&\mathcal{L}_{\mathrm{cls}}^{G}(G)=-\mathbb{E}_{\mathbf{x} \sim p(\mathbf{x}), c \sim p(c)}\left[\log p_{C}(c \mid G(\mathbf{x}, c))\right]
\end{aligned}
$$
当输入为 GT 时（${\mathbf{y}}|c$），分类器分类正确则损失函数的值小，当输入生成器的结果（fake voice，$\hat{\mathbf{y}}|c$）时，分类器分类正确损失函数的值也小。
**循环一致损失**为：
$$
\begin{aligned}
&\mathcal{L}_{\text {cyc }}(G) \\
&=\mathbb{E}_{c^{\prime} \sim p(c), \mathbf{x} \sim p\left(\mathbf{x} \mid c^{\prime}\right), c \sim p(c)}\left[\left\|G\left(G(\mathbf{x}, c), c^{\prime}\right)-\mathbf{x}\right\|_{\rho}\right]
\end{aligned}
$$
解释为，把真实的输入（real）$\mathbf{x}$ 和目标 attribute $c$ 输入 $G$，得到 fake 的输出 $\hat{\mathbf{y}}$，然后再把 fake 的输出和原始的 $\mathbf{x}$ 对应的 attribute $c^\prime$ **再一次**输入 $G$，得到 $\hat{\mathbf{x}}$ 和原始输入 $\mathbf{x}$ 两者应该一样。
最后，**恒等映射损失**为：
$$
\mathcal{L}_{\mathrm{id}}(G)=\mathbb{E}_{c^{\prime} \sim p(c), \mathbf{x} \sim p\left(\mathbf{x} \mid c^{\prime}\right)}\left[\left\|G\left(\mathbf{x}, c^{\prime}\right)-\mathbf{x}\right\|_{\rho}\right]
$$
最终，生成器、判别器、分类器的损失函数如下：
$$
\begin{aligned}
\mathcal{I}_{G}(G)=& \mathcal{L}_{\mathrm{adv}}^{G}(G)+\lambda_{\mathrm{cls}} \mathcal{L}_{\mathrm{cls}}^{G}(G) \\
&+\lambda_{\mathrm{cyc}} \mathcal{L}_{\mathrm{cyc}}(G)+\lambda_{\mathrm{id}} \mathcal{L}_{\mathrm{id}}(G) \\
\mathcal{I}_{D}(D)=& \mathcal{L}_{\mathrm{adv}}^{D}(D) \\
\mathcal{I}_{C}(C)=& \mathcal{L}_{\mathrm{cls}}^{C}(C)
\end{aligned}
$$
其中，$\lambda$ 对应不同损失函数的权值。

两个模型的对比如下：
![](image/Pasted%20image%2020230916215719.png)

### STARGAN-VC 转换过程和模型架构
假定输入的声学特征为mel谱，则 inference 的过程为：
1. $\hat{\mathbf{y}} = G(\mathbf{x},c)$, $c$ 为目标 attribute（也就是目标说话人的特征）
2. 输出波形：
   + 简单的方法是直接把 $\hat{\mathbf{y}}$ 作为声码器的输入
   + 或者计算重构特征序列 $\hat{\mathbf{y}}^{\prime}=G\left(\mathbf{x}, c^{\prime}\right)$ （前提，$c^\prime$ 已知），然后根据 $\hat{\mathbf{y}},\hat{\mathbf{y}}^{\prime}$ 计算谱增益函数，将得到的谱增益函数乘以谱包络，得到的结果再通过声码器。
  
使用CNN来构建生成器，生成器包含encoder和decoder两个部分；判别器和分类器输入为声学特征序列，输出为概率序列。细节如下：
1. 生成器中，使用 gated CNN（最初被用于语言模型， 从LSTM中获取灵感，增加一个gate来控制卷积的激活值，其实就是做dot product），在encoder中，$\mathbf{x}$ 为输入，在decoder中，$\mathbf{x}, c$ 进行concatenate后作为输入。
2. 利用 PatchGAN的思想设计判别器，也使用 gated CNN进行设计，输出概率序列表示输入 $\mathbf{y}$ 的每个segment的是real or fake的概率，最终的概率为所有概率的乘积。
3. 分类器也用 gated CNN 设计，输出不同 segment 的概率，最终计算概率乘积。

特征参数：
1. 每5ms计算一次谱包络、F0（对数）、APs
2. 36维的MCC
