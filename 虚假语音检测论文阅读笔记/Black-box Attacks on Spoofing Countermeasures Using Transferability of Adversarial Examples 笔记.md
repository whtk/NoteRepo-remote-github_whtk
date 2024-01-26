
> 这篇文章的目的不是防御 而是 攻击

1. 本文目标：在高 transferability 的对抗样本下的进行黑盒对抗欺诈
2. 使用 MI-FGSM 提高对抗样本的 transferability，同时采用迭代集成方法（IEM）来进一步提高 transferability


## Introduction

1. 反欺诈模型容易受到对抗攻击
2. 对抗攻击分为三类：
	+ 白盒：攻击者可以访问受害者模型的所有信息的攻击
	+ 灰盒：攻击者仍然可以多次查询模型，以获取受攻击模型的替代模型
	+ 黑盒：攻击者只能访问很少的信息
3. 攻击的 transferability 是指，即使从特定的白盒受攻击的模型生成，它也可以成功攻击其他模型
4. 本文研究了 transferability 对抗样本攻击下欺骗对抗模型的脆弱性
5. 本文主要贡献：
	1. 研究在黑盒攻击下反欺骗系统的鲁棒性
	2. 证明 MI-FGSM 攻击和基于集成的攻击提高了音频对抗样本的 transferability 
	3. 提出了一种新的迭代集成方法（IEM），提高了对抗样本的 transferability


## 反欺诈模型

1. LCNN：[[STC Antispoofing Systems for the ASVspoof2019 Challenge 笔记]]
2. AFN：[[Attentive Filtering Networks for Audio Replay Attack Detection 笔记]]
3. SENet + ResNet：[[ASSERT- Anti-Spoofing with Squeeze-Excitation and Residual neTworks 笔记]]

具体原理详见笔记啦。

## 音频对抗样本

音频对抗样本是具有小扰动的信号，人类无法察觉，但会改变机器学习系统的输出。

### 攻击算法

1. Fast Gradient Sign Method（FGSM）基于梯度的正负来实现对抗样本：$$\mathbf{x}^{\prime}=\mathbf{x}+\epsilon \cdot \operatorname{sign}\left(\nabla_{\mathbf{x}} L\left(\mathbf{x}, y^{\text {true }} ; \theta\right)\right)$$FGSM 的扰动满足无穷范数下的边界，即 $\left\|\mathbf{x}^{\prime}-\mathbf{x}\right\|_{\infty} \leq \epsilon$
2. Iterative Fast Gradient Sign Method（I-FGSM）就是多跑几步：$$\begin{gathered}
\mathbf{x}_0^{\prime}=\mathbf{x} \\
\mathbf{x}_{i+1}^{\prime}=\operatorname{Clip}_x^\epsilon\left\{\mathbf{x}_{\mathbf{i}}{ }^{\prime}+\alpha \cdot \operatorname{sign}\left(\nabla_{x_i^{\prime}} L\left(\mathbf{x}_{\mathbf{i}}^{\prime}, y^{\text {true }} ; \theta\right)\right)\right\}
\end{gathered}$$其中，$i$ 表示迭代的次数，$\operatorname{Clip}_x^\epsilon$ 用来满足武器范数的边界条件。跌打方法的攻击性更强，但是 transferability 更差。
3. Momentum Iterative Fast Gradient Sign Method（MI-FGSM）提高了transferability ：$$\begin{gathered}
\mathbf{g}_{i+1}=\mu \cdot \mathbf{g}_i+\frac{\nabla_x L\left(\mathbf{x}, y^{\text {true }} ; \theta\right)}{\left\|\nabla_x L\left(\mathbf{x}, y^{\text {true }} ; \theta\right)\right\|_1} \\
\mathbf{x}_{\mathbf{i}+\mathbf{1}}^{\prime}=\operatorname{Clip}_{\mathbf{x}}^\epsilon\left\{\mathbf{x}_i^{\prime}+\alpha \cdot \operatorname{sign}\left(\mathbf{g}_{i+1}\right)\right\}
\end{gathered}$$当 $\mu$ 为 $0$ 时退化成 I-FGSM。

### 多模型集成

论文  Boosting Adversarial Attacks with Momentum 采用多个模型的 logits 的加权和来得到一个集成模型，可以实现更高的 transferability ，为了同时攻击 $K$ 个白盒模型，融合 logits 如下：$$l\left(x ; \theta_1, \ldots, \theta_K\right)=\Sigma_{k=1}^K w_k l_k\left(x ; \theta_k\right)$$其中，$l_k\left(x ; \theta_k\right)$ 表示第 $k$ 个白盒的 logits，$w_k$ 为权重。


### 迭代集成方法 IEM

找到可以同时欺骗所有集成白盒模型的对抗样本，采用迭代策略来最大化所有使用的集成模型上的攻击成功率，并在黑盒模型上实现了更多 transferable 的对抗样本，算法如下：
$$\begin{aligned}
&\hline \text { Algorithm } 1 \text { Iterative Ensemble Method } \\
&\hline \text { Input: White-box models } L=\left\{l_1 \ldots l_K\right\} \text {, clean input } x \\
&\quad \text { with corresponding label } y \text {, adversarial attack function } f \\
&\quad \text { (such as FGSM, I-FGSM, MI-FGSM) with parameters } \theta= \\
&\quad\{\alpha, \epsilon\} \text {, perturbation range } \epsilon \text {, step size } \alpha \text {, iteration times } T \\
&\text { Output: Adversarial example } x^{\prime} \\
&\text { 1: Initialize perturbation } \delta_0 \leftarrow \text { random start in the } \epsilon \text {-ball } \\
&\text { 2: for iteration time } t \leftarrow 1 \text { to } T \text { do } \\
&\text { 3: for model } l_i \in L \text { do } \\
&\text { 4: } \quad \delta_m \leftarrow f_\theta\left(\delta_{m-1}, y ; l_i\right) \text { (where } m=K *(t-1)+i \text { ) } \\
&\text { 5: end for } \\
&\text { 6: end for } \\
&\text { 7: } \hat{\delta} \leftarrow \delta_{T \times K} \\
&\text { 8: return } x^{\prime}=x+\hat{\delta} \\
&\hline
\end{aligned}$$
下图给出了 IEM 算法的解释：![[Pasted image 20221115195514.png]]使用基于集成的方法，可以通过同时欺骗多个模型来提高对抗样本的 transferability 。

## 实验和结果

数据集：ASVspoof 2019 LA

特征：对数幅度谱

进行 XAB 听力测试表明，扰动 $\epsilon \le 10$  时人类无法察觉。

评价指标：攻击成功率，也就是说，如果一个对抗样本可以使收攻击的模型判断失误，则将其视为成功的样本。


1. ![[Pasted image 20221115200812.png]]和![[Pasted image 20221115201010.png]]表明，白盒攻击的效果两个算法都很好，但是 I-FGSM 生成的样本的 transferability 更差。
2. 下表表明了迭代集成方法的效果最优：![[Pasted image 20221115201135.png]]
3. 下表表明，扰动越大，transferability 越高：![[Pasted image 20221115201219.png]]





