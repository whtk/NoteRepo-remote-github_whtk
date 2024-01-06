> ICASSP 2023，SK Telecom, South Korea

1. Single-stage 的 TTS 的效果已经超过了 two-stage
2. 提出 VITS2，改进架构和训练机制，从而提升了语音自然度、多说话人下的相似度、训练和推理效率

## Introduction

1. VITS 有下面的问题：
	1. intermittent unnaturalness（断续不自然）
	2. duration predictor 效率很低
	3. 对齐受限
	4. 说话人相似度不够
	5. 训练慢
	6. 依赖于 phoneme 
2. 本文提出：
	1. 采用对抗学习来训练 stochastic duration predictor
	2. 采用 transformer block 来改进 normalizing flows
	3. 采用 speaker-conditioned text encoder 来建模多说话人特征

## 方法

### 基于 Time Step-wise Conditional Discriminator 的 Stochastic Duration Predictor

基于 flow 的 SDP 可以提升语音合成的自然度，但是需要相对更多的计算。于是提出基于对抗学习的 SDP，如图：
![](image/Pasted%20image%2020231024152628.png)

用的是条件 discriminator，输入条件和 duration predictor 一致。提出一个 time wise 的 discriminator 用来判别所有 token 的 duration，采用两种损失，least-squares loss 用于对抗学习，还有一个 mean squared error loss：
$$\begin{aligned}
L_{adv}(D)& =\mathbb{E}_{(d,z_d,h_{text})}\left[(D(d,h_{text})-1)^2\right.  \\
&\left.+(D(G(z_d,h_{text}),h_{text}))^2\right], \\
L_{adv}(G)& =\mathbb{E}_{(z_d,h_{text})}\bigg[(D(G(z_d,h_{text}))-1)^2\bigg],  \\
L_{mse}& =MSE(G(z_d,h_{text}),d) 
\end{aligned}$$
> 真实的 duration通过判别器是1，模型预测的 duration 糖果判别器是 0。

### 基于高斯噪声的 MAS

MAS 效果很好，但是当搜索和优化一个特定的对齐之后，就不会搜索其他的对齐了。

于是在计算概率的时候添加了一个额外的噪声，从而使得模型有概率搜索到其他的对齐。

只在训练开始阶段加这个噪声，此时动态规划计算如下：
$$\begin{aligned}
&P_{i,j} =\log\mathcal{N}(z_j;\mu_i,\sigma_i)  \\
&Q_{i,j} =\max_A\sum_{k=1}^j\log\mathcal{N}(z_k;\mu_{A(k)},\sigma_{A(k)})  \\
&=\max(Q_{i-1,j-1},Q_{i,j-1})+P_{i,j}+\epsilon 
\end{aligned}$$
其中 $\epsilon$ 为噪声。

### 基于 transformer block 的 normalizing flow

尽管之前的 flow 模型中的卷积层可以捕获邻近的 pattern，但是受限于感受野仍然无法捕获长期依赖。于是在 residual connection 中添加一个小的 transformer block，如图：
![](image/Pasted%20image%2020231024154847.png)
如图是实际的 attention map：
![](image/Pasted%20image%2020231024155449.png)
黄色部分是只用卷积层提取的感受野，所以实际上用卷积层看到的信息很少。

### 基于 说话人的 text encoder

由于说话人的一些特别的发音或者重音会极大地影响语音特征的表达性，但是这些信息不包含在文本中，于是在 text encoder 中引入 speaker 的信，如图：
![](image/Pasted%20image%2020231024160016.png)

## 实验

![](image/Pasted%20image%2020231024160106.png)



