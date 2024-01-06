> NIPS 2023，NVIDIA，首尔大学

1. 现有的 zero-shot TTS 存在以下几个问题：
	1. 缺乏鲁棒性
	2. 采样速度慢
	3. 依赖于预训练的 codec 表征
2. 提出 P-Flow，快速的、data- efficient zero-shot  TTS，采用 speech prompt 实现说话人自适应，然后用 flow matching generative decoder 实现高质量和快速的语音合成
	1. speech-prompted text encoder 采用 speech prompts 和 text 来生成speaker- conditional 的 表征
	2. flow matching generative decoder 用这些表征来合成语音

## Introduction

1. 用 codec 得到的表征，又复杂又耗计算，而且还没有可解释性；本文就采用的是标准的 mel 谱
2. 同时为了提高推理速度，用最近的 ODE-based 生成模型，如  Flow Matching，和 diffusion 很相似，但是更简单也更直接
3. 本文贡献：
	1. 提出一种 speech prompt 方法，超过了 speaker  embedding，可以提供 in-context learning 的能力
	2. 采用 flow matching generative model 提高速度和质量
	3. 可以在更少的训练数据、更少的 encoder 下实现 comparable 的性能

## 相关工作（略）

## 方法

### P-Flow

P-Flow 训练和 mask-autoencoders 类似，给定 <文本，语音> 对，定义语音的 mel 谱 为 $x$，文本为 $c$，$m^p$  为 indicator mask，用于随机 mask 3s 的音频段。定义 $(1-m^p)\cdot x$ ，其中 $p$ 表示这个变量会被替换为任意的 3s 的 prompt。

P-Flow 的训练目标为：给定 $c$ 和 $x^p$ 的条件下重构 $x$，即学习条件概率 $p(x|c,x^p)$。

引入一个 text encoder $f_{enc}$（架构用的是非自回归的 transformer ），输入 文本 $c$ 和随机段 $x^p$ 来生成 speaker- conditional 的 text representation $h_{c}=f_{enc}(x^{p},c)$，然后通过 flow- matching 生成模型将其映射为 mel 谱。

![](image/Pasted%20image%2020231128211250.png)

同时采用了一个 encoder loss 来直接最小化 text encoder representation 和 mel 谱之间的距离。

由于 $h_c$ 和 $x$ 的长度不同，采用 MAS 来得到对齐 $A=MAS(h_{c},x)$。

实际用的时候发现，即使 $x^p$ 是随机选的，训练的时候模型会 collapse 为简单复制粘贴 $x^p$，于是把 $x^p$ 这段的 loss mask 掉，此时的 loss 定义为：
$$L_{enc}^p=MSE(h\cdot m^p,x\cdot m^p)$$

采用 flow-matching 生成模型作为 decoder 建模概率分布 $p(x|c,x^p)\:=\:p(x|h)$。

flow-matching decoder 建模 连续归一化流（Continuous Normalizing Flows）的条件向量场 $v_t(\cdot|h)$，这个向量表示了从标准正太分布转为数据分布的条件映射。然后训练用的 loss 是 flow-matching loss $L_{cfm}^{p}$（这里也用了 mask $m^p$）。

然后用的是类似 Glow-TTS 中的 duration predictor，采用 text encoder 的输出作为其输入。这个模块用于估计对数 duration $\log\widehat{d}$，其目标函数为 MAS 得到的 $\log d$ 之间的 MSE loss。

总的训练损失为 $L=L_{enc}^{p}+L_{cfm}^{p}+L_{dur}$。推理的时候用从参考样本中的随机段作为 prompt，还有文本作为输入。然后用 duration predictor 得到的 $\hat{d}$，拓展之后通过 flow-matching 得到 mel 谱。

### Flow Matching Decoder

采用 flow matching 来建模条件分布。

flow matching 用于拟合数据分布 $p_1(x)$ 和简单分布（正态分布）$p_0(x)$ 之间的time-dependent probability path。

采用 Conditional Flow Matching，其路径更简单直接，从而可以得到更少的采样次数。为了简单，下面分析中忽略条件 $h$。

采用下述 ODE 定义 flow $\phi:[0,1]\times\mathbb{R}^d\to\mathbb{R}^d$ 为两个密度函数之间的映射：
$$\frac d{dt}\phi_t(x)=v_t(\phi_t(x));\quad\phi_0(x)=x$$
