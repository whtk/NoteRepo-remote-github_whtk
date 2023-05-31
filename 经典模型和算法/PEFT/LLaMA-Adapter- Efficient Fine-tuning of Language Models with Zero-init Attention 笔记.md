> CUHK MMLab，2023

1. 提出 LLaMA-Adapter，一种轻量级的 adaptation 方法，可以高效地利用指令来微调模型
2. 采用 52K self-instruct demonstrations，LLaMA-Adapter 只有 1.2M 的可学习的参数（相比于 LLaMA 7B 模型），在 8台 A100 GPU 上只需要训练一个小时
3. 具体来说，采用可学习的 adaption prompts 集合，放在 高层的 transformer layers 的文本输入 tokens 的前面，然后提出 zero-init attention 机制，可以自适应地将新的知识引入 LLaMA，同时还可以保留其预训练的知识
4. 最终可以实现和 full fine tune 的 Alpaca 7B 性能相当，同时方法还可以拓展到多模态输入

## Introduction

1. 最近的 instruction-following models 模型很成功，如 ChatGPT
2. Stanford Alpaca 提出了 LLaMA 模型使其更容易复现，Alpaca 利用GPT-3.5以 self-instruct 方式，对 LLaMA 中的整个 7B 参数进行 fine tune，得到一个性能类似于 GPT-3.5 的模型
3. 但是 full fine tune 大规模 LLaMA 仍然耗时、计算量大、不支持多模态且难以转移到不同的下游场景，于是提出 LLaMA-Adapter
4. 采用 52K instruction-output 数据用于训练，但是资源效率优于 Alpaca
5. 提出的 LLaMA-Adapter 有以下四个特点：![](image/Pasted%20image%2020230510151118.png)
	1. 只有 1.2M 的可训练参数，但是性能和 7B Alpaca 差不多
	2. 一小时的 fine tune。在8个A100 GPU上 收敛不到一个小时。
	3. 灵活且即插即用。对于不同的场景可以训练不同的 adapter
	4. 支持多模态。可以拓展到图像输入用于多模态推理

## 相关工作（略）

## LLaMA-Adapter

给定数据集包含 52K 的 instruction-to-output data 和一个预训练好的 $N$ 层 Transformer 的 LLaMA 模型， 引入可学习的 adaption prompts 集合用于指令微调。

$L$ 层的 prompts 的集合记为 $\{P\}^L_{l=1}$ ，其中 $P_l\in\mathbb{R}^{K\times C}$ 表示每层包含 $K$ 个 prompt，$C$ 为模型的特征维度。注意这里的 $L\le N$ 表示只有 top $L$ 层才有 prompt，从而只 fine tune 那些具有高阶语义的层。

以第 $l$ 层为例，设输入的原本的 word token 长为 $M$，记为 $T_l\in\mathbb{R}^{M\times C}$，然后把 adaptation prompt 和输入 tokens 进行拼接得到：$$[P_l;T_l]\in\mathbb{R}^{(K+M)\times C}$$
从而，$P_l$ 学习的指令知识可以引导 $T_l$ 生成上下文响应。

### Zero-init Attention

如果 adaptation prompts 是随机初始化的，开始训练的时候可能干扰 word tokens，从而损害fine tune的性能和稳定性。于是将最后 $L$ 层的 Transformer 的注意力机制修改为 zero-init 注意力，如下图：![](image/Pasted%20image%2020230511150500.png)

假设第 $l$ 层，模型正在基于 $[P_l;T_l]$ 生成第 $(M+1)$ 个单词，其对应的 embedding 记为 $t_l\in\mathbb{R}^{1\times C}$，计算 attention 为，首先获得query, key, value：$$\begin{aligned}
Q_l & =\operatorname{Linear}_{\mathrm{q}}\left(t_l\right) \\
K_l & =\operatorname{Linear}_{\mathrm{k}}\left(\left[P_l ; T_l ; t_l\right]\right) \\
V_l & =\operatorname{Linear}_{\mathrm{v}}\left(\left[P_l ; T_l ; t_l\right]\right)
\end{aligned}$$
然后计算 attention score（归一化之前）：$$S_l=Q_l K_l^T / \sqrt{C} \in \mathbb{R}^{1 \times(K+M+1)}$$
这个 score 是一个向量，表示 $t_l$ 和 $K+M+1$ 个 tokens 之间的相似度，同时 $S_l$ 可以写成以下两部分：$$S_l=[S_l^K;S_l^{M+1}]^T$$
其中，$S_l^K\in\mathbb{R}^{K\times 1},S_l^{M+1}\in\mathbb{R}^{(M+1)\times 1}$ 分别表示 $K$ 个 adaptation promps 和 $M+1$ 个 word tokens 的 score，前者表示prompts 对 $t_l$ 的贡献，这是在训练初期导致干扰的部分。

于是采取一个可学习的 gating factor 记为 $g_l$，用来自适应的控制 $S_l^K$ 的重要程度。$g_l$ 初始化为 0，从而可以消除欠拟合的问题，然后训练的时候逐步增加，最后的归一化的 score 计算为：$$S_l^g=\left[\operatorname{Softmax}\left(S_l^K\right) \cdot g_l ; \quad \operatorname{Softmax}\left(S_l^{M+1}\right)\right]^T$$
> 其实就是，开始训练的时候，$g_l$ 很小，此时可以认为 adaptation prompt 不起作用，随着训练的进行增加 $g_l$ 再逐步引入。

最后考虑 value 计算最终的输出：$$t_l^o=\operatorname{Linear}_{\mathrm{o}}\left(S_l^g V_l\right) \in \mathbb{R}^{1 \times C}$$

### 多模态

![](image/Pasted%20image%2020230511150847.png)
LLaMA-Adapter 可以很轻松用在多模态：
+ 首先利用预先训练的视觉编码器，例如 CLIP，提取其多尺度全局特征，记为 $\{I_m\}_{m=1}^M$，$M$ 为缩放数量，$I_m\in\mathbb{R}^{1\times C_m}$，然后把这 $M$ 个特征拼接起来，再通过一个 投影层：$I_p=\operatorname{Projection}\left(\operatorname{Concat}\left(\left\{I_m\right\}_{m=1}^M\right)\right)$，得到 $I_p\in\mathbb{R}^{1\times C}$
+ 重复 $I_p$ 向量 $K$ 次，element-wise 地加到长为 $K$ 的 adaptation prompts 上，对于第 $l$ 层，得到的多模态 prompt 记为：$P_l^v=P_l+\operatorname{Repeat}\left(I_p\right) \in \mathbb{R}^{K \times C}$
+ 最后当成正常的 LM 模型 fine tune 即可

也可以简单地拓展到音视频。

## 实验和结果

使用 Stanford Alphaca 的 Alphaca-52K 指令数据进行训练，每条数据包含：
+ instruction：任务描述
+ input：上下文
+ output：GPT3.5 的回答

8 个 A100 GPU 训练 5个 epoch，基于 7B 的 LLaMA 模型，top-p 采样。

仅通过微调120万个参数，产生的回答与完全微调的 Alpaca 和大规模 GPT-3 相当：
![](image/Pasted%20image%2020230511171906.png)
