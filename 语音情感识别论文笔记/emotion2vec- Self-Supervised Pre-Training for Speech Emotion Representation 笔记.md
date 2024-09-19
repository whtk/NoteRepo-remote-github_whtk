> ACL 2024 Findings，上交、复旦、港中文、阿里

1. 提出 emotion2vec，通用的语音情感表征模型：
    1. 通过 self-supervised online distillation 在 unlabeled emotion 数据上进行预训练，结合了 utterance-level loss 和 frame-level loss；
    2. 在 IEMOCAP 上，只训练 speech emotion recognition 任务的线性层，emotion2vec 超越了现有的预训练通用模型
2. emotion2vec 是第一个在各种情感任务中的通用表征模型

## Introduction

1. 传统的提取语音情感表征的方法使用 FBanks 或 MFCCs 作为特征，但缺乏语义信息，导致在情感任务上性能有限；SSL 效果很好
2. 但 SSL 模型不是完全适用于情感任务，需要通用的语音情感表征模型
3. 提出 emotion2vec，通用的情感表征模型，通过自监督预训练在 262 小时情感数据上训练，结合了 utterance-level loss 和 frame-level loss；在 IEMOCAP 上，emotion2vec 的线性模型超越了所有主流 SSL 模型和最新的专家模型；

## Related Work（略）

## 方法

emotion2vec 的核心是，采用 Utterance-level Loss 和 Frame-level Loss，用 Online Distillation 训练模型。

### 模型 Pipeline

如图：
![](image/Pasted%20image%2020240919103718.png)

emotion2vec 包含两个网络：
+ teacher network $\mathcal{T}$
+ student network $\mathcal{S}$

两个模型架构相同，都包含：
+ 特征提取器 $\mathcal{F}$：多层卷积神经网络
+ backbone 网络 $\mathcal{B}$：多层 Transformer

给定原始音频 utterance $X = [x_1, \cdots, x_{N_x}]$，Teacher 和 Student 分别使用特征提取器 $\mathcal{F}^\mathcal{T}$ 和 $\mathcal{F}^\mathcal{S}$ 得到下采样特征 $Z_0 = [z_1, \cdots, z_{N_z}]$：
$$\begin{aligned}Z_0^{\mathcal{T}}=\mathcal{F}^{\mathcal{T}}(X),\\Z_0^{\mathcal{S}}=\mathcal{F}^{\mathcal{S}}(X).\end{aligned}$$

对于 Teacher 网络，下采样特征 $Z_0^{\mathcal{T}}$ 直接输入 backbone 网络 $\mathcal{B}^{\mathcal{T}}$；对于 Student 网络，下采样特征 $Z_0^{\mathcal{S}}$ 以概率 $p$ 来mask $l$ 个连续帧，然后在输入 backbone 网络 $\mathcal{B}^{\mathcal{S}}$ 之前，加入可学习的 utterance embedding $U = [u_1, \cdots, u_{N_u}]$：
$$\begin{gathered}Z_i^{\mathcal{T}}=\mathcal{B}_i^{\mathcal{T}}(Z_{i-1}^{\mathcal{T}}),\\Y^{\mathcal{T}}=\frac1k\sum_{i=n-k+1}^nZ_i^{\mathcal{T}},\\ U^{\mathcal{S}};Y^{\mathcal{S}}=\mathcal{B}^{\mathcal{S}}(U;Mask(Z_0^{\mathcal{S}})), \end{gathered}$$
其中 $Y^{\mathcal{T}}$ 是 $n$ 个 Transformer Block $\mathcal{B}_i^{\mathcal{T}}$ 的前 $k$ 个输出的平均值；$Y^{\mathcal{S}}$ 和 $U^{\mathcal{S}}$ 是 Student backbone 网络的输出；`Mask` 是 mask 操作。
> $U$ 是通过 concatenation 引入的。

### Utterance-level Loss

Utterance-level loss 用于学习全局情感，使用均方误差（MSE）计算损失：
$$L_{Utt}=(\bar{Y}^{\mathcal{T}}-\bar{U}^{\mathcal{S}})^2,$$
其中：
$$\bar{Y}^{\mathcal{T}}=\frac1{N_z}\sum_{i=1}^{N_z}Y_i^{\mathcal{T}},\\\bar{U}^{\mathcal{S}}=\frac1{N_u}\sum_{i=1}^{N_u}U_i^{\mathcal{S}},$$

通过对 $Y^{\mathcal{T}}$ 和 $U^{\mathcal{S}}$ 沿时间轴进行 pooling 计算 utterance-level loss；提出了三种计算方式：token embedding、chunk embedding 和 global embedding。，如图：
![](image/Pasted%20image%2020240919120819.png)

Token Embedding：使用单个 token 表示学生网络 $\mathcal{S}$ 编码的全局情感信息，即，在 $U=[u_1, \cdots, u_{N_u}]$ 中设置 $N_u=1$。

Chunk Embedding：使用多个 token 表示全局情感信息，可以在 chunk 内聚合更多的全局信息。

Global Embedding：不添加额外的 utterance token，使用 frame-level output embedding $Y^{\mathcal{S}}$ 沿时间轴进行 pooling 计算 loss。

### Frame-level Loss

Frame-level loss 用于学习上下文情感，只在 mask 的部分计算 loss，损失为：
$$L_{Frm}=\frac1M\sum_{i\in\mathbb{M}}(Y_i^{\mathcal{T}}-Y_i^{\mathcal{S}})^2,$$
其中 $\mathbb{M}$ 是 mask 的 frame-level output embedding $Y^{\mathcal{S}}$ 的索引序列，$M$ 是 mask 的 token 总数。

### Online Distillation

Online distillation 是 teacher-student 模型的自监督学习策略，其中 student 网络通过反向传播更新参数，teacher 网络通过指数移动平均（EMA）更新参数。对于 student 网络 $\mathcal{S}$，总损失 $L$ 为 frame-level loss $L_{Frm}$ 和 utterance-level loss $L_{Utt}$ 的组合：
$$L=L_{Frm}+\alpha L_{Utt},$$
其中 $\alpha$ 是可调权重。对于 teacher 网络 $\mathcal{T}$，参数 $\theta_0^{\mathcal{T}}$ 初始化为 student 网络 $\theta_0^{\mathcal{S}}$ 的参数，然后在每个 mini-batch 中通过 EMA 更新参数：
$$\theta_{t+1}^\mathcal{T}=\tau\theta_t^\mathcal{T}+(1-\tau)\theta_{t+1}^\mathcal{S}.$$
其中 $\tau$ 线性增加。每个 mini-batch 中 teacher 特征提取器 $\mathcal{F}^{\mathcal{T}}$ 的参数直接从 $\mathcal{F}^{\mathcal{S}}$ 复制，而 teacher backbone 网络 $\mathcal{B}^{\mathcal{T}}$ 的参数通过 EMA 从 $\mathcal{B}^{\mathcal{T}}$ 和 $\mathcal{B}^{\mathcal{S}}$ 更新。
> 也就是说，EMA 更新的只有 backbone 网络，特征提取器共享参数。

## 实验设置

采用 data2vec and data2vec 2.0，特征提取器 $\mathcal{F}$ 是 7 层 1-D 卷积神经网络。输入音频 $X$ 采样率为 16000 Hz，输出 $Z$ 为 50 Hz，维度 512。然后进行线性投影，维度从 512 到 768，然后 mask 操作构建 backbone 网络输入。

预训练阶段，用 262 小时的 unlabeled emotion 数据训练 emotion2vec：
+ 使用 4 NVIDIA A10 Tensor Core GPUs，模拟 16 GPUs（梯度累计）
+ 使用动态 batchsize，最大 token 数量为 $1\times10^6$。
+ 化策略使用 Adam，学习率 $7.5\times10^{-5}$，权重衰减 $1\times10^{-2}$
+ 使用 cosine lr scheduler，线性 warm-up 占 5%
+ Student 模型每个时间步以 $p=0.5$ 开始的后续 $l=5$ 个 step 被 mask，超参数 $\alpha=1$
+ Teacher 模型使用前 $k=8$ 个 transformer 层输出的平均值作为训练目标。$\tau$ 从 $0.999$ 线性增加到 $0.99999$。

监督微调阶段：所有下游任务的模型架构设计尽可能简单：
+ 对于非序列任务，使用两个线性层，中间夹有 ReLU 激活函数
+ 对于序列任务，使用两层 GRU 进行预测。

数据集：18 个情感数据集，10 种不同语言：9 种英语，1 种中文、孟加拉语、法语、德语、希腊语、意大利语、波斯语、俄语和乌尔都语，所有数据统一处理为 16k Hz 的单通道

## 结果

结果如图：
![](image/Pasted%20image%2020240919144428.png)

结论：在 IEMOCAP 数据集上，emotion2vec 超越了所有现有的 SSL 预训练模型，包括大模型。

在其他数据集上的效果：
![](image/Pasted%20image%2020240919144527.png)
结论：在其他主流英语数据集上，emotion2vec 也表现优异：
+ MELD 是噪声数据集，用于测试模型在复杂环境中的 SER 性能
+ RAVDESS 和 SAVEE 是 out-of-domain 数据集

在其他语种上的效果：
![](image/Pasted%20image%2020240919144923.png)
结论：emotion2vec 在各种语言的 SER 数据集上表现优异。
