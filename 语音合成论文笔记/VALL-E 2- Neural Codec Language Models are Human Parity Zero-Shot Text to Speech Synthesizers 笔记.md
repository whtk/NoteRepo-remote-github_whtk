> Microsoft，2024.6


1. 提出 VALL-E 2，首次实现 human parity
2. 提出两个改进：
    1. Repetition Aware Sampling：改进了原始的 nucleus sampling 过程，考虑了解码历史中的 token 重复。可以稳定解码，避免无限循环问题
    2. Grouped Code Modeling：将 codec 的 codes 分组，可以缩短序列长度，提高推理速度，也可以解决长序列建模的挑战
3. 在 LibriSpeech 和 VCTK 数据集上，VALL-E 2 是第一个达到人类水平的系统

## Introduction

1. VALL-E 训练自回归模型生成 coarse codec codes，然后用非自回归模型生成 fine codec codes，但是有两个问题：
    1. Stability：推理时的随机采样可能导致输出不稳定，nucleus sampling 可能导致无限循环问题
    2. Efficiency：VALL-E 的自回归架构受到 off-the-shelf 音频编解码器模型的高帧率限制，推理速度较慢
2. 非自回归模型需要 frame-aligned text-speech 数据，且生成的 tokens 有预先确定的持续时间，限制了生成搜索空间，牺牲了韵律和自然度
3. VALL-E 2 使用了两个关键修改：repetition aware sampling 和 grouped code modeling
    1. repetition aware sampling：根据解码历史中的 token 重复，自适应地使用随机或 nucleus sampling，增强了解码过程的稳定性，避免了 VALL-E 中遇到的无限循环问题
    2. grouped code modeling：将 codec codes 分组，每个组在 AR 建模过程中建模为一个 frame，减少了序列长度，加速了推理，改善了性能
4. VALL-E 2 在大规模的 Libriheavy 数据集上训练，在 LibriSpeech 和 VCTK 数据集上达到了人类水平的性能（out-of-domain）


## 相关工作（略）

本文采用 [EnCodec- High Fidelity Neural Audio Compression 笔记](../语音领域其他论文笔记/EnCodec-%20High%20Fidelity%20Neural%20Audio%20Compression%20笔记.md) 来 tokenize 语音信号，采用 [Vocos- Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis 笔记](轻量化/Vocos-%20Closing%20the%20gap%20between%20time-domain%20and%20Fourier-based%20neural%20vocoders%20for%20high-quality%20audio%20synthesis%20笔记.md) 来生成波形。

## VALL-E 2

### Grouped Codec Language Modeling

将 TTS 视为条件 codec language modeling 任务，将 codec code sequence 分为一定大小的 groups，每个 group 的 codec codes 作为一个 frame 建模。这样可以摆脱之前模型的帧率约束，整数倍地减少帧率，好处：
1. 提高推理效率
2. 减少长序列建模问题

用的是 TTS 的目标函数，优化 grouped code sequence 的似然。给定音频样本 $\mathbf{y}$ 和对应的 tokenize 后的文本 $\mathbf{x}=[x_0,x_1,\ldots,x_{(L-1)}]$，其中 $L$ 是文本序列长度，首先用预训练的 codec 将 $\mathbf{y}$ 转换为 codec code sequence $\mathbf{C}^{T\times J}=[c_0,c_1,\ldots,c_{(T-1)}]$，其中 $T$ 是 code sequence 长度，$J$ 是 codec 模型中的量化器数量（这里的 $J=8$），每个 $c_t$ 表示每个时间步的 8 个 codes。然后将其分为 grouped code sequence $\mathbf{C}^G=[C_{0:G},C_{G:2G},\ldots,C_{(T-G):T}]$，其中 group size 为 $G$，$C_{0:G}$ 表示 group $[c_0,c_1,\ldots,c_{(G-1)}]$。最后，训练 VALL-E 2 模型 $\theta$，最小化给定文本序列 $\mathbf{x}$ 条件下 grouped code sequence $\mathbf{C}^G$ 的负对数似然：
$$\begin{aligned}\text{L}&=-\log p(\mathbf{C}^G|\mathbf{x};\theta)\\&=-\sum_{t=0}^{T/G-1}\log p(\mathbf{C}_{t\cdot G:(t+1)\cdot G}|\mathbf{C}_{<t\cdot G},\mathbf{x};\theta),\end{aligned}$$

其中 $C_{t\cdot G:(t+1)\cdot G}$ 是第 $t$ 个 group 的 codec codes $[c_{t\cdot G},\ldots,c_{((t+1)\cdot G-1)}]$，$C_{<t\cdot G}$ 是之前 $(t-1)$ 个 groups 中的所有 codec codes。

推理的时候，给定文本输入（同时包含 speech prompt 和要合成的文本）和 unseen speaker 的 grouped codec codes，作为条件和 prompt，模型可以生成包含对应的内容和说话人声音的 grouped codec codes。具体来说，给定文本序列 $\mathbf{x}$ 和 unseen speaker 的 speech 样本 $\mathbf{y}'$，可以得到对应的 grouped code sequence $\mathbf{C}^P=\mathbf{C}^{G<T'}=[C_{0:G},C_{G:2G},\ldots,C_{(T'-G):T'}]$。然后，在给定文本序列 $\mathbf{x}$ 和 code prompt $\mathbf{C}^P$ 的条件下，生成目标的 grouped code sequence $\mathbf{C}^T=\mathbf{C}^{G\geq T'}=[C_{T'}:(T'+G),\ldots,C_{(T-G):T}]$：
$$\begin{aligned}
\text{CT}& =\underset{\mathbf{C}}{\operatorname*{\arg\max}}p(\mathbf{C}|\mathbf{C}^P,\mathbf{x};\theta)  \\
&=\arg\max_{\mathbf{C}}\sum_{t=T^{\prime}/G}^{T/G-1}\log p(\mathbf{C}_{t\cdot G:(t+1)\cdot G}|\mathbf{C}_{<t\cdot G},\mathbf{x};\theta).
\end{aligned}$$

最后，可以使用 off-the-shelf neural codec decoder 将 code sequence $\mathbf{C}^T$ 转为波形。

### VALL-E 2 架构

VALL-E 2 也使用了 hierarchical 结构：
+ 一个 AR codec language model，以自回归方式生成每个 frame 的第一个 codec code 序列
+ 一个 NAR codec language model，以非自回归方式生成每个剩余的 code 序列

两个模型使用相同的 Transformer 架构，对于来自不同 quantizer 的 codes，使用不同的 embeddings，并且 code prediction 层的参数与 code embedding 层的参数共享。

AR 模型有一个 group embedding 层，将 code embedding 投影到 group embedding，以及一个 group prediction 层，用于预测一个 group 中的 codes。

NAR 模型有一个 code ID embedding 层，用于指定要预测的 code 序列的 ID。AR 模型和 NAR 模型有不同的 attention mask 策略：AR 模型使用 causal attention 策略，NAR 模型使用 full attention 策略。

### VALL-E 2 训练

VALL-E 2 的训练只需要简单的 utterance-wise speech-transcription pair 数据，训练过程如下：
![](image/Pasted%20image%2020240718155904.png)

具体来说，对于训练集中的每个音频和对应的文本，首先使用音频 codec encoder 和 text tokenizer 得到 codec codes $\mathbf{C}=[c_0,c_1,\ldots,c_{(T-1)}]$ 和文本序列 $\mathbf{x}=[x_0,x_1,\ldots,x_{(L-1)}]$，然后用于 AR 模型和 NAR 模型的训练。

#### 自回归模型训练

AR 模型训练预测第一个 codec code 序列 $c_{:,0}=[c_{0,0},c_{1,0},\ldots,c_{(T-1),0}]$，条件是文本序列 $\mathbf{x}$。

首先使用 text embedding matrix $\mathbf{W}_x$ 和 code embedding matrix $\mathbf{W}_c$ 得到 text embedding sequence $\mathbf{E}_x=[e_{x0},e_{x1},\ldots,e_{x(L-1)}]$ 和 code embedding sequence $\mathbf{E}_c=[e_{c0},e_{c1},\ldots,e_{c(T-1)}]$：
$$\mathbf{e}_l^x=\mathbf{W}^x\odot x_l,\\\mathbf{e}_t^c=\mathbf{W}^c\odot c_{t,0},$$

其中 $l$ 和 $t$ 分别表示文本序列和 code 序列中的索引，$\odot$ 表示索引选择。然后将 code embedding sequence 分为大小为 $G$ 的 groups，将每个 group 的 code embeddings 在 hidden dimension 上连接起来，使用 group embedding matrix $\mathbf{W}_g$ 得到 group embedding sequence $\mathbf{E}_g=[e_{g0},e_{g1},\ldots,e_{g(T/G-1)}]$：
$$\mathbf{e}_t^g=\mathbf{e}_{t\cdot G:(t+1)\cdot G}^c\cdot\mathbf{W}^g$$

将 text embedding sequence $\mathbf{E}_x$ 和 group embedding sequence $\mathbf{E}_g$ 进行 concat，插入特殊 token eos 和 bos 的 embedding：
$$\mathbf{E}^0=\mathbf{E}^x\parallel[\mathbf{e}_{<\text{eos}>},\mathbf{e}_{<\text{bos}>}]\parallel\mathbf{E}^g,$$

其中 $\parallel$ 表示在时间维度上的 concat。然后分别将可学习的 position embedding 加到 text embedding sequence 和 group embedding sequence 上。模型输入 $\mathbf{E}^0$，使用 group prediction 层和 softmax code prediction 层，预测对应的 code sequence 和末尾的特殊 token eos。由于 causal attention mask 策略，每个 code group $c_{t\cdot G:(t+1)\cdot G,0}$ 的预测只能关注文本序列 $\mathbf{x}$ 和前面的 codes $c_{<t\cdot G,0}$。

最终，AR 模型的参数 $\theta_{\text{AR}}$ 通过最小化给定文本序列 $\mathbf{x}$ 条件下第一个 code 序列 $c_{:,0}$ 的负对数似然进行优化：
$$\begin{aligned}
\mathcal{L}_{AR}& =-\log p(\mathbf{c}_{:,0}|\mathbf{x};\theta_{\mathrm{AR}})  \\
&=-\sum_{t=0}^{T/G-1}\log p(\mathbf{c}_{t\cdot G:(t+1)\cdot G,0}|\mathbf{c}_{<t\cdot G,0},\mathbf{x};\theta_{\mathrm{AR}}) \\
&=-\sum_{t=0}^{T/G-1}\sum_{t^{\prime}=t\cdot G}^{(t+1)\cdot G-1}\log p(c_{t^{\prime},0}|\mathbf{c}_{<t\cdot G,0},\mathbf{x};\theta_{\mathrm{AR}}).
\end{aligned}$$

在 VALL-E 2 的 AR 模型中，group sequence $c_{:,0}=[c_{0:G},c_{G:2G,0},\ldots,c_{(T-G):T,0}]$ 以自回归方式建模，而每个 group 中的 codec codes $c_{t\cdot G:(t+1)\cdot G,0}=[c_{t\cdot G,0},c_{(t\cdot G+1),0},\ldots,c_{((t+1)\cdot G-1),0}]$ 以非自回归方式建模。
> 其实就是批量预测。

#### 非自回归模型训练

给定 AR 模型生成的第一个 code 序列，NAR 模型训练生成剩余的 code 序列 $c_{:,j}$，条件是文本序列 $\mathbf{x}$ 和前面的 code 序列 $c_{:,<j}$，以非自回归方式进行，其中 $j\in[1,\ldots,7]$。

在推理时， prompt 的所有 8 个 code 序列是已知的，为了更好地建模 prompt 的说话人信息，在训练时显式地将所有 code 序列 $\mathbf{C}$ 分为 acoustic condition $\mathbf{C}^{<T'}$ 和 target code sequences $\mathbf{C}^{\geq T'}$，长度 $T'$ 为随机采样。模型通过预测每个 target code sequence $c^{\geq T'}_j$ 来进行优化，条件包含三个：
+ 文本序列 $\mathbf{x}$
+ acoustic condition $\mathbf{C}^{<T'}$ 下的所有 $J=8$ 个 code sequences 
+ 前面的 target code sequences $c^{\geq T'}_{<j}$
> 因为训练的时候，数据只有 语音-文本 对，所以才需要这样手动选一部分作为 prompt。

首先使用 text embedding matrix $\mathbf{W}_x$ 得到 text embedding sequence $\mathbf{E}_x=[e_{x0},e_{x1},\ldots,e_{x(L-1)}]$，然后使用 code embedding matrix $\mathbf{W}_c$ 得到 code embedding sequence $\mathbf{E}_c=[e_{c0},e_{c1},\ldots,e_{c(T-1)}]$，将 acoustic condition $\mathbf{C}^{<T'}$ 和前面的 target code sequences $\mathbf{C}^{\geq T',<j}$ 中的所有 code embeddings 求和：
$$\mathbf{e}_t^c=\begin{cases}\sum_{k=0}^7\mathbf{W}^c\odot c_{t,k},&t<T'\\\sum_{k=0}^{j-1}\mathbf{W}^c\odot c_{t,k},&t\geq T'\end{cases},$$

其中 $t$ 是时间步，$j$ 是 codec code ID。然后使用 code ID embedding matrix $\mathbf{W}_{\text{id}}$ 得到 codec code ID embedding $e_j$：
$$\mathrm{e}^j=\mathbf{W}^{id}\odot j$$

将 text embedding sequence $\mathbf{E}_x$，code embedding sequence $\mathbf{E}_c$ 和 codec code ID embedding $e_j$ 进行 concat，插入特殊 token eos 的 embedding：
$$\mathbf{E}^j=\mathbf{E}^x\parallel[\mathbf{e}_{<\text{eos}>}]\parallel\mathbf{E}^c\parallel[\mathbf{e}_{<\text{eos}>}]\parallel[\mathbf{e}^j].$$

然后分别将可学习的 position embedding 加到 text embedding sequence 和 code embedding sequence 上，类似于 AR 模型。NAR 模型输入为 $\mathbf{E}^j$，使用 code prediction 层，预测每个 codec code id $j$ 的对应 code sequence $c_{:,j}$。由于 full attention mask 策略，每个 token $c_{t,j}$ 的预测可以关注整个输入序列。

最终，NAR 模型通过最小化给定文本序列 $\mathbf{x}$，所有 acoustic condition $\mathbf{C}^{<T'}$ 的 code sequences 和前面的 $j$ 个 target code sequences $\mathbf{C}^{\geq T',<j}$ 条件下，每个第 $j$ 个 target code sequence $c^{\geq T',j}$ 的负对数似然进行优化：
$$\begin{aligned}
\mathcal{L}&_{NAR} =-\log p(\mathbf{C}_{\geq T^{\prime},\geq1}|\mathbf{x},\mathbf{C}_{<T^{\prime}},\mathbf{c}_{\geq T^{\prime},0};\theta_{\mathrm{NAR}})  \\
&=-\sum_{j=1}^7\log p(\mathbf{c}_{\geq T^{\prime},j}|\mathbf{x},\mathbf{C}_{<T^{\prime}},\mathbf{C}_{\geq T^{\prime},<j};\theta_{\mathrm{NAR}}).
\end{aligned}$$

实际上，为了在训练时优化计算效率们不会遍历所有的 $j$ 值计算训练损失并求和，而是随机选择一个 $j\in[1,\ldots,7]$，使用训练损失优化模型：
$$\mathcal{L}_{\mathrm{NAR_j}}=-\log p(\mathbf{c}_{\geq T^{\prime},j}|\mathbf{x},\mathbf{C}_{<T^{\prime}},\mathbf{C}_{\geq T^{\prime},<j};\theta_{\mathrm{NAR}}).$$

### VALL-E 2 推理

在推理时，通过 prompting 实现 zero-shot TTS 任务，如图：
![](image/Pasted%20image%2020240718164240.png)

给定文本和 unseen speaker 的 speech 及其对应的文本，首先将 speech transcription 和 text sentence 进行 concat，得到文本序列 $\mathbf{x}$ 作为文本条件。speech 样本转换为 codes $\mathbf{C}^P=\mathbf{C}^{<T'}=[c_0,c_1,\ldots,c_{(T'-1)}]$ 作为 prompt。AR 模型和 NAR 模型基于这些 prompt 生成 codes $\mathbf{C}^{\geq T'}=[c_{T'},\ldots,c_{(T-1)}]$。最后将 codes 通过 audio codec decoder 合成 personalized 语音。

#### 自回归模型推理

首先 AR 模型进行推理，生成第一个 target codes 的 code sequence $c^{\geq T'}_{0}$，条件是文本序列 $\mathbf{x}$ 和 code prompt $c^{<T'}_{0}$。使用 grouped codec language modeling 方法，将 grouped code sequence 输入 AR 模型，以自回归方式生成每个 group 的 target codes：
$$\begin{aligned}
\mathbf{c}\geq T^{\prime},0& \begin{aligned}&=\arg\max p(\mathbf{c}_{\geq T^{\prime},0}|\mathbf{x},\mathbf{c}_{<T^{\prime},0};\theta_{\mathrm{AR}})\\&\mathbf{c}_{\geq T^{\prime},0}\end{aligned}  \\
&=\arg\max_{\mathbf{c}_{\geq T^{\prime},0}}\sum_{t=T^{\prime}/G}^{T/G-1}\log p(\mathbf{c}_{t\cdot G:(t+1)\cdot G,0}|\mathbf{x},\mathbf{c}_{<t\cdot G,0};\theta_{\mathrm{AR}}) \\
&=\arg\max_{\mathbf{c}_{\geq T^{\prime},0}}\sum_{t=T^{\prime}/G}^{T/G-1}\sum_{t^{\prime}=t\cdot G}^{(t+1)\cdot G-1}\log p(c_{t^{\prime},0}|\mathbf{x},\mathbf{c}_{<t\cdot G,0};\theta_{\mathrm{AR}}).
\end{aligned}$$

与 VALL-E 中使用的随机采样方法不同，本文提出了一种 repetition aware sampling 方法，以增强 nucleus sampling 以获得更好的解码稳定性。

算法如下：
![](image/Pasted%20image%2020240718165602.png)

+ 给定 AR 模型预测的概率分布 $p(c_{t',0}|\mathbf{x},\mathbf{c}_{<t\cdot G,0};\theta_{\mathrm{AR}})$，首先使用预定义的 top-p 值 $v$ 进行 nucleus sampling 生成 target code $c_{t'}$
+ 计算窗口大小 $K$ 下 token $c_{t'}$ 在前面 code sequence 中的重复率 $r$
	+ 如果重复率 $r$ 超过预定义的重复阈值 $t_n$，则从 $p(c_{t'}|\mathbf{x},\mathbf{c}_{<t\cdot G,0};\theta_{\mathrm{AR}})$ 中随机采样替换 target code $c_{t'}$

虽然一个 group 中的 codes 是非自回归建模，但它们是自回归地预测的，所以才可以计算重复率 $r$ 并在这两种采样方法之间切换。repetition aware sampling 使得解码过程不仅受益于 nucleus sampling 的稳定性，还可以避免无限循环问题。
> repetition aware sampling 不会增加解码延迟，额外采样操作的时间几乎可以忽略不计。

#### 非自回归推理

给定第一个 target codes 的 code sequence $c_{\geq T',0}$，可以使用文本条件 $\mathbf{x}$ 和 acoustic condition $\mathbf{C}_{<T'}$ 推理 NAR 模型，生成剩余的 target codes 的 code sequences $\mathbf{C}_{\geq T',\geq1}$：
$$\begin{aligned}\mathbf{C}_{\geq T^{\prime},\geq1}&=\arg\max_{\mathbf{C}_{\geq T^{\prime},\geq1}}p(\mathbf{C}_{\geq T^{\prime},\geq1}|\mathbf{x},\mathbf{C}_{<T^{\prime}},\mathbf{c}_{\geq T^{\prime},0};\theta_{\mathbf{NAR}})\\&=\arg\max_{\mathbf{C}_{\geq T^{\prime},\geq1}}\sum_{j=1}^7\log p(\mathbf{c}_{\geq T^{\prime},j}|\mathbf{x},\mathbf{C}_{<T^{\prime}},\mathbf{C}_{\geq T^{\prime},<j};\theta_{\mathbf{NAR}}).\end{aligned}$$

为了生成第 2-8 个 code sequence，要对 NAR 模型进行七次推理，使用贪心解码方法逐个生成。将 AR 模型生成的第一个 codec codes 与 NAR 模型生成的所有 code matrix $\mathbf{C}_{\geq T'}$ 一起用于生成 target personalized 波形。

VALL-E 2 不仅可以使用未知说话人的 reference utterance 作为 prompt 生成 语音来克隆声音，还可以实现 zero-shot speech continuation，其中，使用 utterance 的完整 transcription 作为文本条件，前 3 秒的前缀作为 prompt 用于生成语音。

## 实验（略）
