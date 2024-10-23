>  preprint 2023.10（ICLR2024 被拒了。。），阿里达摩院

1. 之前的 LM 使用离散的 token 作为音频表征，但是在 ASR、STT 和 speech enhancement 等任务效果不好
2. 提出 LauraGPT，一个 audio-and-text GPT-based LLM，实现 audio recognition, understanding 和 generation
3. 提出将连续和离散的特征结合的数据表示方法：使用 audio encoder 将输入音频编码为连续表征，从离散 codes 得到输出音频：
    1. 提出 one-step codec vocoder 解决 codec tokens 的多模态分布问题
4. 使用有监督的多任务学习微调 LauraGPT，实验表明，在 ASR、STT、TTS、speech enhancement、captioning、emotion recognition 和 SLU 等任务上，LauraGPT 性能优于 strong baselines

> 多任务框架，两个所谓的创新点：
> 1. 输入音频改成连续表征
> 2. 一步 codec vocoder（其实就是训练一个 predictor，用第一个 quantizer 的 codes 预测所有 codes 的和）

## Introduction

1. Audio-and-Text LLMs 有两个方向：
    1. 使用 LLMs 作为 controller，对接专门的 audio models，如 ASR 和 TTS，支持 audio-and-text 任务，但是复杂度高，资源消耗大，错误累积
    2. 使用 LLMs 作为 backbone，支持 audio-and-text 任务，decoder-only audio-and-text LLMs 是主流，将连续音频转为离散 tokens，但是量化导致信息损失严重
2. 提出 LauraGPT，基于 GPT 的 audio recognition, understanding 和 generation 的 统一 Audio-and-Text LLM：
    1. 可以同时处理文本和音频输入，生成文本和音频输出
    2. 提出一种结合连续和离散特征的数据表示方法
3. 提出 one-step codec vocoder：
    1. 使用 transformer-based predictor 估计所有 codec token groups 的和
    2. 简化音频生成过程，克服 codec tokens 的多模态分布问题
4. 使用有监督的多任务学习微调 LauraGPT，包括 ASR、S2TT、TTS、SE、AAC、SER 和 SLU 等任务 

## 相关工作（略）

## 方法

结构如图：
![](image/Pasted%20image%2020241022163925.png)

### 修改 LM 实现 audio-and-text 建模

对于音频输入，不使用离散 tokens，而是提取 log-compressed mel 谱，使用 Conformer-based audio encoder 转为连续表征。
> 文本输入和输出使用 Qwen tokenizer。

tokenized 文本经过 embedding matrix 得到 dense vectors。音频和文本的表征维度 $D$ 相同。
> Conformer-based encoder 使用预训练 ASR 模型的权重进行初始化。

为了实现音频生成，音频输出使用 audio tokenizer 转为离散 tokens，采用 audio token 来增强 softmax 输出层。从而权重局长 $\mathbf{W}$ 的大小为 $(N+M+L)\times D$，用于计算每个位置的音频和文本 tokens 的 logits，其中 $N$、$M$ 和 $L$ 分别表示文本、音频和任务 tokens 的 vocab size。
> 任务 tokens 用于指定模型执行的任务。

GPT 模型通过最小化交叉熵损失训练来建模各种 audio 和 text 任务：
$$\mathcal{L}_{LM}=-\frac{1}{T_v}\sum_{j=1}^{T_v}\log p_\theta\left(\mathbf{v}_j|\mathbf{u}_{1:T_u},\mathbf{u}_{task},\mathbf{v}_{1:j-1}\right)$$
其中 $\mathbf{u}$ 表示输入 embeddings，长度为 $T_u$，$\mathbf{v}$ 表示目标 tokens，长度为 $T_v$。任务相关的 token $\mathbf{u}_{task}$ 插入在输入 embeddings 和输出 tokens 之间。只考虑输出的损失，输入和任务 token embeddings 的损失被 mask 掉。采用开源 GPT LLM Qwen 作为 backbone。所有参数一起优化。vocoder 独立训练，训练和推理阶段保持冻结。

### Audio Tokenizer

采用 codec 作为 audio tokenizer 提取离散表征。codec 模型结构与 [EnCodec- High Fidelity Neural Audio Compression 笔记](../语音领域其他论文笔记/EnCodec-%20High%20Fidelity%20Neural%20Audio%20Compression%20笔记.md) 类似，包括 convolutional recurrent encoder 和 decoder 以及 RVQ。encoder 和第一个 RVQ 作为 audio tokenizer，第一个 quantizer 的输出作为 audio tokens。

### 一步 codec vocoder

提出一步 codec vocoder 从 audio tokens 生成波形。vocoder 包括：
+ transformer-based predictor：
+ codec decoder

predictor 通过最小化预测 embeddings 和 ground truth embeddings 的 L1 和 L2 距离来估计 32 个 RVQ quantizers 的 embeddings 和：
$$\mathcal{L}_{p r e}=\sum_{t,i}^{T,D_{c}}|\mathbf{E}_{t,i}-\hat{\mathbf{E}}_{t,i}|1+|\mathbf{E}_{t,i}-\hat{\mathbf{E}}_{t,i}|_{2}$$
其中 $T$ 表示帧数，$D_c$ 表示 codec embeddings 的维度。得到预测 embeddings 后，使用预训练 codec 模型的 decoder 重构原始音频波形。

除了 LLM 预测的 audio tokens，还把 文本、音频作为 condition 输入到 predictor 中。zero-shot TTS 任务中，文本输入和 prompt 音频的特征作为 condition。SE 任务中，带噪语音特征作为条件。
> 与现有的 Text-to-Audio LLMs 不同，这里将音频生成过程简化为单个 feed-forward 的计算。

### 多任务微调

基本任务包括 ASR、SLU、S2TT、SER、AAC、SE 和 TTS。

LauraGPT 基于统一的任务框架：`[输入 embeddings, 任务 ID, 输出 tokens]`。相同的输入，不同的任务有不同的输出。例如，ASR 和 S2TT 任务对于相同的音频输入需要不同的输出。任务 tokens 包含在输入 embeddings 和输出权重矩阵中。TTS 任务使用文本 embeddings 作为输入，ASR、S2TT、SLU、SE、ACC 和 SER 任务使用音频编码作为输入。TTS 和 SE 任务使用音频 tokens 作为目标输出，其他任务使用文本 tokens 作为目标输出。

## 实验设置（略）

## 结果和分析（略）
