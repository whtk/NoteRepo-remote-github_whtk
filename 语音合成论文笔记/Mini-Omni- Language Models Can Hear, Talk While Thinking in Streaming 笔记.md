> preprint 2024，清华

1. 现有的 LM 通常依赖于 TTS 实现语音合成，导致延迟
2. 提出 Mini-Omni，端到端的音频对话模型，可以实现实时语音交互
    1. 提出 text-instructed 语音生成方法 + 推理时的 batch-parallel 策略
    2. 模型可以保留原模型的语言能力
3. 引入 VoiceAssistant-400K 数据集，微调模型以优化语音输出
4. Mini-Omni 是第一个完全端到端的开源实时语音交互模型

## Introduction
1. 语音输出的难点在于：
    1. 音频推理的复杂性：直接训练音频推理非常困难，模型输出不连贯
    2. 模型复杂性：增加语音输入输出模块会增加整体复杂度
    3. 模态对齐困难：文本推理能力难以转移到音频领域
    4. 资源需求：将模型的文本能力转移到语音模态需要将所有数据标签转换为音频并重新训练，资源消耗大
2. 提出 Mini-Omni，具有实时对话能和全端到端的语音输入输出能力：
    1. 使用现有的离散 token 方法，采用最简单的模型架构
    2. 使用 0.5B 模型和少量合成音频数据实现了直接音频推理
3. 提出 parallel generation 范式，transformer 同时生成音频和文本 token
4. 提出 "Any Model Can Talk" 方法，通过额外的 adapter 和预训练模型实现语音输出
5. 在传统 TTS 多模态任务上评估 Mini-Omni 性能，包含 textQA、ASR、TTS response 和 speechQA

## 相关工作（略）

## Mini-Omni

提出同步文本和音频生成：在生成音频 token 时，模型以对应的文本 token 为 condition。在生成音频 token 之前，填充 N 个 token 确保先生成对应的文本 token。

### Audio Language Model

给定 $Y=\left(y_{i}\in{\mathcal{V}}_{\operatorname{txt}}\mid i=1,\cdot\cdot,t_{\operatorname{xt}}\right)$ 为文本 utterance，$\mathcal{V}_{\operatorname{txt}}$ 是文本词汇表，长度为 $t_{\operatorname{txt}}$。$Y$ 的概率可以表示为 $p(Y)=\prod_{i=1}^{t_{\operatorname{txt}}}p\left(y_{i} \mid y_{1}, \cdot \cdot, y_{i-1}\right)$。将连续语音信号转换为离散语音 token，表示为 $D=\left(d_{i} \in \mathcal{V}_{\operatorname{dst}} \mid i=1, \cdot \cdot, t_{\operatorname{dst}}\right)$。$\mathcal{V}_{\operatorname{dst}}$ 是离散语音 token 词汇表。将文本和语音合并到新的词汇表 $\mathcal{V}_{\operatorname{voxt}}=\mathcal{V}_{\operatorname{txt}} \cup \mathcal{V}_{\operatorname{dst}}$，语音和文本 token 的概率表示为 $Z=\left(z_{i} \in \mathcal{V} \mid i=1, \cdot \cdot, t\right)$，$Z$ 的概率表示为 $p(Z)=\prod_{i=1}^{t} p\left(z_{i} \mid z_{1}, \cdot \cdot, z_{i-1}\right)$，$Z$ 表示离散语音 token $D(\mathcal{V}=\mathcal{V}_{\operatorname{dst}})$ 或文本 token $Y(\mathcal{V}=\mathcal{V}_{\operatorname{txt}})$ 或 $Y$ 和 $D$ 的各种组合。对于同时生成的音频和文本 token，负对数似然损失可以表示为：
$$\mathcal{L}(T,A|C)=\sum_{j=1}^{m}\sum_{i=1}^{n_{j}}\log P(T_{i,j},A_{i,j}|T_{<i,j},A_{<i,j},X_{j})$$
其中 $T,A$ 是训练语料库 $C$ 中的文本-音频 对，$m$ 是训练样本数。$X_{j}$ 是第 $j$ 个样本的输入条件，$n_{j}$ 是样本 $T_{j}$ 和 $A_{j}$ 的最大 token 数，$T_{i,j}$ 和 $A_{i,j}$ 表示第 $j$ 个样本的第 $i$ 个文本 token 和音频 token。

### 解码策略

模型结构如图：
![](image/Pasted%20image%2020241112153833.png)

text instruction 下的音频生成：Mini-Omni 通过 text-audio 并行解码方法实现流式音频输出。同时输出音频和文本 token，音频波形通过 TTS 得到。为了与大模型的输入对齐，所有并行生成的序列在生成下一个 token 之前求和。从而模型可以实现实时语音输出，最小化首个 token 的延迟。

text-delay 并行解码：音频 token codebook 包含多个 layer，同时生成所有层可以提高速度。本文使用 SNAC 作为 audio encoder，包含 7 个 token 层。因此，我们使用 8 个 sub-Language Model heads 一次生成 8 个 token（包括文本）。音频 token 基于文本合成，所以文本 token 先输出，然后是 SNAC 的 token。text-first delay 并行解码过程如图 b：
![](image/Pasted%20image%2020241112154343.png)

batch 并行解码：采用 Batch 方法，将单输入的推理扩展到 batch 为 2：
+ 其中一个样本，需要文本和音频 response
+ 另一个样本只需要文本 response，实现基于文本的音频合成

然后第一个样本的输出的文本 token 丢弃，把第二个样本的文本输出 embed 到第一个样本的对应的文本 token 位置。同时，第一个样本的音频使用第二个样本的文本 response 进行流式传输；从而可以将模型的文本能力转移到音频且没有额外资源开销。batch parallel decoding 的推理过程如上图 c。

### Any Model Can Talk

训练策略需要尽可能保留原模型的能力。
> 基础模型的性能 + 一种可以用在其他的文本输出上效果好但不能进行语音交互的模型的方法

Audio 编码：对于输入特征提取，用 Hubert 或单独预训练的 audio encoder（如 Whisper）。对于音输出，选择多 codebook 的方法，采用 delay pattern 和 text conditions 进行并行解码。

三阶段训练：
1. Modality Alignment：冻结核心模型，只训练两个 adapter，使用语音识别和语音合成数据训练模型的语音识别和合成能力
2. Adaptation Training：冻结 adapter，只训练模型的文本能力，使用语音识别、口语问答和文本回答任务数据（传统的 LM 训练方式）
3. Multi-modal Finetuning：解冻所有权重，使用 comprehensive data 进行微调

模型输入：给定 8 个并行输出序列，输入也需要 8 个序列，导致复杂性增加。模型可以接受文本或音频输入，放在对应的模态序列中。对于音频输入，通过 adapter 将输入 token 和 Whisper 特征转换为相同维度的张量，然后进行拼接。根据任务，在不同位置放置 `<answer>` token 来引导模型的输出，实现多模态输出。在输入模型之前，所有序列求和并平均。一些任务的组织如图：
![](image/Pasted%20image%2020241112163549.png)

## 实验（略）
