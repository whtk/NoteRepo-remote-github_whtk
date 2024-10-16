> NIPS 2024，上交、微软

1. 提出 CoVoMix（Conversational Voice Mixture Generation），可以实现 zero-shot、human-like、多说话人的多轮对话语音生成
    1. 先将对话文本转换为多个离散 token 的 stream，每个 stream 为一个说话人的 semantic 信息
    2. 将 token stream 输入基于 flow-matching 的 acoustic model 生成混合 mel-spectrograms
    3. 使用 HiFi-GAN 生成语音波形
2. 实验表明，CoVoMix 可以生成自然、连贯的对话

## Introduction

1. spontaneous dialogues 生成的三个难点为：
    1. 缺乏高质量的 spontaneous conversational datasets，以及分割 paralinguistic behaviors 困难
    2. 多说话人对话中的 turn-taking 机制研究较少
    3. 传统方法中多轮对话的一致性不足
2. 提出 CoVoMix，用于多说话人对话生成，主要贡献如下：
    1. 第一个 zero-shot、human-like、多说话人对话混合语音生成，提出：
        1. 从对话文本中预测多个 stream 的 semantic token，每个 stream 代表一个说话人
        2. 基于 flow-matching 的 acoustic model 生成混合 mel-spectrogram
    2. 设计了多种对话生成评估指标，展示 CoVoMix 在生成 human-like 对话和 monologue 方面的能力
    3. 使用 Fisher dataset，包括了 monologue 和 dialogue speech 的训练和评估策略

## 相关工作（略）

## CoVoMix

zero-shot 语音合成需要模型在目标语音中合成未出现在训练数据中的语音，只需要文本和 speech prompt，通常通过 in-context learning 实现。给定 transcribed speech 数据集 $\{x, y\}$，其中 $y$ 和 $x$ 分别表示语音 utterances 和 transcriptions。zero-shot 多说话人语音合成则基于他们的 transcriptions 和 prompts 同时生成多个说话人的语音。

提出的 CoVoMix 如图：
![](image/Pasted%20image%2020241016151208.png)

包含：
+ multi-stream text-to-semantic model
+ acoustic model
+ HiFi-GAN vocoder

采用对话数据集 $D = \{x, y\}$ 进行训练，其中 $y = [y^1, y^2]$ 表示两个说话人的 stereo dialogue，$x$ 为文本并带有说话人标签。

### Multi-stream Text-to-Semantic Model

multi-stream text-to-semantic 模型基于 encoder-decoder，输入 BERT text tokenizer 得到的 text token，输出 multi-stream semantic token 序列。本文关注两个说话人 dual-stream。使用预训练的 HuBERT model 作为 speech tokenizer 提取聚类的离散 HuBERT hidden units 作为 semantic token 序列。如果对话是单通道录制的，则需要进行说话人分离得到双通道 waveform。模型使用 CE loss 训练：
$$\mathcal{L}_{t2s}=\sum_{c=1}^C\sum_i\log p(s_i^{(c)}|s_{1:i-1}^{(c)};\theta,x)$$
其中 $s_i$ 为第 $i$ 个 semantic token，$c$ 表示第 $c$ 个说话人。为了预测两个 stream 的 semantic token 序列，将 semantic embedding 分为两个不同的 segment，每个 segment 对应不同的说话人。

### Acoustic Model

acoustic model 是基于 flow-matching 的 transformer encoder，给定 multi-stream semantic token 序列和 multi-speaker prompts 生成混合 mel-spectrogram。

在每个 timestamp $t \in [0, 1]$，lookup table 首先将 semantic token 序列 $s = [s^1, s^2]$ 嵌入为 $s_{\text{emb}} = [s^1_{\text{emb}}, s^2_{\text{emb}}]$。提取相应的混合 mel-spectrogram $m$ 和每个说话人的 mel-spectrogram $[m^1, m^2]$。随机选择一个 mask，被 mask 的部分 $\tilde{m} = m \odot \text{mask}$ 作为预测部分，已知部分 $m_{\text{ctx}} = [m^1 \odot (1 - \text{mask}), m^2 \odot (1 - \text{mask})]$ 作为 prompt。

在每个 flow step $t$，采样 $w = (1 - (1 - \sigma_{\text{min}})t) \tilde{m}_0 + tm$，其中 $\sigma_{\text{min}}$ 是控制 deviation 的超参数，$\tilde{m}_0$ 从 $N(m|0, I)$ 中采样。然后，采样 $w$、acoustic prompt $m_{\text{ctx}}$ 和 semantic embedding sequences $s_{\text{emb}}$ 逐帧拼接得到输入矩阵 $W_{\text{input}}$。训练 acoustic model 学习混合 mel-spectrogram，损失函数为：
$$\mathcal{L}_{CFM}=\mathbb{E}_{t,q(m,s),p_0(m_0)}\|mask\odot((m-(1-\sigma_{min})\tilde{m_0})-v_t(w,m_{ctx},s_{emb};\theta))\|^2$$

推理时，从学习到的分布 $p_1(m | s, m_{\text{ctx}})$ 中采样混合 mel-spectrogram $m$，采样高斯噪声 $m_0$，使用 ODE solver 计算 flow $\phi_1(m_0)$，给定 $d\phi_t(m_0) / dt = v_t(w, m_{\text{ctx}}, s_{\text{emb}}; \theta)$ 和 $\phi_0(m_0) = m_0$。

使用 classifier-free guidance 方法，训练时，acoustic prompt $m_{\text{ctx}}$ 和 semantic sequences $s_{\text{emb}}$ 被丢弃。推理时，使用下面的 vector field $\tilde{v}_t(w, m_{\text{ctx}}, s_{\text{emb}}; \theta)$ 替换 $v_t(w, m_{\text{ctx}}, s_{\text{emb}}; \theta)$，其中 $\alpha$ 是控制 guidance 强度的超参数：
$$\tilde{v}_t(w,m_{ctx},s_{emb};\theta)=(1+\alpha)v_t(w,m_{ctx},s_{emb};\theta)-\alpha\tilde{v}_t(w;\theta)$$

## 实验

使用 Fisher dataset，包含 2000h 英语对话，随机划分为 train/valid/test 集，比例为 97/1/2。

对于 monologue，将长对话切分为小的 mono-channel 样本并拼接，确保最小持续时间为 10s，同时得到对应的 transcriptions、mel-spectrogram 和 semantic token sequence。对于 dialogue，将长对话切分为短的 stereo-channel 对话，确保每个处理过的对话的第一句和最后一句不与其他对话重叠。对 multi-round dialogue transcript 按照每个 utterance 的开始时间排序。相邻的同说话人 utterance 直接拼接，不同说话人 utterance 用 [spkchange] 分隔。使用 HuBERT speech tokenizer 提取每个 channel 的 semantic tokens。混合两个 channel 的音频并从混合 waveform 中提取 mel-spectrogram。

### 模型

模型包含两个 text-to-semantic model，CoSingle 和 CoMix，两个 acoustic models，VoSingle 和 VoMix。CoSingle 和 VoSingle 仅在 monologue 数据上训练，VoMix 在 dialogue 数据上训练，CoMix 在 monologue 和 dialogue 数据上训练。vanilla HiFi-GAN vocoder 在 monologue 数据上训练。

text-to-semantic model 是基于 transformer 的模型，encoder 有 4 层，decoder 有 4 层。text encoder 和 CoSingle decoder 维度为 512，CoMix decoder 维度为 1024。CoMix 为多说话人应用多个 heads 生成 semantic token sequences。acoustic model 基于 transformer encoder，有 8 层和 hidden dimension 1024。VoMix 和 VoSingle 有相同的架构，除了第一个输入 linear layer。

baseline 是基于 flow-matching 语音合成模型，使用 phoneme representation，类似于 VoiceBox。baseline 包含 acoustic model 和 duration model。baseline 的 acoustic model 与 VoSingle model 相同，但从 phoneme sequence 生成 mel-spectrogram。baseline 的 duration model 预测每个 phoneme 的持续时间，也使用 flow matching objective 训练，架构为 2 层和 hidden size 1024。

所有模型从头训练，使用验证集上表现最好的模型进行推理。使用 8 NVIDIA TESLA V100 32GB GPU 进行训练。text-to-semantic model 训练 10 epochs，batch size 48。acoustic model 和 duration model 训练 100 epochs，batch size 64。使用 Adam 优化器，学习率 1e-4。训练时的 dropping condition 概率为 puncond = 0.3，推理时的 guidance 为 $\alpha = 0.7$。

### 参数和评估（略）

## 结果和分析（略）
