> preprint 2024，清华
<!-- 翻译&理解 -->
<!-- Recent advances in language models have achieved significant progress. GPT-4o,
as a new milestone, has enabled real-time conversations with humans, demonstrat-
ing near-human natural fluency. Such human-computer interaction necessitates
models with the capability to perform reasoning directly with the audio modality
and generate output in streaming. However, this remains beyond the reach of
current academic models, as they typically depend on extra TTS systems for speech
synthesis, resulting in undesirable latency. This paper introduces the Mini-Omni,
an audio-based end-to-end conversational model, capable of real-time speech inter-
action. To achieve this capability, we propose a text-instructed speech generation
method, along with batch-parallel strategies during inference to further boost the
performance. Our method also helps to retain the original model’s language ca-
pabilities with minimal degradation, enabling other works to establish real-time
interaction capabilities. We call this training method "Any Model Can Talk". We
also introduce the VoiceAssistant-400K dataset to fine-tune models optimized for
speech output. To our best knowledge, Mini-Omni is the first fully end-to-end,
open-source model for real-time speech interaction, offering valuable potential for
future research. -->
1. 现有的 LM 通常依赖于 TTS 实现语音合成，导致延迟
2. 提出 Mini-Omni，端到端的音频对话模型，可以实现实时语音交互
    1. 提出 text-instructed 语音生成方法 + 推理时的 batch-parallel 策略
    2. 模型可以保留原模型的语言能力
3. 引入 VoiceAssistant-400K 数据集，微调模型以优化语音输出
4. Mini-Omni 是第一个完全端到端的开源实时语音交互模型

## Introduction
<!-- Enhancing models with speech output capabilities is a challenging task, primarily due to four factors:
(1) Complexity of Audio Reasoning: Our experiments indicate that direct training for audio modality
reasoning is highly challenging, often resulting in incoherent outputs from the model. (2) Model
Complexity: Incorporating additional modules for speech input and output increases the overall
complexity. (3) Difficulty in Modality Alignment: The reasoning abilities developed for text are
difficult to transfer to the audio domain. (4) Resource Demands: Adapting a model’s text capabilities
to the speech modality requires converting all data labels into audio and retraining, significantly
increasing resource consumption. -->
1. 语音输出的难点在于：
    1. 音频推理的复杂性：直接训练音频推理非常困难，模型输出不连贯
    2. 模型复杂性：增加语音输入输出模块会增加整体复杂度
    3. 模态对齐困难：文本推理能力难以转移到音频领域
    4. 资源需求：将模型的文本能力转移到语音模态需要将所有数据标签转换为音频并重新训练，资源消耗大
<!-- In this paper, we propose Mini-Omni, the first open-source multi-model large language model with
real-time conversational capabilities, featuring fully end-to-end speech input and output abilities.
It also includes various other audio-to-text functionalities such as Automatic Speech Recognition
(ASR). We adapt currently available off-the-shelf methods for discretizing speech tokens and employ
the simplest model architecture, making it easy for our model and approach to be adapted by other
researchers. Direct audio reasoning poses significant challenges; however, our approach successfully
addresses this using only a 0.5B model and a limited amount of synthesized audio data. Importantly,
our training framework achieves this without heavy reliance on extensive model capabilities or large
volumes of data. -->
2. 提出 Mini-Omni，具有实时对话能和全端到端的语音输入输出能力：
    1. 使用现有的离散 token 方法，采用最简单的模型架构
    2. 使用 0.5B 模型和少量合成音频数据实现了直接音频推理
<!-- To leverage and preserve the original capabilities of the language model, we propose a parallel genera-
tion paradigm in which the transformer simultaneously produces audio and text tokens. Subsequently,
we observed a minimal impact of the audio modality on text capabilities and further introduced
batch-based parallel generation, which significantly enhances the model’s reasoning ability during
streaming audio output. As a poinerr, we opted not to sacrifice audio quality for a simpler and lower
bitrate audio encoder, in order to reduce the complexity of audio inference in the model. However, to
ensure audio quality, we selected SNAC [Siuzdak, 2024], a music-grade encoder features 8 layers of
codebooks and processes hundreds of tokens per second. Innovatively, we applied text-instructed
delayed parallel generation to address the issue of long SNAC codebook sequences. Experiments
show that the audio output quality is on par with common TTS systems. -->
3. 提出 parallel generation 范式，transformer 同时生成音频和文本 token
<!-- We also propose a method that requires minimal training and modification of the original model,
enabling other works to rapidly develop their own speech capabilities. We refer to this approach as
"Any Model Can Talk", designed to achieve speech output using a limited amount of additional
data. The approach extend speech capabilities through additional adapters and pre-trained models,
fine-tuning with a small amount of synthesized data. This is combined with the aforementioned
parallel modeling approach to enable streaming output in the new modality while preserving the
original model’s reasoning capabilities. -->
4. 提出 "Any Model Can Talk" 方法，通过额外的 adapter 和预训练模型实现语音输出
<!-- To evaluate the capabilities of Mini-Omni, we first assessed its performance on traditional text-
to-speech multi-modal tasks, including text-based question answering (textQA), automatic speech
recognition (ASR), text-to-speech response, and speech-based question answering (speechQA). The
model demonstrated strong proficiency in these fundamental tasks. Additionally, we conduct a
series of experiments to investigate the impact on the original model’s capabilities and assess the 
effectiveness and variations of our inference method. Preliminary experiments demonstrate that batch
parallel inference preserves the model’s original capabilities. We will conduct further experiments
and provide additional details in due course.-->
5. 在传统 TTS 多模态任务上评估 Mini-Omni 性能，包含 textQA、ASR、TTS response 和 speechQA

## 相关工作（略）

## Mini-Omni
<!-- Our innovation stems from existing methods such as SpeechGPT [Zhang et al., 2023a] and Spectron
[Nachmani et al., 2023] utilize the A-T-T-A approach, which mitigates the challenges of direct audio
learning by guiding the speech generation process through text. However, generating text first and
then audio is suboptimal for real-time dialogue scenarios. To address this, we propose a novel method
for simultaneous text and audio generation. This approach hypothesizes that text outputs have higher
information density, allowing for the same response with fewer tokens. During the generation of
audio tokens, the model effectively conditions on corresponding text tokens, akin to an online TTS
system. Prior to generating audio tokens, padding with N tokens ensures that the corresponding
text tokens are produced first, allowing this to serve as a hyperparameter adjustment. Additionally,
the model can also condition on speaker and style embeddings, facilitating control over speaker
characteristics and stylistic elements. In this section, we will detail how we implement our idea step
by step -->
提出同步文本和音频生成：在生成音频 token 时，模型以对应的文本 token 为 condition。在生成音频 token 之前，填充 N 个 token 确保先生成对应的文本 token。

### Audio Language Model
<!-- Consider Y = (yi ∈Vtxt |i = 1, . . . , ttxt) as a text utterance from a vocabulary Vtxt with length ttxt.
The probability of Y can be expressed as p(Y ) = ttxt
i=1 p(yi |y1, . . . , yi−1). Now, when dealing
with a continuous speech signal, we can convert it into discrete speech tokens (dst), represented as
D = (di ∈Vdst|i = 1,···, tdst) using a tokenizer. In this context Vdst is the vocabulary of discrete
speech tokens. These discrete speech tokens can be treated as spoken language within Vdst and
modeled in a manner similar to text. We combine text and speech in a new vocabulary Vvoxt by
Vvoxt = Vtxt ∪Vdst. Therefore, we can model the probability of both speech and text tokens as Z, where
Z = (zi ∈V|i = 1,···, t). This probability is expressed as p(Z) = t
i=1 p(zi |z1,···, zi−1), Z
represent discrete speech tokens D(V= Vdst) or text tokens Y (V= Vtxt) or various combinations of
Y and D. For the audio and text tokens generated simultaneously, the negative log-likelihood loss
can be formulated as in Equation (1). -->
给定 $Y=\left(y_{i}\in{\mathcal{V}}_{\operatorname{txt}}\mid i=1,\cdot\cdot,t_{\operatorname{xt}}\right)$ 为文本 utterance，$\mathcal{V}_{\operatorname{txt}}$ 是文本词汇表，长度为 $t_{\operatorname{txt}}$。$Y$ 的概率可以表示为 $p(Y)=\prod_{i=1}^{t_{\operatorname{txt}}}p\left(y_{i} \mid y_{1}, \cdot \cdot, y_{i-1}\right)$。将连续语音信号转换为离散语音 token，表示为 $D=\left(d_{i} \in \mathcal{V}_{\operatorname{dst}} \mid i=1, \cdot \cdot, t_{\operatorname{dst}}\right)$。$\mathcal{V}_{\operatorname{dst}}$ 是离散语音 token 词汇表。将文本和语音合并到新的词汇表 $\mathcal{V}_{\operatorname{voxt}}=\mathcal{V}_{\operatorname{txt}} \cup \mathcal{V}_{\operatorname{dst}}$，语音和文本 token 的概率表示为 $Z=\left(z_{i} \in \mathcal{V} \mid i=1, \cdot \cdot, t\right)$，$Z$ 的概率表示为 $p(Z)=\prod_{i=1}^{t} p\left(z_{i} \mid z_{1}, \cdot \cdot, z_{i-1}\right)$，$Z$ 表示离散语音 token $D(\mathcal{V}=\mathcal{V}_{\operatorname{dst}})$ 或文本 token $Y(\mathcal{V}=\mathcal{V}_{\operatorname{txt}})$ 或 $Y$ 和 $D$ 的各种组合。对于同时生成的音频和文本 token，负对数似然损失可以表示为：
$$\mathcal{L}(T,A|C)=\sum_{j=1}^{m}\sum_{i=1}^{n_{j}}\log P(T_{i,j},A_{i,j}|T_{<i,j},A_{<i,j},X_{j})$$
<!-- where T , A is the text-audio output pairs in the training corpus C, and m is the number of training
examples. Xj is the input condition of j-th example, nj is max number of tokens of sample Tj and
Aj , Ti,j and Ai,j represent the i-th text token and audio token of j-th sample. -->
其中 $T,A$ 是训练语料库 $C$ 中的文本-音频 对，$m$ 是训练样本数。$X_{j}$ 是第 $j$ 个样本的输入条件，$n_{j}$ 是样本 $T_{j}$ 和 $A_{j}$ 的最大 token 数，$T_{i,j}$ 和 $A_{i,j}$ 表示第 $j$ 个样本的第 $i$ 个文本 token 和音频 token。

### 解码策略
<!-- Audio Generation with text instruction. Language models have undergone substantial advance-
ments, demonstrating exceptional reasoning capabilities within the text modality. In response,
Mini-Omni has been restructured to transfer these reasoning abilities to streaming audio output
through a text-audio parallel decoding approach. This method simultaneously outputs both audio and
text tokens, with the audio generated via text-to-speech synthesis, ensuring real-time delivery while
leveraging the text-based reasoning strengths. To align with the inputs of large models, all sequences
generated in parallel are summed before producing the next token, as illustrated in Figure 1. This
approach enables the model to achieve real-time voice output in chat scenarios with minimal first
token delay. -->
模型结构如图：
![](image/Pasted%20image%2020241112153833.png)

text instruction 下的音频生成：Mini-Omni 通过 text-audio 并行解码方法实现流式音频输出。同时输出音频和文本 token，音频波形通过 TTS 得到。为了与大模型的输入对齐，所有并行生成的序列在生成下一个 token 之前求和。从而模型可以实现实时语音输出，最小化首个 token 的延迟。
<!-- Text-delay Parallel Decoding. Parallel generation was first introduced by MusicGen [Copet et al.,
2024] to accelerate the music generation process, and we have integrated this approach into the
text modality to enhance reasoning capabilities. Parallel decoding is feasible because audio token
codebooks used in language model training typically consist of multiple layers; generating all layers
simultaneously can significantly increase model speed. For real-time speech output models, parallel
decoding is even more critical, allowing for the generation of hundreds of audio tokens per second
on standard devices. In this paper, we employ SNAC as the audio encoder, which comprises seven
token layers with complementary relationships. Therefore, we employ eight sub-Language Model
heads to generate eight tokens, including text, in a single step, while maintaining a one-step delay
between adjacent layers. Since audio tokens are derived from text synthesis, the text token is output
first, followed by SNAC tokens from the first to the seventh layer. The process of text-first delay
parallel decoding we propose is illustrated in Figure 2(b). -->
text-delay 并行解码：音频 token codebook 包含多个 layer，同时生成所有层可以提高速度。本文使用 SNAC 作为 audio encoder，包含 7 个 token 层。因此，我们使用 8 个 sub-Language Model heads 一次生成 8 个 token（包括文本）。音频 token 基于文本合成，所以文本 token 先输出，然后是 SNAC 的 token。text-first delay 并行解码过程如图 b：
![](image/Pasted%20image%2020241112154343.png)

<!-- Batch Parallel Decoding. Although the previously introduced parallel generation method effectively
transfers reasoning capabilities from the text modality to the audio modality, our experiments reveal
that the model’s reasoning performance still varies between text and audio tasks, with audio responses
tending to be simpler. We hypothesize that this is due to limitations in model capacity or insufficient
audio data. To address this issue and further enhance the model’s reasoning capabilities during
dialogue, maximizing the transfer of its text-based abilities, we experimentally employ a Batch
approach. Given the model’s stronger performance in the text modality, we expand the inference task
for a single input to a batch size of 2: one sample requires both text and audio responses, as described
earlier, while the other sample only requires a text response, focusing on text-based audio synthesis.
However, the text token output from the first sample is discarded, and the text output from the second
sample is embedded into the corresponding text token positions of the first sample. Simultaneously,
the audio from the first sample is streamed using the content from the text-only response of the second
sample; we term this process batch parallel decoding. Through this method, we effectively and almost
entirely transfer the model’s text-based capabilities to the audio modality with minimal resource
overhead, significantly enhancing its reasoning abilities in the new modality. The inference process of
batch parallel decoding is illustrated in Figure 2(c). We believe batch parallel decoding represents a
key algorithmic innovation that enables such a small model to exhibit strong conversational abilities. -->
batch 并行解码：采用 Batch 方法，将单输入的推理扩展到 batch 为 2：
+ 其中一个样本，需要文本和音频 response
+ 另一个样本只需要文本 response，实现基于文本的音频合成

然后第一个样本的输出的文本 token 丢弃，把第二个样本的文本输出 embed 到第一个样本的对应的文本 token 位置。同时，第一个样本的音频使用第二个样本的文本 response 进行流式传输；从而可以将模型的文本能力转移到音频且没有额外资源开销。batch parallel decoding 的推理过程如上图 c。

### Any Model Can Talk
<!-- In this section, we present our training methodology. Our approach is designed to preserve the
capabilities of the original model as much as possible. This is achieved firstly due to the strong
performance of our base model, and secondly because our method can be applied to other works that
excel in text output but lack robust speech interaction capabilities. -->
训练策略需要尽可能保留原模型的能力。
> 基础模型的性能 + 一种可以用在其他的文本输出上效果好但不能进行语音交互的模型的方法

<!-- Audio Encoding: The audio input primarily focuses on feature extraction from the input audio,
with options including Hubert or a separately pretrained audio encoder. Given our focus on speech
input, Whisper [Radford et al., 2023] and Qwen2-audio [Chu et al., 2024] also demonstrate effective
performance for general audio tasks. For audio output, selecting audio tokens with a multi-codebook
approach better captures audio details. We experimented with flattening for audio token modeling,
but it resulted in excessively long tokens, which are detrimental to streaming and lead to unstable
learning. Instead, parallel decoding, inspired by MusicGen [Copet et al., 2024], employs a delay
pattern combined with text conditions, as illustrated in Figure 2. -->
Audio 编码：对于输入特征提取，用 Hubert 或单独预训练的 audio encoder（如 Whisper）。对于音输出，选择多 codebook 的方法，采用 delay pattern 和 text conditions 进行并行解码。
<!-- Three-Stage Training. Our training methodology is divided into three distinct stages: (1) Modality
Alignment. The goal of this stage is to enhance the text model’s ability to understand and generate
speech. The core model of Mini-Omni is entirely frozen, with gradients allowed only in two adapters.
During this stage, we use data from speech recognition and speech synthesis to train the model’s
speech recognition and synthesis capabilities. (2) Adaption Training. Once the new modality is
aligned with the text model’s input, the adapters are frozen. In this stage, we focus solely on training
the model’s text capabilities when given audio inputs, as audio output is simply synthesized from
text. The model is trained using data from speech recognition, spoken question answering, and text
response tasks. (3) Multi-modal Finetuning. In the final stage, the entire model is fine-tuned using
comprehensive data. At this point, all model weights are unfrozen and trained. Since the primary
modality alignment tasks are handled during adapter training, the original model’s capabilities are
maximally preserved. -->
三阶段训练：
1. Modality Alignment：冻结核心模型，只训练两个 adapter，使用语音识别和语音合成数据训练模型的语音识别和合成能力
2. Adaptation Training：冻结 adapter，只训练模型的文本能力，使用语音识别、口语问答和文本回答任务数据（传统的 LM 训练方式）
3. Multi-modal Finetuning：解冻所有权重，使用 comprehensive data 进行微调
<!-- Model Input Ids. Given the eight parallel output sequences, the input also requires eight sequences,
leading to significant complexity. Therefore, we briefly outline the organization of model inputs here.
The model can accept either text or audio inputs, which are placed in the corresponding modality
sequences. For audio inputs, the input tokens and Whisper features are transformed into tensors of the
same dimension via adapters and then concatenated. Depending on the task, we place the <answer>
special token in different positions to guide the model’s output, achieving multi-modal output. The
organization of some tasks is illustrated in Figure 4. Before being fed into the model, all sequences
are summed and averaged to integrate features. -->
模型输入：给定 8 个并行输出序列，输入也需要 8 个序列，导致复杂性增加。模型可以接受文本或音频输入，放在对应的模态序列中。对于音频输入，通过 adapter 将输入 token 和 Whisper 特征转换为相同维度的张量，然后进行拼接。根据任务，在不同位置放置 `<answer>` token 来引导模型的输出，实现多模态输出。在输入模型之前，所有序列求和并平均。一些任务的组织如图：
![](image/Pasted%20image%2020241112163549.png)

## 实验（略）
