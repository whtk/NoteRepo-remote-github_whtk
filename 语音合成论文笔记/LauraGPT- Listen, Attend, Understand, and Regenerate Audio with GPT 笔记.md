>  preprint 2023.10（ICLR2024 被拒了。。），阿里达摩院
<!-- 翻译&理解 -->
<!-- Generative Pre-trained Transformer (GPT)
models have achieved remarkable performance
on various natural language processing tasks,
and have shown great potential as back-
bones for audio-and-text large language models
(LLMs). Previous mainstream audio-and-text
LLMs use discrete audio tokens to represent
both input and output audio; however, they suf-
fer from performance degradation on tasks such
as automatic speech recognition, speech-to-text
translation, and speech enhancement over mod-
els using continuous speech features. In this
paper, we propose LauraGPT, a novel uni-
fied audio-and-text GPT-based LLM for au-
dio recognition, understanding, and generation.
LauraGPT is a versatile LLM that can process
both audio and text inputs and generate out-
puts in either modalities. We propose a novel
data representation that combines continuous
and discrete features for audio: LauraGPT en-
codes input audio into continuous representa-
tions using an audio encoder and generates out-
put audio from discrete codec codes. We pro-
pose a one-step codec vocoder to overcome
the prediction challenge caused by the multi-
modal distribution of codec tokens. We fine-
tune LauraGPT using supervised multi-task
learning. Extensive experiments show that
LauraGPT consistently achieves comparable
to superior performance compared to strong
baselines on a wide range of audio tasks re-
lated to content, semantics, paralinguistics, and
audio-signal analysis, such as automatic speech
recognition, speech-to-text translation, text-to-
speech synthesis, speech enhancement, auto-
mated audio captioning, speech emotion recog-
nition, and spoken language understanding. -->
1. 之前的 LM 使用离散的 token 作为音频表征，但是在 ASR、STT 和 speech enhancement 等任务效果不好
2. 提出 LauraGPT，一个 audio-and-text GPT-based LLM，实现 audio recognition, understanding 和 generation
3. 提出将连续和离散的特征结合的数据表示方法：使用 audio encoder 将输入音频编码为连续表征，从离散 codes 得到输出音频：
    1. 提出 one-step codec vocoder 解决 codec tokens 的多模态分布问题
4. 使用有监督的多任务学习微调 LauraGPT，实验表明，在 ASR、STT、TTS、speech enhancement、captioning、emotion recognition 和 SLU 等任务上，LauraGPT 性能优于 strong baselines

## Introduction
<!-- Audio-and-Text LLMs can be categorized into
two directions. One direction builds a collabo-
rative AI system using LLMs as controllers to
interface specialized audio models, such as ASR
and TTS models, to support various audio-and-
text tasks (Shen et al., 2023; Huang et al., 2023b).
These methods have serious drawbacks, including
high complexity, significant resource consumption,
and unavoidable error accumulation problems. The
other direction develops a unified Audio-and-Text
LLM leveraging LLMs as the backbone to support
audio-and-text tasks (Ao et al., 2022; Chen et al.,
2021b; Wang et al., 2023b; Rubenstein et al., 2023).
Decoder-only audio-and-text LLMs (Zhang et al.,
2023a; Wang et al., 2023b; Rubenstein et al., 2023)
are the dominant technique under this category.
These models convert continuous audio into dis-
crete tokens and integrate text and audio tokens into
unified vocabulary. These models suffer from in-
formation loss from quantization of speech signals
into discrete tokens, which leads to notable perfor-
mance degradation on ASR compared to models us-
ing continuous speech features (Chen et al., 2023a;
Chang et al., 2023; Yang et al., 2023c; Puvvada
et al., 2023). In this paper, we focus on improv-
ing the second category of unified Audio-and-Text
LLMs. Moreover, recent advances in audio gen-
eration from unified audio-and-text LLMs (Wang
et al., 2023a,b) discretize speech into codec codes,
then use an autoregressive language model (LM)
to predict output tokens from the first quantizer
and use a non-autoregressive model to predict to-
kens from the other quantizers individually. One
limitation of this mechanism is that it needs many
prediction steps (hence called multi-step audio
synthesis scheme) to generate good quality speech.
Another limitation is that predicting the indices
of the other codec groups is challenging due to
the multi-modal distribution nature of codec to-
kens (Jenrungrot et al., 2023). -->
1. Audio-and-Text LLMs 有两个方向：
    1. 使用 LLMs 作为 controller，对接专门的 audio models，如 ASR 和 TTS，支持 audio-and-text 任务，但是复杂度高，资源消耗大，错误累积
    2. 使用 LLMs 作为 backbone，支持 audio-and-text 任务，decoder-only audio-and-text LLMs 是主流，将连续音频转为离散 tokens，但是量化导致信息损失严重
<!-- To overcome the drawbacks of existing unified
audio-and-text LLMs, we propose LauraGPT, a
novel unified Audio-and-Text LLM based on the
GPT framework for audio recognition, understand-
ing, and generation. LauraGPT is a versatile LLM
that can process both audio and text inputs and
generate outputs in either modalities, with a single
model. We propose a novel data representation
that combines continuous and discrete features
for audio: LauraGPT encodes input audio into con-
tinuous representations using an audio encoder and
generates output audio from discrete codec codes.
This data representation improves the performance
of audio-input tasks and also facilitates joint au-
toregressive modeling of audio and text features
for audio generation tasks. -->
2. 提出 LauraGPT，基于 GPT 的 audio recognition, understanding 和 generation 的 统一 Audio-and-Text LLM：
    1. 可以同时处理文本和音频输入，生成文本和音频输出
    2. 提出一种结合连续和离散特征的数据表示方法
<!-- We also propose a one-step codec vocoder in
LauraGPT to address the two limitations of the
popular multi-step audio synthesis scheme. Our
one-step codec vocoder uses a transformer-based
predictor to estimate the sum of all codec token
groups instead of the individual indices, by min-
imizing the reconstruction losses. Our approach
simplifies the audio generation process to a single
feed-forward calculation and also overcomes the
prediction challenge caused by the multi-modal
distribution of codec tokens. -->
3. 提出 one-step codec vocoder：
    1. 使用 transformer-based predictor 估计所有 codec token groups 的和
    2. 简化音频生成过程，克服 codec tokens 的多模态分布问题
<!-- We fine-tune LauraGPT using supervised multi-
task learning on diverse audio tasks, includ-
ing tasks focusing on content, semantics, paralin-
guistics, and audio-signal analysis, such as ASR,
speech-to-text translation (S2TT), TTS, SE, auto-
mated audio captioning (AAC), speech emotion
recognition (SER), and SLU. Comprehensive ex-
periments show that, to the best of our knowl-
edge, LauraGPT1 consistently achieves com-
parable to superior performance compared to
strong baselines on the largest and the most di-
verse set of audio recognition, understanding,
and generation tasks among existing decoder-
only unified audio-and-text LLMs focusing on
these tasks (Zhang et al., 2023a; Wang et al.,
2023b; Rubenstein et al., 2023). The results are
remarkable since existing general speech models
either focus solely on speech recognition and under-
standing tasks but neglect speech generative tasks,
or support speech generation but suffer from se-
vere performance degradation on speech recogni-
tion and understanding tasks. -->
4. 使用有监督的多任务学习微调 LauraGPT，包括 ASR、S2TT、TTS、SE、AAC、SER 和 SLU 等任务 

## 相关工作（略）

## 方法
<!-- Figure 1 depicts the architecture of the proposed
LauraGPT. Section 3.1 describes the audio encoder,
the text tokenizer, and the modified GPT LM for
unified audio-and-text modeling. Section 3.2 elab-
orates the audio tokenizer. Section 3.3 introduces
an efficient one-step codec vocoder for convert-
ing audio tokens into high-quality raw waveforms.
Section 3.4 describes the multi-task fine-tuning and
shows that LauraGPT provides an extensible frame-
work for supporting more complex tasks. -->
结构如图：
![](image/Pasted%20image%2020241022163925.png)

<!-- Modified Language Model for Unifying
Audio-and-Text Modeling -->
### 修改 LM 实现 audio-and-text 建模
<!-- For audio inputs, different from other audio-and-
text LLMs using discrete tokens to represent audio
inputs, we extract the log-compressed Mel spec-
trogram features and convert them into continuous
representations using a Conformer-based audio en-
coder. Text inputs and outputs are tokenized using
the Qwen tokenizer (Bai et al., 2023), which inher-
its the tiktoken tokenizer (Jain, 2022) and incorpo-
rates additional augmentations for commonly used
characters and words in different languages. The
tokenized input text undergoes embedding matrix
transformation to generate dense vectors. The au-
dio representations and text embeddings have the
same dimension D. The Conformer-based encoder
is initialized with weights from a pre-trained ASR
model (Gao et al., 2023). Since batch normaliza-
tion can lead to endless loop decoding, we replace
it with layer normalization in the Conformer-based
encoder (details are in Appendix C.2). -->
对于音频输入，不使用离散 tokens，而是提取 log-compressed mel 谱，使用 Conformer-based audio encoder 转为连续表征。
> 文本输入和输出使用 Qwen tokenizer。

tokenized 文本经过 embedding matrix 得到 dense vectors。音频和文本的表征维度 $D$ 相同。
> Conformer-based encoder 使用预训练 ASR 模型的权重进行初始化。

<!-- To achieve audio generation capabilities, the au-
dio outputs are discretized into tokens using an
audio tokenizer (Section 3.2) to obtain discrete rep-
resentations and the softmax output layer is aug-
mented with the audio tokens. As a result, the
weight matrix W in the output layer is of size
(N + M + L) ×D and is utilized to calculate the
logits for audio and text tokens at each position,
where N , M , and L denote the vocabulary sizes
of text, audio, and task tokens, respectively. Task
tokens are used to inform the model which task
should be performed. Note that in order to con-
trol the sequence length, we perform the low frame -->
为了实现音频生成，音频输出使用 audio tokenizer 转为离散 tokens，采用 audio token 来增强 softmax 输出层。从而权重局长 $\mathbf{W}$ 的大小为 $(N+M+L)\times D$，用于计算每个位置的音频和文本 tokens 的 logits，其中 $N$、$M$ 和 $L$ 分别表示文本、音频和任务 tokens 的 vocab size。
> 任务 tokens 用于指定模型执行的任务。

<!-- Based on the aforementioned representations,
the GPT backbone is trained to model various audio
and text tasks by minimizing the cross-entropy loss: -->
GPT 模型通过最小化交叉熵损失训练来建模各种 audio 和 text 任务：
$$$$