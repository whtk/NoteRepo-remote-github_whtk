> preprint 2024.10，开源非营利性组织 Kyutai
<!-- 翻译&理解 -->
<!-- We introduce Moshi, a speech-text foundation model and full-duplex spoken dialogue frame-
work. Current systems for spoken dialogue rely on pipelines of independent components,
namely voice activity detection, speech recognition, textual dialogue and text-to-speech.
Such frameworks cannot emulate the experience of real conversations. First, their complex-
ity induces a latency of several seconds between interactions. Second, text being the inter-
mediate modality for dialogue, non-linguistic information that modifies meaning— such as
emotion or non-speech sounds— is lost in the interaction. Finally, they rely on a segmenta-
tion into speaker turns, which does not take into account overlapping speech, interruptions
and interjections. Moshi solves these independent issues altogether by casting spoken dia-
logue as speech-to-speech generation. Starting from a text language model backbone, Moshi
generates speech as tokens from the residual quantizer of a neural audio codec, while model-
ing separately its own speech and that of the user into parallel streams. This allows for the
removal of explicit speaker turns, and the modeling of arbitrary conversational dynamics.
We moreover extend the hierarchical semantic-to-acoustic token generation of previous work
to first predict time-aligned text tokens as a prefix to audio tokens. Not only this “Inner
Monologue” method significantly improves the linguistic quality of generated speech, but we
also illustrate how it can provide streaming speech recognition and text-to-speech. Our re-
sulting model is the first real-time full-duplex spoken large language model, with a theoret-
ical latency of 160ms, 200ms in practice, and is available at github.com/kyutai-labs/moshi. -->
1. 提出 Moshi，是一个 speech-text foundation model，实现全双工的语音对话
2. 现有的语音对话需要各种 pipeline，包括 VAD、ASR、文本对话和 TTS
3. Moshi 将语音对话看成是 speech-to-speech 生成，其从文本的 LM 出发，生成 speech tokens，同时并行建模自己的语音和用户的语音
4. 进一步将 hierarchical semantic-to-acoustic token 生成扩展到首先预测时间对齐的文本 token 作为音频 token 的 prefix


## Introduction
<!-- Yet, the experience offered by these interfaces remains far from natural conversations.
First, latency compounds along the many components of these pipelines, resulting in a
typical global latency of several seconds. This is unlike natural conversations which demon-
strate response times of a few hundred milliseconds. Second, as language understanding
and generation happens in the textual domain, any non-written information is ignored
by the model. This goes from paralinguistic information, such as emotion and accent, to
non-speech audio, such as surrounding acoustic events. Finally, these models remain fun-
damentally turn-based, assuming that dialogue is a sequence of well-defined single-speaker
segments. While this paradigm is suited to text dialogue, it falls short in modeling aspects
of spoken conversations such as interruptions, overlapping speech— which amounts for 10 to
20% of spoken time (C¸ etin and Shriberg, 2006) —and backchanneling (i.e. non-interrupting
interjections such as “OK” or “I see”). -->
1. 现有的对话系统还是不如自然对话：
    1. pipeline 太复杂，延迟很高
    2. 语言理解和生成是在 textual domain，忽略了 non-written 信息
    3. 模型基于 turn-based，其假定对话是一系列明确定义的单说话人片段，无法处理打断、重叠的语音和 backchanneling（非打断的插话）
<!-- In this work we introduce Moshi, a speech-text foundation model and real-time spoken
dialogue system that aims at solving the aforementioned limitations: latency, textual infor-
mation bottleneck and turn-based modeling. Moshi augments a text LLM backbone with
a smaller audio language model (Borsos et al., 2022; Yang et al., 2023) that ingests and
predicts discrete audio units. This removes the information bottleneck of text by under-
standing inputs and generating outputs directly in the audio domain, while benefiting from
the knowledge and reasoning abilities of the underlying text LLM. We extend previous work
on audio language models and design a streaming, hierarchical architecture, with a theo-
retical latency of 160 ms—lower than the 230 ms average in natural conversations measured
over 10 languages (Stivers et al., 2009). We furthermore introduce the first multi-stream
audio language model, i.e. a model that explicitly processes the input and output audio
streams jointly into two autoregressive token streams. This altogether removes the concept
of speaker turn and thus allows training the model on natural conversations with arbitrary
dynamics including overlap and interruptions. Our resulting model is the first full-duplex—
it always listens and always generates sound, either speech or silence—real-time conversa-
tional LLM. We summarize our contributions below: -->
2. 提出 Moshi，是一个 speech-text foundation model 和实时语音对话系统，采用一个更小的 audio LM 来增强 text LLM，直接在 audio domain 处理输入和输出，同时利用 text LLM 的知识和推理能力；设计了一个 streaming、hierarchical 架构；引入了第一个 multi-stream audio LM，显式地将输入和输出音频流联合成两个自回归 token stream 来消除 speaker turn 的概念，从而允许训练模型在具有任意动态的自然对话上，包括重叠和打断；最后得到的模型是第一个全双工的实时对话 LLM；贡献如下：
<!-- We present Helium, a 7B-parameter text LLM that we pretrain on 2.1T tokens of
public English data. Section 3.2 describes the architecture and training of the model,
while Section 4.1 provides details on the pretraining data collection and filtering -->
    + 提出 Helium，一个 7B 参数的 text LLM，使用 2.1T tokens 数据预训练
<!-- We train Mimi, a neural audio codec (Zeghidour et al., 2022; D´efossez et al., 2023) that
converts audio into the discrete tokens predicted by Moshi and back, using residual
vector quantization (RVQ). Audio language models typically combine such acoustic to-
kens with semantic tokens from a self-supervised speech model as it is necessary to pro-
duce intelligible speech in absence of text conditioning (Borsos et al., 2022). We rather
extend the approach of Zhang et al. (2024b) by distilling semantic information into the
first level of acoustic tokens and introduce improved training tricks. Section 3.3 de-
scribes the architecture and training of Mimi while Section 5.2 details ablation studies. -->
    + 训练 Mimi codec，，使用 RVQ 将 audio 转换为 Moshi 预测的离散 token；将语义信息蒸馏到 acoustic tokens 的第一级，并引入改进的训练技巧
<!-- We propose Moshi, a new architecture for audio language modeling, which combines
Helium with a smaller Transformer (Vaswani et al., 2017) model to predict audio to-
kens in a hierarchical and streaming fashion. We show how challenging it is for such
unconditioned audio language models to generate intelligible speech, and we pro-
vide solutions that outperform the intelligibility and audio quality of non-streaming
models while generating audio in a streaming fashion. We furthermore extend this ar-
chitecture to model several audio streams in parallel, allowing for a conceptually and
practically simple handling of full-duplex dialogues with arbitrary dynamics. Section
3.4 describes this architecture. -->
    + 提出 Moshi，audio LM 架构，将 Helium 与一个较小的 Transformer 模型结合起来，以 hierarchical 和 streaming 的方式预测 audio tokens；将其扩展到并行模拟多个 audio stream，实现全双工对话
<!-- In Section 3.4.4, we introduce Inner Monologue, a new training and inference setup
for audio language models that significantly improves the factuality and linguistic
quality of generated speech by predicting time-aligned text tokens before audio to-
kens. Moshi is a speech-to-speech model as it allows reasoning about non-linguistic
information, both from the user audio and from Moshi’s audio. Yet, this is not in-
compatible with Moshi producing text along its speech output. Based on the past
observation (Borsos et al., 2022; Zhang et al., 2024b) that coarse-to-fine generation
(from semantic to acoustic tokens) is critical to generating consistent speech, we ex-
tend this hierarchy to using text tokens as a per-timestep prefix to the semantic token.
Our experiments show that not only this drastically improves the length and quality
of generated speech, but we also show how forcing a delay between text and audio
tokens allows deriving streaming ASR and streaming TTS from a Moshi model. -->
    + 引入 Inner Monologue，训练和推理方式，通过在 audio tokens 之前预测时间对齐的文本 tokens，提高生成语音的真实性和语言质量
<!-- We evaluate all components of Moshi along several axes, including text understanding,
speech intelligibility and consistency, audio quality and spoken question answering.
Our experiments, reported in Section 5, show that our model is state of the art among
existing speech-text models for speech modeling and spoken question answering while
being streaming compatible and able to model several minutes of context (5 min in
our experiments). -->

## 相关工作（略）

## 模型

### 概览
<!-- Moshi is a multi-stream speech-to-speech Transformer model, which allows for full-duplex
spoken dialogue with a user thanks to an innovative architecture summarized in Figure 1.
Moshi is built on top of Helium, a text LLM which we build from scratch (Section 3.2),
relying on high-quality text data to provide strong reasoning abilities to the model. We also
propose Inner Monologue (Section 3.4.4), a training and inference procedure in which we
jointly model text and audio tokens. This allows the model to fully exploit the knowledge
imparted from the text modality, while remaining a speech-to-speech system. To enable
real-time dialogue, we also design Moshi as a multi-stream architecture from the get-go
(Section 3.4.3): The model is able to both speak and listen to the user at the same time,
and does not need to explicitly model speaker turns. In addition, to capture the input user
audio and output Moshi’s voice with high quality and in an efficient manner, we propose
Mimi (Section 3.3), a neural audio codec combining semantic and acoustic information
into a single tokenizer by using residual vector quantization and knowledge distillation. To
jointly model the audio streams from Moshi and the user, as well as Moshi’s text tokens, we
rely on a Depth Transformer compatible with streaming inference (Sections 3.4.1, 3.4.2). -->
架构如图：
![](image/Pasted%20image%2020241019104516.png)

基于 Helium LLM，采用 Inner Monologue 训练和推理方式，使得模型可以利用文本模态的知识，同时实现 speech-to-speech；设计为 multi-stream 架构，可以同时说话和听取用户，无需显式建模 speaker turns；提出 Mimi codec，通过 RVQ 和 knowledge distillation 将语义和 acoustic 信息合并为单一 tokenizer；为了联合模拟 Moshi 和用户的 audio streams，以及 Moshi 的 text tokens，使用 Depth Transformer 来支持 streaming 推理。

### Helium Text LLM
<!-- Helium is an autoregressive language model, based on the Transformer architecture (Vaswani
et al., 2017). Following previous work in this area, we make the following changes to the
original architecture: First, we use RMS normalization (Zhang and Sennrich, 2019) at the
input of the attention blocks, the feed-forward blocks and the output linear layer of the
model. We use rotation positional embeddings (Su et al., 2024, RoPE), a context length
of 4,096 tokens and FlashAttention (Dao et al., 2022) for efficient training. Finally, we
change the architecture of the feed-forward blocks and use Gated Linear Units (Shazeer,
2020), with the SiLU activation as a gating function (Hendrycks and Gimpel, 2016b). Our
tokenizer is based on the unigram model from SentencePiece (Kudo and Richardson, 2018),
and contains 32,000 elements mostly targeting English. We split all numbers into single
digits, and use byte-backoff to ensure that our tokenizer does not lose information. We
train the model with the AdamW (Loshchilov and Hutter, 2017) optimizer, with a fixed
learning rate followed by a cosine learning rate decay (Loshchilov and Hutter, 2016). -->
Helium 是一个基于 Transformer 架构的自回归语言模型，对原始架构进行了如下修改：
+ 采用 RMS normalization
+ 采用 RoPE 位置编码
+ 采用 FlashAttention 实现高效训练
+ 将 feed-forward blocks 架构改为 Gated Linear Units，使用 SiLU 作为门控函数

Tokenizer 基于 SentencePiece 的 unigram 模型，包含 32,000 个元素。使用 AdamW 优化器，固定学习率，然后使用余弦学习率衰减。
<!-- Training data is one of the critical ingredients to train LLMs: we now describe our method to
obtain a large and high-quality text dataset. We start from high-quality data sources, such
as Wikipedia, Stack Exchange and a large collection of scientific articles. As the quantity
of data from these sources is too small to train a LLM, we also rely on web crawled data,
specifically from CommonCrawl, to extend our dataset. See more details on data sources in
Section 4.1. Web data requires extensive processing to obtain a high-quality training set:
we perform deduplication, language identification and quality filtering. In the following, we
describe each operation in more details. -->
从高质量数据源（如 Wikipedia、Stack Exchange 和大量科学文章）开始，再用 CommonCrawl 爬取数据。数据预处理包括去重、语种识别和质量过滤。
<!-- Deduplication. We start from the WET files, which contain only the text content of web-
pages, which was extracted by the CommonCrawl project. Because this format contains
all the text of a page, it includes a lot of boilerplate such as navigation menus. Thus, the
first step of our pipeline is to deduplicate each shard (there is 100 shards per crawl) at the
line level, to remove this boilerplate. To do so, we compute the FNV-1a6 hash of each line,
and use a bloom filter to remove duplicates. We also train a fastText (Joulin et al., 2016)
classifier on duplicates vs. non-duplicates, to perform fuzzy deduplication: here we only
remove blocks of at least 3 consecutive lines that are classified as duplicates. -->
去重：从 WET 文件开始，只包含 CommonCrawl 提取的网页文本。其包括很多 boilerplate，是对每个 shard（每次爬取有 100 个 shard）进行去重，去除这些 boilerplate。
<!-- Language identification. Once deduplication is performed, we apply a language identi-
fier based on fastText to keep English data only. Language identification is performed at
the document level, and we only keep documents above a certain threshold (0.85). -->
语种识别：在去重后，使用 fastText 进行语种识别，只保留英文数据。
<!-- Quality filtering. The last step is to filter the remaining data, to keep high-quality web-
pages only. To perform this step, we train a fastText classifier on lines from our high quality
data sources and from random CommonCrawl webpages. We obtain a classifier with 9 cat-
egories, corresponding to our different high quality sources such as Wikipedia or Wikibooks
and to subsets of StackExchange such as STEM or humanities. The motivation is to obtain
a finer control over which documents to keep, not only based on similarity to high quality
sources, but also based on their domains. This classifier is applied at the line level, and
an aggregated score is obtained by computing the average scores of each line, weighted by
their length. Again, we keep documents corresponding to scores above a certain threshold. -->
质量过滤：训练 fastText 分类器，对高质量数据源和随机 CommonCrawl 网页的行进行分类。分类器有 9 个类别，对应不同的高质量来源，如 Wikipedia 或 Wikibooks，以及 StackExchange 的子集。计算每行的平均分数（按长度加权）得到聚合分数，保留分数高于某个阈值的文档。

### Audio Tokenization
<!-- To discretize waveforms into audio tokens, we introduce Mimi, a neural audio codec (Zeghi-
dour et al., 2022; D´efossez et al., 2023) that operates as an autoencoder with a discrete
bottleneck (van den Oord et al., 2017). In the literature, and following the terminology de-
fined by Borsos et al. (2022), these tokens are referred to as acoustic tokens, as they model
fine audio details and are optimized for high-quality reconstruction. While these acous-
tic tokens provide appropriate targets for conditioned text-to-audio models (e.g. text-to-
speech (Wang et al., 2023) or text-to-music (Copet et al., 2023)), unconditioned speech gen-
eration requires combining them with semantic tokens extracted from self-supervised speech
models (Baevski et al., 2020; Hsu et al., 2021; Chung et al., 2021). Unlike their acoustic
counterpart, semantic tokens do not allow for reconstructing high-quality audio but correlate
strongly with linguistic content. This similarity with language allows generating intelligible
and consistent speech, even without text conditioning, by using semantic audio tokens as a
prefix to predicting acoustic tokens. Yet, this hybrid tokenization approach is not compati-
ble with real-time generation. Semantic tokens are typically not causal and can thus only be
computed in an offline manner. Moreover, generating acoustic and semantic tokens with sep-
arate encoders represents a non-negligible computational burden. Consequently, and taking
inspiration from previous work on SpeechTokenizer (Zhang et al., 2024b), Mimi uses distil-
lation to transfer non-causal, high-level semantic information into the tokens produced by
a causal model, allowing for streaming encoding and decoding of semantic-acoustic tokens -->
Mini codec 是一个带有离散 bottleneck 的自编码器，将 waveform 离散化为 audio tokens（acoustic tokens）semantic tokens 与 acoustic tokens 不同，不可以重构音频，但与内容相关。从而可以在没有文本条件是，使用 semantic audio tokens 作为 acoustic tokens 的 prefix 来生成可理解的语音。然而，这种混合 tokenization 方法不适用于实时生成。semantic tokens 只能离线计算。而且用不同的 encoder 生成 acoustic 和 semantic tokens 增加了计算量。因此，Mimi 使用 distillation 将非因果的语义信息转移到 causal model 生成的 tokens 中，实现流式编解码。
<!-- Our baseline architecture takes inspiration from SoundStream (Zeghidour et al., 2022) and
Encodec (D´efossez et al., 2023) and consists of a SeaNet (Tagliasacchi et al., 2020) autoen-
coder and a Residual Vector Quantizer (Zeghidour et al., 2022). The encoder projects a
single-channel waveform x ∈RL to a latent representation enc(x) ∈RS×D by cascading
residual convolutional blocks that interleave dilated (van den Oord et al., 2016) and strided
convolutions along with ELU (Clevert et al., 2016) non-linearities and Weight Normaliza-
tion (Salimans and Kingma, 2016). All convolutions are causal, such that this autoencoder
can run in a streaming fashion. With 4 convolutional blocks and respective striding fac-
tors (4,5,6,8), and a final 1D convolution with stride 2, Mimi’s encoder projects a 24kHz
waveform to a latent representation of 12.5 frames per second and dimension D = 512.
Symmetrically, the decoder adopts a similar structure but with transposed convolutions
rather than strided ones, to project the latent representation back to 24kHz audio. We
discretize the latent space with a Residual Vector Quantizer (Zeghidour et al., 2022), which
iteratively applies vector quantization (VQ) to the residuals of the previous quantizer. With
Q quantizers, each with a codebook of NA centroids, the RVQ discretizes the latent space
into {1,...,NA}S×Q. As a baseline, we train this model with a combination of reconstruc-
tion and adversarial losses, following the setup of Encodec (D´efossez et al., 2023). We detail
below the main changes of Mimi with respect to this default configuration. -->