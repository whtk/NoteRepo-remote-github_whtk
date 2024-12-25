> preprint 2024.11，浙大、微软、阿里、腾讯优图
<!-- Recent advancements in spoken dialogue models, exemplified by systems like
GPT-4o, have captured significant attention in the speech domain. In the broader
context of multimodal models, the speech modality offers a direct interface for
human-computer interaction, enabling direct communication between AI and users.
Compared to traditional three-tier cascaded spoken dialogue models that comprise
speech recognition (ASR), large language models (LLMs), and text-to-speech
(TTS), modern spoken dialogue models exhibit greater intelligence. These ad-
vanced spoken dialogue models not only comprehend audio, music, and other
speech-related features, but also capture stylistic and timbral characteristics in
speech. Moreover, they erate high-quality, multi-turn speech responses with low
latency, enabling real-time interaction through simultaneous listening and speaking
capability. Despite the progress in spoken dialogue systems, there is a lack of
comprehensive surveys that systematically organize and analyze these systems and
the underlying technologies. To address this, we have first compiled existing
spoken dialogue systems in the chronological order and categorized them
into the cascaded and end-to-end paradigms. We then provide an in-depth
overview of the core technologies in spoken dialogue models, covering aspects
such as speech representation, training paradigm, streaming, duplex, and
interaction capabilities. Each section discusses the limitations of these technolo-
gies and outlines considerations for future research. Additionally, we present a
thorough review of relevant datasets, evaluation metrics, and benchmarks from
the perspectives of training and evaluating spoken dialogue systems. We hope this
survey will contribute to advancing both academic research and industrial applica-
tions in the field of spoken dialogue systems. The related material is available at
https://github.com/jishengpeng/WavChat. -->
1. 本文以时间顺序编排了现有的 SDM，分为级联和端到端两种范式
2. 给出了 SDM 的核心技术综述，包括语音表征、训练范式、流式、双工和交互能力
3. 详细阐述相关的数据集、评估指标和 benchmarks

## Introduction
<!-- Spoken dialogue models [44, 243, 224] represent one of the most direct methods of human-computer
interaction, evolving from traditional voice assistants such as Alexa3, Siri4, and Google Assistant5
to the latest intelligent dialogue systems, such as GPT-4o6. The fundamental definition of a spoken
dialogue model refers to a dialogue system capable of generating intelligent verbal responses based on
the input speech. On the one hand, the speech modality serves as both the input and output interface
for the human-computer interaction in the spoken dialogue models. On the other hand, the dialogue
system [52] requires the model to possess a certain level of textual intelligence, including the ability
to comprehend the knowledge of human society and generating professional and intelligent responses.
Recently, intelligent spoken dialogue systems, exemplified by GPT-4o and Moshi [44], have garnered
significant attention for their ability to extend speech intelligence capabilities beyond traditional
text-based dialogue models [85]. These dialogue models can not only generate natural, human-
like speech responses [44, 196] but also demonstrate an advanced understanding and generation of
acoustic features beyond text, such as timbre, emotion, and style [128, 129, 228]. Additionally, they
exhibit strong performance in processing other speech-related representations, including music and
audio events [33, 34, 67, 199]. Their realistic conversational interactivity [61, 224] and low-latency
dialogue experiences [44] further distinguish them among the traditional spoken dialogue models. -->
1. SDM 指：能够根据输入语音生成智能语音回复的对话系统，语音模态既是输入又是输出接口，对话系统需要具备一定的文本智能
<!-- The history of spoken dialogue models can be traced back to early systems like dGSLM [158] and
AudioGPT [85], leading up to more recent advancements such as GPT-4o and Moshi [44]. During
this period, many notable spoken dialogue models have emerged. As shown in Figure 1, we have
organized these models in chronological order. Broadly, they can be categorized into two types:
cascaded spoken dialogue models [33, 34] and end-to-end [150, 223, 247, 249] spoken dialogue
models. Given that most current spoken dialogue models rely on alignment with the text modality,
the distinction between cascaded and end-to-end models is crucial. As illustrated in Figure 2,
we classify all spoken dialogue models based on whether the core language model can directly
understand and generate speech representations, dividing them into cascaded and end-to-end
categories. Traditional cascaded spoken dialogue systems such as AudioGPT [85] are structured
around text as the central intermediary, typically comprising three cascaded modules. First, the
input audio is transcribed into text by an automatic speech recognition (ASR) module [170]. The
transcribed text is then fed into a large language model (LLM) such as ChatGPT to generate a textual
response. Finally, this textual response is converted back into audio through a text-to-speech (TTS)
module [110, 177]. While this cascaded architecture leverages the strong in-context capabilities of
large language models, it introduces several challenges, including high latency, limited interactivity,
and the inability to process non-textual information. To address these issues, recent research has
taken two primary directions. Some approaches [34, 199] focus on optimizing the understanding and
generation components within the cascaded system to mitigate the aforementioned limitations. Some
other approach [223, 224, 245, 249] seek to directly solve these problems by adopting end-to-end
architectures for spoken dialogue systems. Although end-to-end spoken dialogue models exhibit
various differences in terms of representations and model architectures, they share a common feature:
they do not rely on text as the central intermediary. Instead, these models aim to directly comprehend
and generate speech representations. We define such systems as end-to-end spoken dialogue models. -->
2. 以时间顺序编排 SDM 如下图：
![](image/Pasted%20image%2020241221155246.png)
3. 基于模型是否直接理解和生成语音表示，分为级联和端到端两种类型，如图：
![](image/Pasted%20image%2020241221160451.png)
    + 级联 SDM 以文本为中心，包括 ASR、LLM 和 TTS
        + 限制：高延迟、有限交互、无法处理非文本信息
    + 端到端 SDM 直接理解和生成语音表征，不依赖文本表征
<!-- When constructing spoken dialogue systems, we identify four core technologies closely related to
spoken dialogue models, based on the different levels of intelligence involved. The first is the design
of speech representations (i.e., tokenizers and detokenizers). The second concerns the paradigm
for training, inference, and generation, specifically how to align the speech modality with the text
modality while preserving or enhancing the intelligence of existing text-based dialogue models.
This part also involves selecting different model architectures, generation strategies, and multi-stage
training approaches. The third challenge involves the design of interactive, duplex, streaming for
spoken dialogue systems. Lastly, the fourth challenge relates to data—specifically, how to construct
training datasets for spoken dialogue systems and evaluate their performance. -->
4. SDM 的四个核心技术：
    1. 语音表征设计（tokenizer 和 detokenizer）
    2. 训练、推理和生成范式，如何将语音模态与文本模态对齐
    3. 交互、双工、流式设计
    4. 数据集构建和性能评估
<!-- Given these considerations, in the following sections of this paper, we address these four key
technologies in the order outlined above. In Section 2, we provide an overview of spoken dialogue
systems, including typical spoken dialogue scenarios (i.e., how to define a spoken dialogue model)
and recent developments in the cascaded and end-to-end spoken dialogue models. Section 3 focuses
on the speech representations used in spoken dialogue systems. In Section 4, we systematically
discuss the training paradigms, with particular emphasis on how to align the speech modality with the
text modality, as well as multi-stage training strategies, model architectures, and generation strategies.
Section 5 highlights the unique characteristics of spoken dialogue systems, particularly their duplex,
streaming nature, which distinguishes them from text-based dialogue systems. In Section 6, we
examine the construction of training datasets and the evaluation methodologies specific to spoken
dialogue models. At the end of each section, we include a summary and discussion to reflect on the
key insights. Finally, in Section 7, we conclude the survey by summarizing the major findings and
discussing open issues for future research. Given the complexity of the technical points, we provide
an overview of the structure of this survey in Figure 3. -->

## 概述
<!-- In this section, we will provide an overall overview of spoken dialogue models. we begin by defining
what constitutes an intelligent spoken dialogue model by examining various dialogue scenarios. We
then provide a comprehensive overview of spoken dialogue models, distinguishing between cascaded
spoken dialogue models and end-to-end spoken dialogue models. -->
<!-- Functions of Spoken Dialogue Systems -->
### SDM 的功能
<!-- Based on the demos and inference interfaces of representative models such as GPT-4o, Moshi [44],
Qwen2-Audio [33], and VITA [61], we categorize the usage scenarios of modern intelligent spoken
dialogue models into the following nine representative categories: 1) Text Intelligence, 2) Speech
Intelligence, 3) Audio and Music Generation, 4) Audio and Music Understanding, 5) Multilingual
Capability, 6) Context Learning, 7) Interaction Capability, 8) Streaming Latency, and 9) Multimodal
Capability. For the nine distinct use cases in spoken dialogue models, we provide corresponding
examples for each scenario in Figure 4. It is clear from these usage scenarios that a spoken dialogue
model is not simply an extension of a text-based dialogue model to the speech modality (i.e., where
the speech modality serves merely as an interface for converting speech into text). Rather, an
intelligent spoken dialogue system must be capable of comprehending and generating acoustic
information embedded in speech (such as timbre, style, and emotion) and of understanding and
producing a wider range of audio representations, including information related to audio events
and music. Additionally, unlike non-streaming text-based systems, spoken dialogue models need to
support real-time, interactive streaming capabilities. These usage scenarios not only highlight the
intelligence inherent in spoken dialogue systems but also present significant challenges for building
end-to-end spoken dialogue models. Below, we provide a detailed examination of each of the nine
usage scenarios. -->
SDM 的使用场景有：
+ 文本智能
+ 语音智能
+ 音频和音乐生成
+ 音频和音乐理解
+ 多语言能力
+ 上下文学习
+ 交互能力
+ 流式延迟
+ 多模态能力

每个场景的例子如下图：
![](image/Pasted%20image%2020241224154455.png)

#### 文本智能
<!-- As illustrated in Figure 4 (a), a spoken dialogue system must retain the fundamental capabilities of
the original text-based dialogue models, such as ChatGPT. We define this usage scenario as textual
intelligence. In this context, the spoken dialogue model can intelligently respond to user requests,
generating appropriate responses such as travel itineraries, work plans, and scheduling. However,
due to the limitations of voice-based interaction, the textual intelligence of current spoken dialogue
systems is more focused on the daily scenarios. In certain contexts, such as complex mathematical
theorem reasoning, the performance requirements for spoken dialogue models differ from those of
text-based dialogue models [201]. These advanced aspects of textual intelligence warrant further
exploration in unified multimodal dialogue models. --> 
SDM 可以智能地回应用户请求，生成适当的回复。

<!-- Speech Intelligence -->
#### 语音智能
<!-- A distinguishing feature of spoken dialogue models, compared to text-based dialogue models [201],
is their ability to understand and generate acoustic information beyond mere textual content. In the
speech modality, not only is the textual content present, but also additional acoustic information,
such as timbre (speaker identity) and style (emotion, prosody, etc.). As illustrated in Figure 4 (b), an
intelligent spoken dialogue system should be capable of understanding the timbre and style
of conversational speech and, ideally, generating responses with specified timbre and style in a
zero-shot manner. -->
SDM 能够理解和生成超出文本内容的声学信息，如音色和风格。首先 SDM 要可以理解对话语音的音色和风格，然后以 zero-shot 方式 生成具有指定音色和风格的回复：
<!-- This capability about speech intelligence involves several use cases. First, on the comprehension
side, the spoken dialogue system should generate responses based on the speaker’s vocal style. For
example, in the E-chat [228], a classic example might be: if a user asks, "My phone won’t turn on,
what should I do?" in a cheerful tone, the system might respond, "It looks like you’re excited about
getting a new phone. What type of phone are you interested in?" Conversely, if the user asks the
same question in a sad tone, the system might reply, "It’s unfortunate your phone isn’t working. If
you’re familiar with the repair policy, let’s proceed with the next steps." This situation indicates that
the spoken dialogue system may generate responses with different content based on varying acoustic
information. Furthermore, the system should comprehend various acoustic cues, such as accents or
emotional states, and adjust its responses of different acoustic information accordingly. For instance,
if the speaker is an American, the system might reply with a native English accent, whereas if the
speaker is a Shanghainese user, the system could respond using the corresponding dialect. Similarly,
if the user speaks with a sad tone, the dialogue system should be able to generate a more encouraging
and empathetic response. -->
1. 根据说话者的声音风格（如说话者的语气、口音、情感状态等）生成回复
<!-- On the generation side, speech intelligence is more prominently reflected in its controllability, such
as voice cloning and style control. For example, the system could be instructed to mimic a specific
voice or respond in a designated style (e.g., mimicking a grandmother’s soft and gentle voice for
a comforting interaction). Additionally, the system could use a voice prompt provided during the
conversation to fully clone the timbre from the prompt and generate speech in that same voice. In
summary, the ability to comprehend and generate acoustic information is one of the key characteristics
of an intelligent spoken dialogue model. -->
2. 具有可控性，实现声音克隆和风格控制
<!-- Audio and Music Generation -->
#### 音频和音乐生成
<!-- In the spoken dialogue models, beyond basic spoken dialogue capabilities, an intelligent spoken
dialogue system may be required to generate music and audio. For example, a user might instruct the
system to generate a one-minute piano piece or a ten-second recording of a dog barking. Additionally,
users might provide lyrics and a musical melody, asking the spoken dialogue model to create a pop
song. The system should thus inherit the generative capabilities of large-scale music [2, 40, 117, 142]
and audio [84, 135, 137] models on the output side. -->
SDM 可以生成音乐和音频，如用户要求系统生成一分钟的钢琴曲或十秒的狗叫声录音，或者提供歌词和音乐旋律，要求系统创作流行歌曲。
<!-- Audio and Music Understanding -->
#### 音频和音乐理解
<!-- Complementing its music and audio generation capabilities, a spoken dialogue model should also
be able to understand music and audio on the input side [33, 199]. For instance, when given an
audio clip, the intelligent system should identify both its content and acoustic characteristics, such
as recognizing whether the sound is a bird chirping or a cat meowing, or whether the music is calm
or energetic. Moreover, the system could extend its understanding by creating literary works—like
poetry or songs—based on the given music or audio. -->
SDM 要能够理解音乐和音频，如识别音频内容和声学特征，如识别鸟叫声或猫叫声，或识别音乐是平静还是充满活力。
<!-- Multilingual Capability -->
#### 多语言能力
<!-- Similar to text-based dialogue models, spoken dialogue systems are expected to possess multilingual
capabilities. Specifically, these models should be able to perform multilingual content translation,
such as translating a spoken segment in Japanese into French speech clips, effectively inheriting
the capabilities of simultaneous interpretation. In addition to multilingual content translation, the
system should also handle multilingual acoustic information. This means that the intelligent spoken
dialogue model should be able to generate responses in various languages and accents, replying in
the corresponding accent of the target language based on the different input speech. -->
SDM 要具备多语言能力，如多语言内容翻译和多语言声学信息处理。
<!-- Context Learning -->
#### 上下文学习
<!-- In the spoken dialogue models, the ability to handle long-form and multi-turn conversations is a key
benchmark for evaluating performance [44]. This requires that spoken dialogue models not only
support long-duration audio inputs but also generate extended audio outputs. Moreover, they must
be capable of engaging in multi-turn conversations based on historical context. An important aspect
of multi-turn dialogue is the ability to revise previous responses based on new user instructions. As
shown in Figure 4 (f), an intelligent spoken dialogue model should be able to continuously modify its
previous replies according to the user’s evolving requests. -->
SDM 要能够处理长对话和多轮对话，支持长时间音频输入和生成扩展音频输出，基于历史上下文进行多轮对话。
<!-- Interaction Capability -->
#### 交互能力
<!-- A distinguishing feature of spoken dialogue systems compared to the text-based dialogue models
is their duplex and interactive nature [44]. In text-based dialogue, interactions typically follow a
half-duplex structure, where the response can only be provided after the question has been completed,
and the user is unable to interrupt the reply in real-time. However, in the spoken dialogue systems,
full-duplex interaction is common. This means that a conversation does not need to be fully completed
before a response can be generated. Both the system and the user can interrupt and interact in real time.
For example, if the user is unsatisfied with the system’s response, they can immediately interrupt,
causing the system to halt its current generation and respond to the new input. Additionally, to emulate
more natural conversational settings, the system can also interrupt the user when appropriate, such as
when clarifying the user’s intent. Beyond the ability to interrupt, interactive dialogue often includes
the use of conversational fillers, such as "okay," "haha," or "oh," which signal acknowledgment or
agreement. Including these within spoken dialogue models enhances the realism and natural flow of
conversations. The underlying requirement for interaction capabilities is that the system should be
able to listen and speak simultaneously, responding dynamically to the flow of the interaction. -->
SDM 具有双工和交互性，支持全双工交互，系统和用户可以实时中断和交互，系统可以在适当时中断用户，增强对话的真实感和自然流畅。这要求模型可以同时听和说，动态响应交互。
<!-- Streaming Latency -->
#### 流式延迟
<!-- Streaming comprehension and generation are also fundamental functionalities of spoken dialogue
models [224, 249, 57]. In the real-world scenarios, a model cannot wait until an entire minute-long
audio segment has been processed before generating a response. Instead, the model must operate on
a chunk-based mechanism, dynamically processing and generating audio in real time, one chunk at a
time. Additionally, the streaming requirement means that the entire system must operate in a causal
manner—understanding and generating audio based solely on past information, without relying on
future information. Streaming function is often closely tied to the need for low latency. In practical
conversational experiences, the latency of the first token generated by the spoken dialogue model
(i.e., the wait time for the user) and the average latency of the generation process are critical factors
that influence the overall responsiveness and usability of the spoken dialogue system. -->
SDM 具有流式理解和生成功能，模型以块为单位实时处理和生成音频，操作方式是因果的，只依赖过去信息，不依赖未来信息。
<!-- Multimodal Capability -->
#### 多模态能力
<!-- Multimodal dialogue capability [25, 61] represents an advanced feature of spoken dialogue models.
In existing systems, this typically refers to the ability to process inputs from multiple modalities,
such as video, images, and text, while generating intelligent speech responses. A spoken dialogue
model equipped with this capability achieves the ability to “hear, see, and speak” simultaneously.
Multimodal inputs significantly enhance the potential of these systems; for instance, users can employ
various gestures to improve the quality of the model’s generated responses, and the system can
develop a deeper understanding of the physical world. Beyond multimodal inputs, the future of
dialogue systems lies in large multimodal models that unify the comprehension and generation
capabilities across all modalities, with spoken dialogue serving as the foundational modality -->
SDM 具有多模态对话能力，可以处理多模态输入，如视频、图像和文本，同时生成智能语音回复。

<!-- Cascaded Spoken Dialogue Systems -->
### 级联 SDM
<!-- The earliest prototype of cascaded spoken dialogue systems can be traced back to AudioGPT [85].
To achieve speech-to-speech dialogue functionality, the system first employed an Automatic Speech
Recognition (ASR) model to convert speech into text, followed by ChatGPT for text-based dialogue,
and finally, a Text-to-Speech (TTS) model to convert the generated text back into speech. In this
primitive version, speech was used solely as an input-output interface, retaining only the most basic
textual intelligence. For example, in the Huggingface’s open-source Speech-To-Speech framework7,
an additional Voice Activity Detection (VAD) module8 was further layered onto the traditional
cascaded modules to distinguish between speech and silent segments, as well as between different
speakers -->
最早的级联 SDM 原型可以追溯到 AudioGPT，实现语音对话功能：首先使用 ASR 模型将语音转换为文本，然后使用 ChatGPT 进行文本对话，最后使用 TTS 模型将生成的文本转换回语音。这里语音仅用作输入输出接口，保留了最基本的文本智能。
<!-- After the basic textual intelligence had been established in the cascaded spoken dialogue models,
researchers began incorporating paralinguistic features, such as emotion and style, to enhance the
speech intelligence in the cascaded spoken dialogue models. For instance, ParalinGPT [129] and
E-chat [228] integrate conversational context, speech embeddings, and paralinguistic attributes into
an autoregressive model via a sliding window, allowing the model to generate more accurate text
responses by combining historical text and emotional representations. Similarly, Spoken-LLM [128]
introduces an Emotion2Vec [144] module to provide style vectors to the Llama2-Chat model. Through
LoRA [80] fine-tuning, Llama2-Chat is trained not only to generate content-based text responses but
also to produce text responses with specific stylistic attributes (e.g., <cheerful, fast, normal>), which
can guide downstream TTS systems in generating expressive speech. -->
后续在级联 SDM 中加入了语音智能，如情感和风格，以增强语音智能。例如，ParalinGPT 和 E-chat 将对话上下文、语音嵌入和语音属性整合到自回归模型中，通过滑动窗口，结合历史文本和情感表示生成更准确的文本回复。类似地，Spoken-LLM 引入 Emotion2Vec 模块为 Llama2-Chat 模型提供风格向量。通过 LoRA 微调，Llama2-Chat 不仅训练生成基于内容的文本回复，还生成具有特定风格属性的文本回复，指导下游 TTS 系统生成富有表现力的语音。
<!-- In addition to understanding acoustic information within cascaded spoken dialogue models, there have
been efforts to directly input speech representations while retaining text as the output modality [41,
34, 112]. This forces cascaded spoken dialogue systems to process input speech directly. A common
approach involves integrating frozen speech encoders (such as Whisper [170]) with trainable encoder
adapters, allowing the speech input to be interpreted as a specialized form of text by the large language
model. By extending the vocabulary of the text-based dialogue model, the large language model
can process speech as if it were a unique form of text, enabling the generation of appropriate text
responses in the cascaded spoken dialogue models. -->
除了在级联 SDM 中理解语音信息，还有直接输入语音表征并保留文本作为输出模态的方法，从而使级联 SDM 直接处理输入语音。一种常见方法是集成 frozen speech encoders（如 Whisper）和可训练的 encoder adapters，将语音输入解释为大型语言模型的特殊文本形式。通过扩展文本对话模型的词汇表，LLM 可以以一种独特的文本形式处理语音，从而在级联 SDM 中生成适当的文本回复。