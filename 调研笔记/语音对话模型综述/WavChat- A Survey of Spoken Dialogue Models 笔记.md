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

