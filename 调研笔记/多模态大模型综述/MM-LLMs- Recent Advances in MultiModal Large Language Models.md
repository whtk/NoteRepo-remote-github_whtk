> preprint 2024.05，腾讯 AI Lab
<!-- In the past year, MultiModal Large Language
Models (MM-LLMs) have undergone substan-
tial advancements, augmenting off-the-shelf
LLMs to support MM inputs or outputs via
cost-effective training strategies. The resulting
models not only preserve the inherent reason-
ing and decision-making capabilities of LLMs
but also empower a diverse range of MM tasks.
In this paper, we provide a comprehensive sur-
vey aimed at facilitating further research of
MM-LLMs. Initially, we outline general de-
sign formulations for model architecture and
training pipeline. Subsequently, we introduce a
taxonomy encompassing 126 MM-LLMs, each
characterized by its specific formulations. Fur-
thermore, we review the performance of se-
lected MM-LLMs on mainstream benchmarks
and summarize key training recipes to enhance
the potency of MM-LLMs. Finally, we explore
promising directions for MM-LLMs while con-
currently maintaining a real-time tracking web-
site1 for the latest developments in the field. We
hope that this survey contributes to the ongoing
advancement of the MM-LLMs domain. -->
1. 本文给出多模态大语言模型（MM-LLM）的综述，给出了通用的模型结构和训练流程
2. 提出了一个包含 126 个 MM-LLM 的分类法
3. 回顾 MM-LLM 在主流 benchmark 上的表现
4. 总结 提升 MM-LLM 效果的训练方法

## Introduction
<!-- MultiModal (MM) pre-training research has wit-
nessed significant advancements in recent years,
consistently pushing the performance boundaries
across a spectrum of downstream tasks (Li et al.,
2020; Akbari et al., 2021; Fang et al., 2021; Yan
et al., 2021; Li et al., 2021; Radford et al., 2021; Li
et al., 2022; Zellers et al., 2022; Zeng et al., 2022b;
Yang et al., 2022; Wang et al., 2022a,b). How-
ever, as the scale of models and datasets continues
to expand, traditional MM models incur substan-
tial computational costs, particularly when trained 
from scratch. Recognizing that MM research op-
erates at the intersection of various modalities, a
logical approach is to capitalize on readily avail-
able pre-trained unimodal foundation models, with
a special emphasis on powerful Large Language
Models (LLMs) (OpenAI, 2022). This strategy
aims to mitigate computational expenses and en-
hance the efficacy of MM pre-training, leading to
the emergence of a novel field: MM-LLMs.-->
<!-- MM-LLMs harness LLMs as the cognitive pow-
erhouse to empower various MM tasks. LLMs
contribute desirable properties like robust language
generation, zero-shot transfer capabilities, and
In-Context Learning (ICL). Concurrently, foun-
dation models in other modalities provide high-
quality representations. Considering foundation
models from different modalities are individually
pre-trained, the core challenge facing MM-LLMs
is how to effectively connect LLMs with models
in other modalities to enable collaborative infer-
ence. The predominant focus within this field has
been on refining alignment between modalities and
aligning with human intent via a MM Pre-Training
(PT) + MM Instruction-Tuning (IT) pipeline. -->
1. MM-LLM 使用 LLMs 作为 concept powerhouse 来支持多模态任务
2. 不同模态的 foundation models 可以得到高质量的表征，MM-LLM 的核心挑战是如何有效地连接 LLMs 和其他模态的模型以实现协作推理
3. MM-LLM 的主要关注点是通过 MM Pre-Training (PT) + MM Instruction-Tuning (IT) pipeline 来提高模态之间的对齐和与人类意图的对齐
<!-- With the debut of GPT-4(Vision) (OpenAI, 2023)
and Gemini (Team et al., 2023), showcasing im-
pressive MM understanding and generation ca-
pabilities, a research fervor on MM-LLMs has
been sparked. Initial research primarily focuses
on MM content comprehension and text genera-
tion, encompassing tasks such as image-text under-
standing, exemplified by projects like BLIP-2 (Li
et al., 2023e), LLaVA (Liu et al., 2023e), MiniGPT-
4 (Zhu et al., 2023a), and OpenFlamingo (Awadalla
et al., 2023); video-text understanding, as demon-
strated by initiatives such as VideoChat (Li et al.,
2023f), Video-ChatGPT (Maaz et al., 2023), and
LLaMA-VID (Li et al., 2023j); and audio-text
understanding, as seen in projects like Qwen-
Audio (Chu et al., 2023b). Later, the capabili-
ties of MM-LLMs have been expanded to sup-
port specific modality outputs. This includes tasks
with image-text output, such as GILL (Koh et al.,
2023a), Kosmos-2 (Peng et al., 2023), Emu (Sun
et al., 2024), and MiniGPT-5 (Zheng et al., 2023b);
as well as speech/audio-text output, exemplified
by projects like SpeechGPT (Zhang et al., 2023a)
and AudioPaLM (Rubenstein et al., 2023). Recent
research endeavors have focused on mimicking
human-like any-to-any modality conversion, shed-
ding light on the path to artificial general intelli-
gence. Some efforts aim to amalgamate LLMs with
external tools to reach an approaching any-to-any
MM comprehension and generation, such as Visual-
ChatGPT (Wu et al., 2023a), HuggingGPT (Shen
et al., 2023), and AudioGPT (Huang et al., 2023b).
Conversely, to mitigate propagated errors in the
cascade system, initiatives like NExT-GPT (Wu
et al., 2023d), CoDi-2 (Tang et al., 2023c), and
ModaVerse (Wang et al., 2024d) have developed
end-to-end MM-LLMs of arbitrary modalities. The
timeline of MM-LLMs is depicted in Figure 1. -->
4. 早期的研究主要集中在 MM 内容理解和文本生成，包括 图像-文本、视频-文本 和 音频-文本理解
5. 后续的 MM-LLM 支持特定模态输出，包括 图像-文本输出 和 音频-文本输出
6. 现有的研究模仿人类任意模态转换，时间线如下：
![](image/Pasted%20image%2020241220173719.png)
<!-- In this paper, we present a comprehensive survey
aimed at facilitating further research of MM-LLMs.
To provide readers with a holistic understanding of
MM-LLMs, we initially delineate general design
formulations from model architecture (Section 2)
and training pipeline (Section 3). We break down
the general model architecture into five compo-
nents: Modality Encoder (Section 2.1), Input Pro-
jector (Section 2.2), LLM Backbone (Section 2.3),
Output Projector (Section 2.4), and Modality Gen-
erator (Section 2.5). The training pipeline elu-
cidates how to enhance a pre-trained text-only
LLM to support MM input or output, primarily
consisting of two stages: MM PT (Section 3.1)
and MM IT (Section 3.2). In that section, we
also provide a summary of mainstream datasets
for MM PT and MM IT. Next, we establish a tax-
onomy encompassing 126 State-of-the-Art (SOTA)
MM-LLMs, each characterized by specific formu-
lations, and summarize their development trends
in Section 4. In Section 5, we comprehensively
review the performance of major MM-LLMs on
mainstream benchmarks and distill key training
recipes to enhance the efficacy of MM-LLMs. In
Section 6, we offer promising directions for MM-
LLMs research. Moreover, we have established
a website (https://mm-llms.github.io) to track the
latest progress of MM-LLMs and facilitate crowd-
sourcing updates. Finally, we summarize the en-
tire paper in Section 7 and discuss related surveys
on MM-LLMs in Appendix A. We aspire for our
survey to aid researchers in gaining a deeper under-
standing of this field and to inspire the design of
more effective MM-LLMs. -->
7. 本文为 MM-LLMs 的综述：
    1. 给出了通用的模型结构和训练流程，包括 5 个部分：
        + Modality Encoder
        + Input Projector
        + LLM Backbone
        + Output Projector
        + Modality Generator
    2. 提供了 126 个 SOTA MM-LLMs 的分类法，并总结了它们的发展趋势
    3. 回顾 MM-LLMs 在主流 benchmark 上的表现，并总结了提高 MM-LLMs 效果的关键训练方法

## 模型架构
<!-- In this section, we provide a detailed overview
of the five components comprising the general
model architecture, along with the implementation
choices for each component, as illustrated in Fig-
ure 2. MM-LLMs that emphasize MM understand-
ing only include the first three components. During
training, Modality Encoder, LLM Backbone, and
Modality Generator are generally maintained in a
frozen state. The primary optimization emphasis
is on Input and Output Projectors. Given that Pro-
jectors are lightweight components, the proportion
of trainable parameters in MM-LLMs is notably
small compared to the total parameter count (typi-
cally around 2%). The overall parameter count is
contingent on the scale of the core LLM utilized
in the MM-LLMs. As a result, MM-LLMs can be
efficiently trained to empower various MM tasks. -->
通用的模型架构如图：
![](image/Pasted%20image%2020241221152620.png)