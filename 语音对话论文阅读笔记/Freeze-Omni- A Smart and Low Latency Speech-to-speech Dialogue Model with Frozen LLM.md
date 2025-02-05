> preprint 2024.11，腾讯优图、ASLP
<!-- 翻译 & 理解 -->
<!-- Rapidly developing large language models (LLMs) have brought tremendous intel-
ligent applications. Especially, the GPT-4o’s excellent duplex speech interaction
ability has brought impressive experience to users. Researchers have recently
proposed several multi-modal LLMs in this direction that can achieve user-agent
speech-to-speech conversations. This paper proposes a novel speech-text mul-
timodal LLM architecture called Freeze-Omni. Our main contribution is that
the speech input and output modalities can be easily connected to a textual LLM
while keeping the LLM’s parameters frozen throughout the training process. We
design a three-stage training strategy for modeling both the speech input and output,
enabling Freeze-Omni to obtain speech-to-speech conversation ability using text-
speech paired data (such as ASR and TTS data) and only 60,000 multi-round text
Q&A data on 8 GPUs. Moreover, we can effectively ensure that the intelligence
of the Freeze-Omni in the speech modality is at the same level compared with that
in the text modality of its backbone LLM, while achieving low latency end-to-end
spoken response. In addition, we also designed a method to achieve duplex dia-
logue ability through multi-task training, giving Freeze-Omni a more natural style
of dialogue ability between users and agents. In summary, Freeze-Omni holds
great potential to conduct speech-to-speech dialogue based on a multimodal LLM
under the condition of a frozen LLM, avoiding the catastrophic forgetting problem
caused by limited data and training resources -->
1. 提出一种新的 speech-text multimodal LLM 架构 Freeze-Omni，speech input 和 output 可以与 textual LLM 连接，同时在训练过程中保持 LLM 参数冻结
2. 设计三阶段训练策略，模拟 speech input 和 output，使用 ASR 和 TTS 数据以及 60,000 个 multi-round text QA 数据在 8 个 GPU 上训练，实现 speech-to-speech 对话
3. 设计多任务训练方法实现双工对话能力，同时实现低延迟的端到端语音回复

## Introduction
