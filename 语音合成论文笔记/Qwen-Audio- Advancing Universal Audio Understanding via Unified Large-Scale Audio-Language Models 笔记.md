> 阿里，2023.11.14 preprint

1. 目前缺乏可以处理多种音频类型和任务的预训练语音模型
2. 提出 Qwen-Audio，把 audio-language pre-training 拓展到 30 个任务和多种音频类型中（如人类语音、自然界的声音、音乐、歌曲等），来实现一种统一的语音理解能力
3. 但是直接一起训练所有的任务和数据会导致干扰问题，因为不同任务不同数据集的文本标签差异很大
4. 于是设计了一个多任务的训练框架，在 decoder 端把 hierarchical tags 序列作为条件来鼓励知识共享，通过 tags 来避免干扰
5. 最终可以在不需要 task- specific fine tune 下，在多种 benchmark 任务上实现较好的性能
6. 开发了 Qwen-Audio-Chat，输入可以是 音频和文本，且可以实现多轮对话

> 输出只限于文本？？

## Introduction

1. 现有的 audio- language 多任务语言模型受限于特定的语音类型
2. 提出 QWen-Audio，是一个基于 audio 和 text 输入的大规模 audio-text model，用了很多的数据训练，包含了超过 30 种任务、8 种语言和多种音频类型：
	1. 设计了一个多任务学习框架在 decoder 端把 hierarchical tags 序列作为条件来鼓励知识共享，通过 tags 来避免干扰
	2. 对语音识别应用 word-level time-stamp prediction 任务，可以提升在声音和音乐上的 grounding 和 grounding-based QA 任务的性能，也可以提升 ASR 的性能
	3. 最终在多种任务上超过了很多多任务训练模型
3. 提出 Qwen-Audio-Chat，通过有监督的指令微调，可以灵活利用文本或者音频作为输入实现多轮对话

## 相关工作（略）

## 方法

### 模型架构

架构如图：
![](image/Pasted%20image%2020231130102125.png)
包含一个 audio encoder 和一个 LLM，给定成对的数据 $(a,x)$，$a$ 为音频序列，$x$ 为文本序列，目标是基于音频表征和之前的文本序列 $\boldsymbol{x}_{<t}$ 预测下一个文本 token 的概率：
$$\mathcal{P}_\theta(x_t|\boldsymbol{x}_{<t},\mathrm{Encoder}_\phi(\boldsymbol{a}))$$

其中的 $\theta$ 为 LLM 的参数，$\phi$ 为 audio encoder 的参数。

Audio Encoder：采用单个 audio encoder 来处理不同类型的 audio，用的是 Whisper-large-v2 作为初始化模型（32 层 Transformer，包含两个卷积下采样层作为 stem），encoder 包含 640M 个参数。

Whisper 是通过有监督训练用于语音识别和翻译，其编码的表征包含了丰富的信息，如背景噪声，甚至可以用于恢复原始的语音。Whisper 将音频采样到 16KHz，然后转为 80-channel 的 mel 谱。其中用了 SpecAugment 来做数据增强。

LLM：用的是预训练的 QWen-7B 模型作为初始化模型，也是一个 32 层的 Transformer decoder，hidden size 为 4096，共计 7.7B 的参数。

### 多任务预训练

音频预处理阶段，包含多种语音数据：
![](image/Pasted%20image%2020231130103757.png)

目前是训练一个统一的模型可以支持所有的音频任务，在 co-training 的时候，不同任务之间互相受益：
+ 相似的任务可以进行知识共享和协同学习
+ 依赖于 lower-level 感知能力的任务可以辅助需要 higher-level 理解或推理能力的任务

但是，不同任务之间的文本标签差异很大，简单地混合会造成干扰。大多数现有的多任务训练方法要么对相似任务进行分组，要么为每个数据集分配一个 dataset ID。

Whisper 提出的多任务框架，把特定的任务和条件信息作为一种输入的 special token，但是其进关注于 speech translation 和 recognition 任务。

受 Whisper 的启发，本文提出的多任务训练框架如下：
+ Transcription Tag：整个预测的开始记为 transcription tag，<|startoftranscripts|> 这个 token 用于正确地转录单词和捕获语音的语言内容，如语音识别和语音翻译任务，而对于其他任务，则用  <|startofanalysis|> 这个 token
+ Audio Language Tag：下一步是引入 language tag 用于表明语音的语种，对于每种语言都有一个独一无二的 token（总共八种语言），对于那种不包含特定语言的语音（如自然界的声音、音乐等），用  <|unknown|> token
+ Task Tag：用于指定任务，分为 五类， <|transcribe|>, <|translate|>, <|caption|>, <|analysis|>, <|question-answer|> ，对于 QA 任务，在 这个 tag 之后添加对应的问题
+ Text Language Tag：用于指定输出文本的语种
+ Timestamps Tag：包含 <|timestamps|> 或 <|notimestamps|> 这两个 token，用于决定模型是否需要预测 timestamps。和 Whisper 中用的 sentence level 的 timestamps 不同，这里需要模型做更精细化的 word-level 的 timestamps 预测，称为  SRWT (Speech Recognition with Word-level Timestamps)。如果需要的话，这个后面是和转录的文本交替预测的，也就是先预测 start time token，然后预测一个文本 token，然后预测 end time token，这种可以可以提高对齐能力，从而在语音识别或 QA 任务中得到很大的性能提升
+ Output Instruction：用于进一步制定任务和不同任务需要的输出格式

上面这几个 token 结束后，就开始文本输出了。

这种操作，目标是，通过共享的 tags 来最大化相似任务之间的知识共享，且确保不同任务和输出格式能够区分开。

## 有监督地 fine tune

采用 instruction-based fine-tuning 技巧，使其和人类的意图对齐，从而可以实现一种交互式的对话模型，称为 QWen-Audio-Chat。

为每个任务手动创建 demonstrations，demo 包含 原始的文本标签、问题和答案。然后用 GPT-3.5 来基于 原始的文本标签来生成进一步的问题和答案。

通过手工标注，创建了一个语音对话数据集，从而可以将 推理、故事生成、multi-image comprehension 能力引入到模型中。

为了处理 multi-audio 对话 和 multiple audio 输入，对不同类型的 audio 引入 audio id。

对于对话格式，用 openai 的 ChatML 格式来构建  instruction tuning dataset。用两个 token <im_start> and <im_end> 来标记对话的起止，示例如下：
![](image/Pasted%20image%2020231130161759.png)

为了实现音频和纯文本都可以作为输入，在训练的时候采用上面提到的 audio-centric instruction data 和 纯文本指令 的组合作为输入。instruction tuning data  的总量为 20K。

## 实验