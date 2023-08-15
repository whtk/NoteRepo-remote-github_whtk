1. 提出 XPhoneBERT，是第一个预训练的用于学习下游 TTS 任务的音素表征的多语言模型
2. 和 BERT 架构相同，使用RoBERTa预训练方法来自近100种语言和地区的330M个音素级句子进行训练
3. 使用XPhoneBERT作为 音素编码器 phoneme encoder 提高了 TTS 模型在自然度和韵律方面的性能

## Introduction

1. 已经有些 TTS 模型将预训练的 BERT 生成的 contextualized word embeddings 用在了 encoder 中，phoneme sequence 通过 encoder 来产生 phoneme 表征；文本通过 BERT 得到 word embedding，然后将两者 concate 得到 decoder 的输入
2. 这其中 BERT 的作用可以帮助提高合成语音的质量，主要是用于为 phoneme 表征提供额外的上下文信息
3. 因此，如果这些信息是直接是直接通过一个 BERT 模型产生的，而且这个模型是用无标签的 phoneme-level 数据训练的，效果应该会更好
4. 最近的模型 PnG BERT、Mixed-Phoneme BERT、Phoneme-level BERT 都已经证明可以提高性能了