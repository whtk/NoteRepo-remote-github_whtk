> Interspeech 2024，(ASLP@NPU)，数据集

1. 提出 WenetSpeech4TTS，一个多领域的中文语料库来自开源的 WenetSpeech，通过对 WenetSpeech 进行处理，包含 12,800 小时的配对音频文本数据
2. 创建了不同大小的子集，按片段质量分数分类

## Introduction

1. 现有最大的开源中文语音数据集是 DIDISPEECH，包含近 800 小时的朗读风格语音
2. WenetSpeech 是目前最大的开源中文语音数据集，包含 12,483 小时的中文语音数据，来源于 YouTube 和 Podcast
3. 本文提出 WenetSpeech4TTS，一个 12,800 小时的大规模数据集，提供三个子集：Basic、Standard 和 Premium，分别包含 7,226、4,056 和 945 小时的有效数据
4. 从 WenetSpeech 到 WenetSpeech4TTS，合并讲话者相似性和暂停持续时间来细化 WenetSpeech，扩展片段边界来防止截断单词，使用去噪模型增强音频质量，然后进行质量评分，说话者分离系统对相同说话者的片段进行聚类，更先进的 ASR 系统提供更准确的转录
5. 用 VALL-E 和 NaturalSpeech 2 系统在不同子集上训练，结果表明 WenetSpeech4TTS 适用于训练大型 TTS 模型，质量更高的子集表现更好，WenetSpeech4TTS 语料库和相应的基准数据公开可用

## 处理流程

包含：
+ 相邻片段合并
+ 边界扩展
+ 语音增强
+ 多说话者检测
+ 语音识别
+ 质量过滤

## WenetSpeech4TTS 数据集

根据 DNSMOS P.808 分数将 WenetSpeech4TTS 语料库划分为子集，分数高于 4.0 的标记为 Premium，高于 3.8 的标记为 Standard，高于 3.6 的标记为 Basic，低于 3.6 的标记为 Rest，如下表：
![](image/Pasted%20image%2020240721154123.png)

合并操作后，小于 3 秒的片段数量大大减少，大于 5 秒的片段数量显著增加，之前稀有的大于 10 秒的片段增多，有利于训练 TTS 模型，合并策略是合理且必要的。

WenetSpeech4TTS 测试集包含 150 个测试句子和 26 个目标说话者，其中 10 个来自 WenetSpeech4TTS 语料库，16 个来自 WenetSpeech 测试集，5 个来自开源数据集 Aishell，另外 5 个是业余说话者。

WenetSpeech4TTS 语料库，包括片段、脚本和 DNSMOS 分数，全部开源。

## 实验（略）
