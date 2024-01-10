> MFA 经典论文，Interspeech 2017，McGill University，University of Maryland

1. 提出 MFA，用于语音-文本对齐的开源系统

## Introduction

1. 提出 MFA，继承并发展了Prosodylab-Aligner的功能：
	1. 用 triphone acoustic models 捕捉音素中的上下文变异，不同于Prosodylab-Aligner 和其他当前对齐器（例如FAVE）使用的单音素声学模型
	2. 还可以实现 speaker adaptation 来建模 interspeaker differences
2. 采用 Kaldi 语音识别工具包
3. 在单词检测、phoneme boundaries 任务上验证性能

## MFA

MFA 是一个开源的命令行工具，在 Windows 和 Mac 上构建。基于 Kaldi，有三个特性：
+ trainability
+ portability
+ scalability

MFA 使用的 ASR 采用标准的 GMM/HMM 