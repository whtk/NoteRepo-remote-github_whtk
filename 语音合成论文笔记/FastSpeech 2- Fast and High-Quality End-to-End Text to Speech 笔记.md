> ICLR 2021，ZJU，MSRA


1. FastSpeech 采用自回归的 teacher model 和 知识蒸馏 来获得 one-to-many 的映射，但是有几个缺点：
	1. the teacher-student distillation 很复杂也很耗时
	2. 提取的 duration 不够准确
	3. 两者都会影响合成质量
2. 提出 FastSpeech 2：
	1. 直接使用 ground-truth target 而不需要 target
	2. 在语音中引入更多 variation information（如 pitch, energy）作为条件输入
3. 提出 FastSpeech 2s，第一个直接从文本中并行生成语音，全端到端的推理

## Introduction

FastSpeech 2 简化了 1 的 训练，同时避免了信息的损失：
+ 直接使用 ground-truth target，而非 teacher 的简化版输出
+ 引入 variation information，训练的时候直接从 target speech 中提取 pitch、energy 和 duration，然后在训练的时候作为条件输入；推理时则使用 predictors 得到的值进行推理
	+ 将 pitch contour 使用连续的小波变换转换为 pitch spectrogram
+ 提出 FastSpeech 2s，不使用 mel 谱 作为中间输出直接生成波形

## FastSpeech 2 和 2s

### Motivation

TTS 是一个 one-to-many 的映射问题，因为一句话可以对应很多条语音（不同的 pitch、duration、volume 和 prosody），而自回归模型仅仅把 文本作为输入显然是不够的，会导致模型泛化性能不好。

![](image/Pasted%20image%2020230918132633.png)

### 模型架构

架构如图 a，encoder 将phoneme embedding 转为 phoneme hidden sequence，然后 variance adaptor 添加不同的 variance information，如 duration、pitch 和 energy 到 hidden 序列中。mel-spectrogram decoder 并行将这些序列转为 mel 谱序列。

整体的架构采用的是 FastSpeech 中的 FFT block，但是移除了 teacher-student distillation，直接采用 ground-truth mel-spectrogram 来避免蒸馏过程中的信息丢失。然后 variance adaptor 不仅包含 duration predictor，也包含 pitch and energy predictors，且：
+ duration 通过 forced alignment 获得，比用 attention map 提取出的更准确
+ 额外的 pitch 和 energy 有更多的 variance information

### Variance Adaptor

variance information 分别是指：
1. phoneme duration
2. pitch：传递情感，影响语音的韵律
3. energy：mel 谱的幅度，邮箱语音的音调和韵律

图 b 给出了 adaptor 的结构，c 给出了 predictor 的结构。

Duration Predictor：duration 是对数域的，通过 MSE loss 优化。对于 duration label，采用  Montreal forced alignment （MFA）工具提取（而非Transformer-TTS）。

Pitch Predictor：之前预测 pitch 都是直接预测 pitch contour，但是很困难。于是采用 continuous wavelet transform (CWT) 将 pitch 分解为 spectrogram，再通过 iCWT 转回 pitch contour。然后将每帧的 pitch $F_{0}$ 量化到对数域的 256 个值，然后转成 embedding $p$。

Energy Predictor：计算 STFT 帧的幅度的 L2 norm 作为 energy。然后量化到的 256 个值，然后转成 embedding $e$，采用 MSE loss 训练。

### FastSpeech 2s

直接从文本生成波形，图 a 的右边。设计 waveform decoder，图 d：
+ 考虑到相位信息很难预测，在 waveform decoder  中引入对抗训练隐式地恢复相位信息
+ 结构基于  WaveNet，将对应于某一段 audio clip 的 sliced hidden sequence 作为输入，然后上采样来匹配 样本的长度
+ 采用  Parallel WaveGAN 中的 判别器，采用  multi-resolution STFT loss 和 LSGAN discriminator loss 训练 waveform decoder

### 其他

## 实验（略）


 






