> ICML 2022，

1. YourTTS 将 multilingual 的方法引入到 zero-shot multi-speaker TT，模型基于 VITS，可以在 zero-shot multi-speaker TTS 中实现 SOTA 的效果，在 zero-shot voice conversion 中实现和 SOTA comparable 的效果
2. 且可以用少于一分钟的语音 fine tune 模型，在语音相似度上可以实现 SOTA 的结果

> 所以，改进就是引入了 language embedding 和 speaker embedding？？？

## Introduction

1. zero-shot TTS：可以合成新的说话人的声音（在训练时没有数据）
2. 目前的 ZS-TTS 仍然需要大量的训练数据
3. multilingual TTS 可以同时学习多语音模态信息，并且可以实现 code-switching，这对 ZS-TTS 很有帮助（因为也是保留说话人信息而改变语音内容）
4. 提出 YourTTS，给出一些新的关于 zero-shot multi-speaker 和  multilingual training 的点，贡献如下：
	1. 英语中可以 实现 SOTA
	2. 第一个采用 multilingual  方法的 zero-shot multi-speaker TTS
	3. 可以实现 zero-shot multi-speaker TTS 和 zero-shot Voice Conversion，质量和相似度都有保障（在 target language 只有一个说话人的时候）
	4. 少于一分钟的语音来 fine tune 模型即可实现好的效果

## YourTTS

![](image/Pasted%20image%2020230927095536.png)

基于 VITS，但是有一些修改以实现 zero-shot multi-speaker 和 multilingual training。

第一，采用 raw text 作为输入而非 phoneme，避免了 grapheme 转到 phoneme 效果不好。

采用的是 VITS 中用的 transformer-based text encode，但是对于 multilingual training，将一个 4维的可训练的 language embedding 拼接到 character embedding 中，同时把 transformer blocks 增加到 10，channel 数 196，decoder 采用 4 层的 affine coupling layers，每个 layer 都包含 4 个 WaveNet residual blocks。

vocoder 采用 HiFi-GAN v1 的结构，为了实现端到端的训练，将 TTS 模型和 vocoder 通过 VAE 连接，也就是采用 VITS 中提出的 Posterior Encoder，包含 16 non-causal WaveNet residual blocks，输入为 linear spectrogram，预测 latent variable，得到的 latent $z$ 作为 vocoder 和 flow-based decoder 的输入。也采用 VITS 中使用的 stochastic duration predictor。

为了使模型有 zero-shot multi-speaker 的生成能力，在
+ affine coupling layers of the flow-based decoder
+  posterior encoder
+ vocoder 
中都添加了 speaker embedding。

研究了 Speaker Consistency Loss 作为损失，采用一个预训练的 speaker encoder 提取 speaker embedding，然后损失为：
$$L_{SCL}=\frac{-\alpha}n\cdot\sum_i^ncos\_sim(\phi(g_i),\phi(h_i))$$
其中的 $g$ 表示 GT，$h$ 表示生成的音频。

训练的时候，Posterior Encoder 的输入为 linear spectrograms 和 speaker embeddings，预测 $z$，然后 $z$ 和 speaker embedding 用于 vocoder 生成波形。Flow-based decoder 将 $z$ 和 speaker embedding 作为条件求 $P_{Z_p}$ 的先验分布，而为了将 $P_{Z_p}$ 和 text encoder 的输出进行对齐，也采用了 MAS。Sochastic duration predictor 的输入为 speaker embedding、language embedding 和 MAS 提取的 duration，其损失为 phoneme duration 的对数似然的变分下界。

推理时，不用 MAS，通过 text encoder 产生 $P_{Z_p}$ 分布，然后从噪声中采样经过 VAE 得到 duration，然后从分布 $P_{Z_p}$ 中采样得到 $z_p$，结合 speaker embedding 通过 inverted Flow-based decoder 转成 $z$，最后通过 vocoder 生成波形。

## 实验（略）
