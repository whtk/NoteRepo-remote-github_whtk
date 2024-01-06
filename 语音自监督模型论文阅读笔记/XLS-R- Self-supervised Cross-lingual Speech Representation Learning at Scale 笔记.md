> Facebook，2021，https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec/xlsr

1. 提出基于 wav2vec 2.0 的跨语言 语音表征学习的大规模模型XLS-R
2. 在128种语言的近50万小时公开可用的语音音频上训练具有高达2B参数的模型

其实模型就是 wav2vec2，但是是在多语言的数据集下训练的，包含 $L$ 种语言，每种语言的采样概率为 $p_l \sim\left(\frac{n_l}{N}\right)^\alpha,l=1,\dots,L$，$n_l$ 为这种语言的数据数量，$\alpha$ 为上采样因子，用来控制高低资源比例语言的权衡因子。 

包含总计 436K小时的公开可用数据：
+ VoxPopuli，372K
+ Multilingual Librispeech，44K
+ CommonVoice，7K
+ VoxLingua107，6.6K
+ BABEL，1K

是当时用的数据集最大的模型。24 种高资源，17 个中等资源和88个低资源数据。

在语音翻译、语音识别、语种识别和说话人识别这几个任务进行了评估。

实验设置了多种架构：![](image/Pasted%20image%2020230525155344.png)
