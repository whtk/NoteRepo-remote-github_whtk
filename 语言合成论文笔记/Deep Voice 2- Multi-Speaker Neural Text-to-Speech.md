> Baidu 硅谷 AI Lab，2017，NIPS

1. 提出一种方法，采用低维的可训练的 speaker embeddings 在单一的 TTS 模型中生成不同说话人的语音
2. 提出 Deep Voice 2，基于 Deep Voice 1 的 pipelines，但是用更高性能的模块来构造，从而极大地提高了音频质量
3. 对于 Tacotron，引入 post-processing neural vocoder 也可以提高合成质量
4. 然后展示了将提出的方法用于 Deep Voice 2 和 Tacotron 的 多说话人语音合成，表明一个 TTS 模型就可以合成很多个独一无二的声音，而且每个说话人的训练音频时间少于半个小时

## Introduction

1. 开发支持多个说话人声音的 TTS 系统需要很多数据
2. 本文表明，可以在共享不同说话人的大部分的参数的情况下，构建纯神经网络的多说话人 TTS 系统，单个模型可生成不同的系统，而且每个人的数据更少（相比于单说话人）
3. 
