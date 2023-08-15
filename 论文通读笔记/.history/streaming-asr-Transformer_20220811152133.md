<!--
 * @Description: Streaming automatic speech recognition with the transformer model 笔记
 * @Autor: 郭印林
 * @Date: 2022-08-11 14:07:15
 * @LastEditors: 郭印林
 * @LastEditTime: 2022-08-11 15:20:33
-->

## Streaming automatic speech recognition with the transformer model 笔记

1. 提出了基于Transformer的流式ASR
2. 在encoder端的注意力机制上，使用了 time-restricted self-attention
3. 在encoder-decoder端的注意力机制上，使用了 triggered attention
4. 是当时最好的流式ASR模型


### Introduction
1. 常见的基于注意力的流式ASR：
    + neural transducer
    + monotonic chunkwise attention
    + triggered attention
2. 不同的底层神经网络架构：
    + LSTM
    + BLSTM
    + LC-BLSTM
    + PTDLSTM
    + ···
3. 本文贡献：
    + 在encoder端的注意力机制上，使用了 time-restricted self-attention
    + 在encoder-decoder端的注意力机制上，使用了 triggered attention
    + Transformer 和 CTC-loss 联合训练

### 模型细节

![1660202357245](image/streaming-asr-Transformer/1660202357245.png)

1. 