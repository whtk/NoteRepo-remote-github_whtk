> ICASSP 2021，USTC，MSRA

1. 提出 LightSpeech，基于 FastSpeech 采用 neural architecture search (NAS) 来自动搜索轻量化的高效模型
2. 可以实现 15 倍的压缩比和 6.5倍的推理速度提升

## Introduction

1. 现有的一些设计轻量化的神经网络的方法有：shrinking, tensor de-composition， quantization，pruning 等，但是大部分用于 CV 任务
2. NAS 可以自动设计轻量化的模型，但是需要很仔细地设计和选择搜索空间和算法
3. 提出 LightSpeech，利用 NAS 实现 lightweight 和 fast TTS models

## 方法

采用 FastSpeech 作为 backbone。

### Profiling the Model

FastSpeech 2 包含五个部分：
+ encoder：4 层 FFT
+ decoder：4层 FFT
+ duration predictor：2 层 1D 卷积
+ pitch predictor：5 层 1D 卷积
+ energy predictor：和 pitch 结构一致

其中每个模块的尺寸和速度如下：
![](image/Pasted%20image%2020231224214600.png)

可以发现：
+ encoder 和 decoder 占了最大的参数和时间，因此主要就是通过 NAS 来减少这部分的大小
+ predictor 占了剩下的 1/3 的参数，这部分可以手动设计

### 搜索空间设计

encoder 和 decoder 的层数还是保持 4；energy predictor 去掉（效果不会有很大变化）。

怎么设计？：
+ LSTM 不行，推理速度太慢
+ 将原始的 Transformer 分成两部分：MHSA 和 FFN，MHSA 用不同的 attention head 搜索、采用 depthwise separable convolution (SepConv) 来替换卷积、搜索不同的卷积核，最终得到 $11^8=214358881$ 个候选

### 搜索算法

采用 基于 accuracy prediction 的 搜索算法，但是由于 TTS 任务评估涉及到 MOS 等人为评估，采用 validation loss 作为 accuracy。

## 实验（略）

结果
![](image/Pasted%20image%2020231224220613.png)

分析：
+ 直接减少层数和维度会导致性能下降
+ SepConv 很有效，减少大小且不降低性能

最终的结构：
+ encoder：SepConv (k=5), SepConv (k=25), SepConv (k=13) 和 SepConv (k=9)
+ decoder：SepConv (k=17), SepConv (k=21), Sep- Conv (k=9), SepConv (k=13)
+ hidden size 256
+ 其他保持不变