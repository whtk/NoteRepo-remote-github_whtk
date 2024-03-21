> TASLP 2019，Xin Wang

<!-- 翻译&理解 -->
<!-- Neural waveform models have demonstrated better performance than conventional vocoders for statistical paramet- ric speech synthesis. One of the best models, called WaveNet, uses an autoregressive (AR) approach to model the distribution of waveform sampling points, but it has to generate a waveform in a time-consuming sequential manner. Some new models that use inverse-autoregressive flow (IAF) can generate a whole waveform in a one-shot manner but require either a larger amount of training time or a complicated model architecture plus a blend of training criteria.-->
1. 神经模型 vocoder 比传统的 vocoder 效果好，但是很慢
<!-- As an alternative to AR and IAF-based frameworks, we pro- pose a neural source-filter (NSF) waveform modeling framework that is straightforward to train and fast to generate waveforms. This framework requires three components to generate wave- forms: a source module that generates a sine-based signal as excitation, a non-AR dilated-convolution-based filter module that transforms the excitation into a waveform, and a conditional module that pre-processes the input acoustic features for the source and filter modules. This framework minimizes spectral- amplitude distances for model training, which can be efficiently implemented using short-time Fourier transform routines. As an initial NSF study, we designed three NSF models under the proposed framework and compared them with WaveNet using our deep learning toolkit. It was demonstrated that the NSF models generated waveforms at least 100 times faster than our WaveNet-vocoder, and the quality of the synthetic speech from the best NSF model was comparable to that from WaveNet on a single-speaker Japanese speech corpus. -->
2. 提出 neural source-filter (NSF) 框架，可以直接训练、快速合成，包含三个组件：
    1. source module 生成 sine-based 信号作为激励
    2. non-AR dilated-convolution-based filter module 将激励转换为波形
    3. conditional module 预处理输入的声学特征
3. 模型最小化频谱幅度之间的距离进行训练，其可以使用短时傅里叶变换高效实现
4. 设计了三个 NSF 模型，并使用 deep learning toolkit 进行了比较，结果表明 NSF 模型比 WaveNet-vocoder 快 100 倍，且质量相当

## Introduction

