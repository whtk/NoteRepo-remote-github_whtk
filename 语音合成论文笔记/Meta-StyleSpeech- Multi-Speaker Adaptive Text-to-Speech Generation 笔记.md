> ICML 2021，KAIST，AITRICS

1. 提出 StyleSpeech，不仅可以合成高质量的语音，也可以适应到新的说话人
2. 提出 Style-Adaptive Layer Normalization，根据从参考语音中提取的 style 来对齐文本输入的 gain 和 bias
3. 拓展到 Meta-StyleSpeech，引入两个 discriminator

## Introduction

1. 现有的 meta learning 都集中在图像上，不能在 TTS 上用
2. 提出 StyleSpeech 和 Meta-StyleSpeech

## 相关工作（略）

## StyleSpeech

包含 mel-style encoder 和 generator，如图：
![](image/Pasted%20image%2020231207112353.png)

### mel style encoder

输入语音 $X$，输出 一个 vector $w\in\mathbb{R}^{N}$，包含以下三部分：
+ Spectral processing：通过全连接层
+ Temporal processing：采用带有残差连接的 G-CNN 来捕获信息
+ Multi-head self-attention：attention 做完之后在时间上求平均

### Generator

generator $G$ 用于从给定的 phoneme 序列 $t$ 和 style vector $w$ 中生成语音 $\widetilde{X}$，结构基于 FastSpeech 2，包含三个部分：
+ phoneme encoder，将 phoneme embedding 转为 hidden sequence
+ mel 谱 decoder，将长度调整后的序列转为 mel 谱
+ variance adaptor，在 phoneme level 的 序列上预测 pitch 和 energy
 
phoneme encoder 和 mel 谱 decoder 用的都是 FFT 模块，但是都不支持多说话人。
