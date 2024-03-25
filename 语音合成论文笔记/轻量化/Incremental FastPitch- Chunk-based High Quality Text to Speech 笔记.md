> preprint 2024，NVIDIA

1. 并行 TTS 相比于 AR 可控性更高，速度更快，但是不适合增量合成（incremental synthesis），因为 transformer 的全并行结构
2. 提出 Incremental FastPitch，通过改进结构（chunk-based FFT blocks）、训练（采用  receptive- field constrained chunk attention masks）和推理（采用 fixed size past model states），可以增量生成高质量的 Mel chunks
3. 实验表明，Incremental FastPitch 可以产生和 parallel FastPitch 相当的语音质量，但是延迟更低

## Introduction

1. 增量式合成其实就是 streaming TTS，现有的模型大多使用自回归声学模型实现 mel 谱生成
2. AR 模型通过顺序生成来捕获韵律和上下文依赖，但是因为逐帧生成导致推理速度慢，而且由于输入音素和输出帧之间的不稳定对齐，容易出现over-generation 和 word-repeatin 问题；而并行模型 FastPitch 可以一步生成完整的 Mel 谱，提供了更快的推理速度，同时也可以在解码之前生成 pitch、duration 和 speed，提供了更大的灵活性
3. [FastPitch- Parallel Text-to-speech with Pitch Prediction 笔记](FastPitch-%20Parallel%20Text-to-speech%20with%20Pitch%20Prediction%20笔记.md) 使用 transformer decoder，通过整个编码特征序列计算 attention 来生成 Mel 谱，直接切分特征序列为 chunk 并解码，会导致 decoder 只关注 chunk，导致 Mel chunk 边缘的不连续性；如果使用 AR decoder，直接退化为逐帧生成，失去并行优势。因此，增量 TTS 的理想 decoder 应该能够在生成 Mel chunk 的过程中保持并行性，同时保持每个 chunk 的计算复杂度一致
4. 本文提出 Incremental FastPitch，可以生成高质量的 Mel chunk，保持并行性和低延迟。使用 chunk-based FFT blocks 和 fixed-size attention state caching，避免 transformer 增量 TTS 的计算复杂度随合成长度增加而增加。使用 receptive-filed constrained training，研究 static 和 dynamic chunk masks

## 方法

### 增量 FastPitch

如图：
![](image/Pasted%20image%2020240325152309.png)

输入为 complete phoneme 序列，然后增量式地输出 Mel 谱，每个 chunk 包含固定数量的 Mel frames。Incremental FastPitch 和 parallel FastPitch 有相同的 encoder、energy predictor、pitch predictor 和 duration predictor。但是，Incremental FastPitch 的 decoder 由一堆 chunk-based FFT blocks 组成。与 parallel FastPitch 的 decoder 一次性接受整个 upsampled unified feature $\bar{\boldsymbol{u}}$ 并一次性生成整个 Mel 谱不同，Incremental FastPitch 的 decoder 先将 $\bar{\boldsymbol{u}}$ 分成 N 个 chunk $[\bar{\boldsymbol{u}}_1, \bar{\boldsymbol{u}}_2, ..., \bar{\boldsymbol{u}}_N]$，然后逐个转换为 Mel chunk $y_i$。训练时，在 decoder 上应用 chunk-based attention mask，帮助其适应增量推理中的 constrained receptive field，称之为 Receptive Field-Constrained Training。

### Chunk-based FFT Block

chunk-based FFT block 包含一个 multi-head attention (MHA) block 和一个 position-wise causal convolutional feed forward block。与 parallel FastPitch 相比，chunk-based FFT block 的 MHA block 需要两个额外的输入：past key 和 past value，由自身在前一个 chunk 生成时产生。但是没有使用所有累积的历史的 past keys 和 values，而是通过在推理时保留尾部来使用固定大小的 past key 和 value。past size 在增量生成过程中保持一致，防止随着 chunk 数量的增加而增加计算复杂度。MHA 的计算定义为：
$$\begin{aligned}
k_{i}^{t}& =\operatorname{concat}(pk_i^{t-1},KW_i^K)  \\
v_{i}^{t}& =\mathrm{concat}(pv_i^{t-1},VW_i^V)  \\
o_i^t& =\text{attention}(k_i^t,v^t,QW_i^Q)  \\
o_{M}^{t}& =\mathrm{concat}(o_1^t,...,o_h^t)W^O  \\
pk_i^t& =\text{tail\_slice}(k_i^t,S_p)  \\
pv_{i}^{t}& =\text{tail\_slice}(v_i^t,S_p) 
\end{aligned}$$
其中 $pk_i^{t-1}$ 和 $pv_i^{t-1}$ 是来自 chunk $t-1$ 的 head $i$ 的 past K 和 past V。$k_{i}^t$ 和 $v_{i}^t$ 是 embedded K 和 V，其中 past 沿着时间维度与过去连接以计算 chunk $t$ 的 head $i$ 的 attention。$o_{M}^t$ 是 chunk $t$ 的 MHA block 的输出。$W_{i}^K$、$W_{i}^V$、$W_{i}^Q$ 和 $W^O$ 是可训练的权重。$S_p$ 是 past 的大小。$pk_{i}^t$ 和 $pv_{i}^t$ 是从 $k_{i}^t$ 和 $v_{i}^t$ 的尾部沿着时间维度切片得到的。

类似地，position-wise causal convolution feed forward block 的计算定义为：
$$\begin{aligned}
c_1^t& =\mathrm{concat}(pc_1^{t-1},o_M^t)  \\
o_{c_1}^t& =\text{relu}(\text{causal-conv}(c_1^t))  \\
c_{2}^{t}& =\mathrm{concat}(pc_2^{t-1},o_{c_1}^t)  \\
o_{c_{2}}^{t}& =\text{relu}(\text{causal-conv}(c_2^t))  \\
pc_1^{t}& =\text{tail}\_\text{slice}(c_1^t,S_{c_1})  \\
pc_2^t& =\text{tail\_slice}(c_2^t,S_{c_2}) 
\end{aligned}$$
其中 $pc_{1}^{t-1}$ 和 $pc_{2}^{t-1}$ 是两个 causal convolutional layers 的过去状态。从 $pc_1$ 开始，将其与 $o_{M}^t$ 连接以产生 $c_{1}^t$，作为第一个 causal conv layer 的输入。接下来，将第一个 causal conv layer 的输出 $o_{c1}^t$ 与 $pc_{1}^{t-1}$ 连接以生成 $c_{2}^t$。然后将其输入到第二个 causal conv layer，得到最终输出 $o_{c2}^t$。最后，$pc_{1}^{t}$ 和 $pc_{2}^{t}$ 通过从 $c_{1}^{t}$ 和 $pc_{2}^{t}$ 的尾部沿着时间维度切片得到。与可配置的 $S_p$ 不同，我们将 $S_{c1}$ 和 $S_{c2}$ 设置为各自的 conv kernel sizes 减 1，这足以达到与并行推理的等效性。

### Decoder Receptive Field Analysis

如图：
![](image/Pasted%20image%2020240325160613.png)

忽略 positional-wise convolutional feed-forward blocks。右上角的橙色块表示 chunk $t$ 的最终 FFT 输出 $O_t$。深绿色 MHA blocks 是其 multi-head attention、past key 和 past value 输出对 $O_t$ 有贡献的。浅绿色 MHA blocks 是其 past key 和 past value 输出对 $O_t$ 有贡献的。类似地，蓝色块（past keys 和 past values）和黄色块（green MHA blocks 的输入）是对 $O_t$ 有贡献的。通过在 chunk $t$ 生成时将 chunk $t-1$ 的固定大小 past key 和 past value 输入到每个 MHA block，我们可以将 chunk $t$ 的 receptive field 扩展到其前几个 chunk，而无需显式将这些前几个 chunk 作为 decoder 输入。

receptive field $R$ 取决于 decoder 层数和 past keys 和 past values 的大小，如下：
$$\mathcal{R}=(N_d+\lfloor S_p/S_c\rfloor+1)\cdot S_c$$
其中 $N_d$ 是 decoder 层数，$S_p$ 是 past keys 和 past values 的大小，$S_c$ 是 chunk 的大小。$R$ 的单位是 decoder frames 的数量。如果 $S_p$ 小于或等于 $S_c$，那么 MHA block 输出的 past key 和 past value 仅取决于该 MHA block 的输入，因此 $R$ 简单等于 $(N_d+1)\cdot S_c$，与图 2 中所示相同；而如果 $S_p$ 大于 $S_c$，那么 chunk $t$ 的 MHA block 的 past key 和 past value 也取决于前几个 chunk 的 past keys 和 values，导致 $R$ 随着 $S_p/S_c$ 的 floor 线性增长。

### Receptive Field-Constrained Training

推理过程中的 decoder 的 receptive field 是 constrained，需要在训练过程中将 decoder 也用这个约束。因此使用 Receptive Field-Constrained Training，通过在所有 decoder 层上应用 chunk-based attention mask。第一张图的 C 可视化了具有给定 chunk 大小（深灰色）和不同 past 大小（浅灰色）的各种 attention masks。

一种方法是，每个 文本-音频 对随机选择一个 chunk 大小和 past 大小，然后创建动态 mask（类似于 WeNet ASR encoder 中使用的 mask）。动态 mask 可以使 decoder 泛化到不同的 chunk 和 past 大小。但大多数增量系统 TTS 在推理时使用固定的 chunk 大小，在训练过程中使用动态 mask 可能会导致 mismatch。

对于静态 mask，则在训练过程中使用固定的 chunk 大小和 past 大小。

## 实验（略）
