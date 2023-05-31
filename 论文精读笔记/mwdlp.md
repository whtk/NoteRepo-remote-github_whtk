<!--
 * @Author: error: git config user.name && git config user.email & please set dead value or install git
 * @Date: 2022-06-24 14:36:49
 * @LastEditors: error: git config user.name && git config user.email & please set dead value or install git
 * @LastEditTime: 2022-06-24 17:07:00
 * @FilePath: \PA\mwdlp.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
## High-fidelity and low-latency universal neural vocoder based on multiband WaveRNN with data-driven linear prediction for discrete waveform modeling 笔记（2021年5月提出）
1. 基于WaveRNN，采用 multiband modeling 实现LLRT。
2. 提出了一种基于短时傅立叶变换（STFT）的新型损失函数，用于Gumbel近似下的离散波形建模。

### Introduction
1. 采用大型的稀疏WaveRNN+multiband
2. 采用数据驱动的线性预测技巧来生成离散波形
3. 生成10bit mu律波形
4. 可以生成高品质合成音频（seen和unseen的speaker下），并且每个speaker只有60条训练语句，同时还能实现实时性。

### 原理

#### 变量定义
令 $\boldsymbol{s}=\left[s_{1}, \ldots, s_{t_{s}}, \ldots, s_{T_{s}}\right]^{\top}$ 为离散波形样本，$\boldsymbol{c}=\left[\boldsymbol{c}_{1}^{\top}, \ldots, \boldsymbol{c}_{t_{f}}^{\top}, \ldots, \boldsymbol{c}_{T_{f}}^{\top}\right]^{\top}$ 为特征向量，其中 $\boldsymbol{c}_{{t}_f}$ 为 $d$ 维输入特征。采样级别的序列长度定义为 $T_s$，帧级别的长度定义为 $T_f$。
考虑 bands 的数量为 $M$，则第 $m$ 个 band 的波形序列为 $\boldsymbol{s}^{(m)} = \left[s_{1}^{(m)}, \ldots, s_{t}^{(m)}, \ldots, s_{T_{m}}^{(m)}\right]^{\top}$，显然有，$T_m = T_s/M$，对应的上采样特征向量为 $\boldsymbol{c}^{(u)}=\left[\boldsymbol{c}_{1}^{(u)^{\top}}, \ldots, \boldsymbol{c}_{t}^{(u)^{\top}}, \ldots, \boldsymbol{c}_{T_{m}}^{(u)^{\top}}\right]^{\top}$。则我们的目标是建模离散波形的PMF：
$$
p(\boldsymbol{s})=\prod_{m=1}^{M} \prod_{t=1}^{T_{m}} p\left(s_{t}^{(m)} \mid \boldsymbol{c}_{t}^{(u)}, \boldsymbol{s}_{t-1}^{(M)}\right)=\prod_{m=1}^{M} \prod_{t=1}^{T_{m}} \boldsymbol{p}_{t}^{(m)^{\top} \boldsymbol{v}_{t}^{(m)}}
$$
其中 $\quad \boldsymbol{s}_{t-1}^{(M)}=\left[s_{t-1}^{(1)}, \ldots, s_{t-1}^{(m)}, \ldots, s_{t-1}^{(M)}\right]^{\top}$, $\boldsymbol{p}_{t}^{(m)}=\left[p_{t}^{(m)}[1], \ldots, p_{t}^{(m)}[b], \ldots, p_{t}^{(m)}[B]\right], \quad \boldsymbol{v}_{t}^{(m)}=$ $\left[v_{t}^{(m)}[1], \ldots, v_{t}^{(m)}[b], \ldots, v_{t}^{(m)}[B]\right]^{\top}, \quad \sum_{b=1}^{B} v_{t}^{(m)}[b]=1$, $v_{t}^{(m)}[b] \in\{0,1\}, B$ 是 sample bins 的数量， $\boldsymbol{v}_{t}^{(m)}$ 是 1-hot 向量， $\boldsymbol{p}_{t}^{(m)}$ 是概率向量 (即网络输出)。

#### 用于离散建模的数据驱动线性预测（DLP）
其中，$p_t^{(m)}(b)$ 通过softmax函数计算如下：
$$
p_{t}^{(m)}[b]=\frac{\exp \left(\hat{o}_{t}^{(m)}[b]\right)}{\sum_{j=1}^{B} \exp \left(\hat{o}_{t}^{(m)}[j]\right)}
$$
定义包含所有sample bins的logits向量为 $\hat{\boldsymbol{o}}_t^m = \left[\hat{o}_{t}^{(m)}[1], \ldots, \hat{o}_{t}^{(m)}[b], \ldots, \hat{o}_{t}^{(m)}[B]\right]^{\top}$，则数据驱动的线性预测计算如下：
$$
\hat{\boldsymbol{o}}_{t}^{(m)}=\sum_{k=1}^{K} a_{t}^{(m)}[k] \boldsymbol{v}_{t-k}^{(m)}+\boldsymbol{o}_{t}^{(m)}
$$
$a_t^{(m)}[k]$ 代表时间 $t$ 下第 $m$ 个 band 的第 $k$ 个系数。

#### 网络结构
1. 输入特征进入分段卷积层产生高维的特征向量，再进入全连接层和RELU。
2. coarse 和 fine embedding层用来对 5 bit 的 one-hot 向量进行编码，并且在所有的band之间进行共享
3. 稀疏GRU的隐藏单元数为1184，密集GRU的隐藏单元数为32
4. dual fully-connected：产生两个输出通道，每个通道都和DLP向量和logits向量相一致