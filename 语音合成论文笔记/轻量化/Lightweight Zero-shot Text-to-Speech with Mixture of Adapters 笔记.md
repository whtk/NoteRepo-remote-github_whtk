> Interspeech 2024，NTT
<!-- 翻译 & 理解 -->
<!-- The advancements in zero-shot text-to-speech (TTS) meth- ods, based on large-scale models, have demonstrated high fi- delity in reproducing speaker characteristics. However, these models are too large for practical daily use. We propose a lightweight zero-shot TTS method using a mixture of adapters (MoA). Our proposed method incorporates MoA modules into the decoder and the variance adapter of a non-autoregressive TTS model. These modules enhance the ability to adapt a wide variety of speakers in a zero-shot manner by selecting appropri- ate adapters associated with speaker characteristics on the ba- sis of speaker embeddings. Our method achieves high-quality speech synthesis with minimal additional parameters. Through objective and subjective evaluations, we confirmed that our method achieves better performance than the baseline with less than 40% of parameters at 1.9 times faster inference speed. -->
1. 提出采用 mixture of adapters (MoA) 的 TTS：
    1. 将 MoA 引入到非自回归 TTS 模型的 decoder 和 variance adapter 中
    2. 通过选择与 speaker embeddings 相关的 adapters 来增强 zero-shot 能力
2. 可以在保持高质量语音合成的同时，减少参数量，提高推理速度

## Introduction
<!-- Advancements in text-to-speech (TTS) synthesis have enabled high-quality natural-sounding speech generation by leveraging large amounts of single and multi-speaker speech data [1–3]. This has facilitated the development of TTS for diverse speak- ers using a minimal amount of target-speaker utterances. This capability extends to zero-shot scenarios in which the acoustic model can adapt without retraining [4–6]. Following the suc- cesses of large-scale language models [7] in zero-shot and few- shot adaptation, zero-shot TTS methods have achieved high fi- delity by using large-scale models [5, 8]. However, these meth- ods are not well-suited for daily applications such as conversa- tion robots, virtual assistants, and personalized TTS services, as they can not run on edge devices such as smartphones due to their substantial parameter sizes. While there is a demand, syn- thesizing high-quality speech under a zero-shot condition with a lightweight TTS model remains an ongoing challenge. Achiev- ing this presents a significant challenge, as it requires maintain- ing the modeling capability to capture the diverse characteristics of thousands or more speakers with a model that has limited ex- pressiveness due to its constrained number of parameters. -->
1. 用 lightweight TTS 在 zero-shot 情况下合成高质量语音仍然具有挑战性
<!-- To meet the demand for lightweight TTS, various meth- ods have emerged, including autoregressive [9, 10], non- autoregressive [11, 12], and diffusion-based [13] methods. However, none of these methods meets the requirement for lightweight zero-shot TTS. While PortaSpeech [11] and Light- Grad [13] are lightweight, they are designed for single-speaker TTS, which only requires lower modeling ability. Light- TTS [12] is a multi-speaker TTS method but is only trained with about a few hundred speakers, which is not enough to achieve natural zero-shot TTS, as suggested with models trained with a few thousand speakers [4–6]. -->
2. 目前的方法都无法满足 lightweight zero-shot TTS 的需求
    1. PortaSpeech 和 LightGrad 只适用于单说话人 TTS
    2. LightTTS 只训练了几百个说话人，不足以实现 zero-shot TTS
<!-- Designing models that are both expressive and parameter-efficient is a common challenge across fields such as classical pattern recognition to large models in natural language process- ing (NLP) and computer vision. One representative approach to address this is mixture of experts (MoE), with which multi- ple parallel expert modules are used, and one or more are se- lectively activated [14–20]. Determining the weights on these experts, i.e., selecting the appropriate expert using an input se- quence or related information, enables the model to effectively handle diverse tasks. Consequently, MoE enhances model ca- pacity while roughly maintaining training and inference effi- ciency by minimal additional parameters. Therefore, we em- brace the concept of mixture of adapters (MoA) [21, 22], an MoE variant primarily used in NLP. ADAPTERMIX [23] lever- ages MoA for speaker adaptation in TTS, focusing on using stronger adapters to capture fine-grained speaker characteristics within the training data, i.e., few-shot adaptation, without delv- ing into its potential for zero-shot TTS. Therefore, the strategy for adapter selection does not explicitly use speaker informa- tion. -->
3. MoE 和 MoA 可以提高模型能力，同时保持训练和推理效率
    1. MoA 主要用于 NLP
    2. ADAPTERMIX 将 MoA 用于 TTS，但是没有涉及 zero-shot TTS
<!-- We propose a lightweight zero-shot TTS method that is based on the concept of MoA. The key idea involves MoA gated with speaker embeddings. The proposed method is able to change the network configuration depending on the speaker characteristics. Therefore, it enables, at inference time, the ar- rangement of an efficient model adapted to the speaker using a small amount of speaker data. Because the model with MoA is trained on a large training dataset containing many speak- ers, the proposed method can cover various speaker character- istics at inference. To evaluate the proposed method, we con- ducted objective and subjective evaluations while varying the model size. The experimental results indicate that the inser- tion of MoA modules significantly enhances the lightweight model with minimal additional parameters and that the pro- posed method achieves better modeling ability than baselines by only using less than 40% of parameters at 1.9 times faster in- ference speed. Audio samples are available on our demo page1. -->
4. 提出了一种基于 MoA 的 lightweight zero-shot TTS
    1. MoA 与 speaker embeddings 相关联
    2. 在推理时，可以根据说话人特征调整网络配置
    3. 通过在大型训练数据集上训练，可以在推理时覆盖各种说话人特征
    4. MoA 模块使用了不到 40% 的参数，推理速度提高了 1.9 倍

## 方法
<!-- With our method, we expand the TTS model with MoA mod- ules. Zero-shot TTS models typically comprise three main com- ponents: the TTS model with encoder and decoder, speaker- embedding extractor, and vocoder. Considering the utility of a TTS method, speaker embeddings can be extracted and stored before speech generation, enabling extraction on powerful com- putational devices or servers and eliminating the need for lightweight speaker embedding extractors. Therefore, we use a TTS method using a speaker-extraction method that is based on a self-supervised learning (SSL) speech model [6], which has demonstrated superior speech quality compared with conven- tional speaker recognition-based methods, e.g., d-vector [24,25] and x-vector [4, 26]. For the vocoder, lightweight methods have recently emerged, including those based on inverse short-time Fourier transform [27,28], and some have zero-shot ability [28]. These recent vocoders can achieve a low real-time factor (RTF), even lower than that of the TTS models. Consequently, it is im- portant to develop lightweight TTS models for decreasing in- ference time of the entire system. Therefore, we concentrate on a lightweight TTS model, excluding the speaker embedding extractor and vocoder from the discussion. We now briefly in- troduce the backbone TTS model and MoA modules. -->
Zero-shot TTS 模型包含三部分：
    + TTS 模型（encoder 和 decoder）
    + speaker-embedding extractor：基于 SSL speech model
    + vocoder

本文主要关注 TTS 模型和 MoA 模块。

### Backbone SSL-based TTS 模型
<!-- The proposed method uses the SSL-based embedding extrac- tor to process input speech sequences. This extractor consists of an SSL model followed by an embedding module, which converts the speech representations from the SSL model into a fixed-length vector, i.e., a speaker embedding. The embed- ding module comprises three components: weighted-sum, bidi- rectional GRU, and attention. In the weighted-sum compo- nent, the speech representations from each layer of the SSL model are weighted using learnable weights then summed, fol- lowing the same approach described in a previous paper [29]. Subsequently, the bidirectional GRU processes these summed representations, and their hidden states are further aggregated through an attention layer [30,31]. Finally, the obtained speaker embeddings are fed into the TTS model. As both the TTS model and embedding module are jointly trained, suitable speaker em- beddings for the TTS model are obtained from the embedding module. During inference, we can use the embedding extractor separately from the TTS model and compute the speaker em- bedding in advance, similar to d-vector and x-vector. -->
模型采用 SSL-based embedding extractor 处理输入语音序列，包括：
    + SSL model
    + embedding module：将 SSL 特征转为 speaker embedding，包含：
        + weighted-sum
        + bidirectional GRU
        + attention
    
在推理时，可以单独使用 embedding extractor 计算 speaker embedding。

### 基于 MoA 的 Speaker embedding
<!-- Figures 1b and 1c respectively show an overview of the feed- forward Transformer (FFT) block of the decoder and the predic- tors (i.e., pitch, energy, and duration predictors) with an inserted MoA module (Fig. 1d). This module comprises N lightweight bottleneck adapters, each consisting of two feed-forward layers with layer normalization [32], and a trainable gating network that determine the weights on the adapters using speaker em- beddings. All components of the networks are jointly trained with the backbone TTS model. -->
下图 b 为 decoder 的 FFT block，c 为 predictors（pitch, energy, duration predictors）和 MoA module。MoA module 包含 N 个 lightweight bottleneck adapters，每个包含两个 feed-forward layers 和一个 trainable gating network，用于根据 speaker embeddings 确定 adapters 的权重。
![](image/Pasted%20image%2020240715115701.png)

<!-- The MoA module is expressed as follows: -->
MoA module 计算如下：
$$\mathrm{MoA}(\mathbf{x},\mathbf{x}_\mathbf{e})=\mathbf{x}+\sum_{i=1}^Ng_i(\mathbf{x}_\mathbf{e})\cdot\text{Adapter}_i(\mathbf{x})$$
<!-- where x ∈ RD is the input, xe ∈ RDemb is the speaker em- bedding, Adapteri : RD 7→ RD represents an adapter from a set {Adapteri(x)}Ni=1 of N adapters, and gi : RDemb 7→ RN is the trainable gating network parameterized by neural networks. We investigated two approaches for MoA. First, a dense MoA, where the summation is executed on all the adapters N. Then a sparse version, with which we only keep the top-k gi weights and set the other weights to zero2. The sparse version enables high representation power by having a large number of adapters during training while reducing the inference time. To encourage balanced load across weights for adapters, our method trained models with multi-task objectives, where the loss consists of the standard mean square error (MSE) losses and an additional auxiliary loss i.e., importance loss (Limportance) [17], defined as -->
其中，$\mathbf{x} \in \mathbb{R}^D$ 为输入，$\mathbf{x}_\mathbf{e} \in \mathbb{R}^{D_{emb}}$ 为 speaker embedding，$\text{Adapter}_i : \mathbb{R}^D \rightarrow \mathbb{R}^D$ 表示 N 个 adapters，$g_i : \mathbb{R}^{D_{emb}} \rightarrow \mathbb{R}^N$ 是一个 trainable gating network。MoA 有两种实现方式：
    1. dense MoA：对所有 adapters 求和
    2. sparse MoA：只保留 top-k 的 $g_i$ 权重，其他设置为 0

用多任务目标函数来训练模型，其中 loss 包括 MSE losses 和 importance loss：
$$\begin{aligned}
L_{importance}(\mathbf{X})& =\left(\frac{\sigma(\text{Importance}(\mathbf{X}))}{\mu(\text{Importance}(\mathbf{X}))}\right)^2  \\
\mathrm{Importance}(\mathbf{X})& =\sum_{\mathbf{x_e}\in\mathbf{X}}g_i(\mathbf{x_e}) 
\end{aligned}$$
<!-- where X ∈ Rn×D is the batch of speaker embeddings, and μ
and σ are average and standard deviation of the sequence. -->
其中，$\mathbf{X} \in \mathbb{R}^{n \times D}$ 为 speaker embeddings 的 batch，$\mu$ 和 $\sigma$ 分别为序列的平均值和标准差。

## 实验
<!-- For training TTS models, we used an in-house 960-hour Japanese speech database which includes 5,362 speakers: 3,242 female and 2,120 male. This database includes several speaker types, such as professional speakers, i.e., newscasters, narrators, and voice actors, as well as non-professional speakers. Among the female and male speakers, 160 and 92 are professionals, respectively. The database was split into three parts: 303,406 utterances by 5,296 speakers for training, 6,807 by 50 speakers for validation (26 females and 25 males, including 6 and 5 pro- fessional speakers, respectively), and 6,809 by 64 speakers for testing (35 females and 30 males, including 9 and 5 professional speakers, respectively). The sampling frequency of the speech was 22 kHz. -->
数据集：960 小时的日语
<!-- The TTS model was FastSpeech2, as implemented in a previous study [3], featuring four-layer Transformers for the encoder and six-layer Transformers for the decoder. To confirm the effec- tiveness of MoA insertion and compare it with models having larger parameters, we trained four models: Small (S), Medium Small (M/S), Medium (M), and Large (L), with 14, 19, 42, and 151M parameters, respectively, obtained by varying the hidden dimensions and filter size of the decoder and encoder and the filter size of the predictors. Table 1 outlines their hyperparame- ters. -->
基于 FastSpeech2 模型，encoder 4 层，decoder 6 层，四种配置：
    + Small (S)：14M 参数
    + Medium Small (M/S)：19M 参数
    + Medium (M)：42M 参数
    + Large (L)：151M 参数

<!-- MoA modules were inserted into the Small model. To con- firm the advantage of sparse gating, we conducted experiments with two types of MoA: sparse (Proposed(s)) and dense (Pro- posed(d)). In the former, there were 8 adapters (N =8), and the k in top-k sampling was set to 3, while in the latter, N and k were both set to 3, i.e., without top-k sampling. Since both types use three adapters during inference, their computational cost at inference is identical. The bottleneck size of the adapters was 96. As shown in Table 1, smaller parameter models achieved higher inference speed, and the additional MoA modules did not largely affect speed. -->
MoA 模块插入 Small 模型，两种 MoA 实现：
    + sparse MoA：8 个 adapters，top-k 为 3
    + dense MoA：3 个 adapters

参数量和速度比较如下：
![](image/Pasted%20image%2020240715144131.png)

<!-- The input and target sequences were a 303-dimensional linguistic vector and 80-dimensional mel-spectrograms with a 10.0 ms frame shift. We used the publicly available HuBERT BASE [33] trained with ReazonSpeech [34]3 as the SSL model. HuBERT processed the 16 kHz raw audio input sequence into 768-dimensional sequences, and the embedding modules con- verted them into fixed-length vectors with the same size as the decoder dimension. For waveform generation, we used HiFi- GAN [35]. Fine-tuning HiFi-GAN for each TTS model could improve naturalness and similarity, but for fair performance comparison between TTS models, we used HiFi-GAN without fine-tuning. While lighter vocoders are available, their usage is beyond the scope of our study. -->
输入和目标序列分别为 303 维的语言向量和 80 维的 mel-spectrograms，帧移为 10.0 ms。使用 HuBERT BASE 作为 SSL model，将 16 kHz 的原始音频输入转为 768 维序列，embedding modules 将其转为与 decoder 维度相同的固定长度向量。使用 HiFi-GAN 生成波形。

## 结果
<!-- We first conducted an objective evaluation to evaluate the per- formance of the proposed method under a data-parallel condi- tion, where the text to be synthesized matches the text of the reference speech, i.e., the input speech sequence to the embed- ding extractor. Past research [6] indicates that objective met- rics may not accurately reflect the reproduction ability under a non-parallel condition due to intra-speaker variation, i.e., slight inconsistencies within utterances from the same speaker, at- tributable to the high reproduction quality of SSL-based TTS models for reference speech. The evaluation metrics included mel-cepstral distortion (MCD), and root MSE (RMSE) of log- arithmic F0 and phoneme durations. To calculate MCD and F0 RMSE between the generated and test speech with the same time alignment, without using dynamic time warping, we gen- erated a log mel-spectrogram using the original phoneme dura- tions extracted from the test speech. The RMSE of the phoneme durations was computed by comparing the original phoneme durations of the test speech with those predicted by the dura- tion predictor. -->
客观评估：MCD，F0 RMSE 和 phoneme durations RMSE，结果如下：
<!-- Figure 2 shows the objective evaluation results. We cal- culated the average of each metric per speaker, and the figures illustrate their distributions. The third quartile is the point, as it serves as a good indicator of modeling ability, reflecting the per- formance in reproducing challenging speakers. A comparison across models with varying parameter sizes, i.e., L, M, M/S, and S, revealed that performance degraded as the number of param- eters decreased. This suggests that simply reducing the dimen- sions in the TTS model could compromise speech generation performance, as smaller models have lower modeling ability. -->
![](image/Pasted%20image%2020240715144919.png)
