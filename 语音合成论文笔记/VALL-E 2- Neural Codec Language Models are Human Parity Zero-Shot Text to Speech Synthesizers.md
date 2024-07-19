> Microsoft，2024.6
<!-- 翻译&理解 -->
<!-- This paper introduces VALL-E 2, the latest advancement in neural codec language models that marks a milestone in zero-shot text-to-speech synthesis (TTS), achiev- ing human parity for the first time. Based on its predecessor, VALL-E, the new iteration introduces two significant enhancements: Repetition Aware Sampling refines the original nucleus sampling process by accounting for token repetition in the decoding history. It not only stabilizes the decoding but also circumvents the infinite loop issue. Grouped Code Modeling organizes codec codes into groups to effectively shorten the sequence length, which not only boosts inference speed but also addresses the challenges of long sequence modeling. Our experiments on the LibriSpeech and VCTK datasets show that VALL-E 2 surpasses previous systems in speech robustness, naturalness, and speaker similarity. It is the first of its kind to reach human parity on these benchmarks. Moreover, VALL-E 2 consistently synthesizes high-quality speech, even for sentences that are tradition- ally challenging due to their complexity or repetitive phrases. The advantages of this work could contribute to valuable endeavors, such as generating speech for individuals with aphasia or people with amyotrophic lateral sclerosis. See https://aka.ms/valle2 for demos of VALL-E 2. -->
1. 提出 VALL-E 2，首次实现 human parity
2. 提出两个改进：
    1. Repetition Aware Sampling：改进了原始的 nucleus sampling 过程，考虑了解码历史中的 token 重复。可以稳定解码，避免无限循环问题
    2. Grouped Code Modeling：将 codec 的 codes 分组，可以缩短序列长度，提高推理速度，也可以解决长序列建模的挑战
3. 在 LibriSpeech 和 VCTK 数据集上，VALL-E 2 是第一个达到人类水平的系统

## Introduction
<!-- Our previous work, VALL-E [Wang et al., 2023a], marked a significant breakthrough in this area. It is capable of synthesizing personalized speech using only a 3-second recording, while preserving the speaker’s voice, emotion, and acoustic environment. VALL-E is a neural codec language model that represents speech signals as discrete codec codes with a neural audio codec model. Specifically, it trains an autoregressive language model to generate the coarse codec codes and another non- autoregressive model to generate the remaining fine codec codes. Instead of using greedy search, which continually generates silence codec codes, VALL-E uses random sampling for model inference. However, VALL-E has two key limitations: 1) Stability: The random sampling used during inference can lead to instability in output, while nucleus sampling with a small top-p value may cause an infinite loop issue. This can be mitigated by multiple-time sampling and subsequent sorting, but this approach increases the computational cost. 2) Efficiency: The autoregressive architecture of VALL-E is bound to the same high frame rate as the off-the-shelf audio codec model, which cannot be adjusted, resulting in a slower inference speed. -->
1. VALL-E 训练自回归模型生成 coarse codec codes，然后用非自回归模型生成 fine codec codes，但是有两个问题：
    1. Stability：推理时的随机采样可能导致输出不稳定，nucleus sampling 可能导致无限循环问题
    2. Efficiency：VALL-E 的自回归架构受到 off-the-shelf 音频编解码器模型的高帧率限制，推理速度较慢
<!-- Several follow-up works have been proposed to address these problems [Song et al., 2024, Xin et al., 2024, Borsos et al., 2023, Le et al., 2024, Ju et al., 2024]. To improve stability, some works leverage text-speech alignment information in model training and inference [Song et al., 2024, Xin et al., 2024]. These methods, relying on a forced-alignment model, inevitably introduces errors in the alignment result, which could affect the final performance. It also complicates the overall architecture and increases the burden for data scaling up. To improve modeling efficiency, some works explore fully non-autoregressive methods for zero-shot TTS [Borsos et al., 2023, Le et al., 2024, Ju et al., 2024]. However, these methods require frame-aligned text-speech data for model training, facing the same problem as discussed before. Additionally, the non-autoregressive model generates the tokens with a pre-determined duration result, which constrains the search space of the generated speech and sacrifices the prosody and naturalness. -->
2. 非自回归模型需要 frame-aligned text-speech 数据，且生成的 tokens 有预先确定的持续时间，限制了生成搜索空间，牺牲了韵律和自然度
<!-- In this work, we propose VALL-E 2, the first human parity zero-shot text-to-speech synthesis system. Building upon its predecessor VALL-E, VALL-E 2 employs a neural codec language modeling method for speech synthesis and incorporates two key modifications: repetition aware sampling and grouped code modeling. Repetition aware sampling, an improvement over the random sampling used in VALL-E, adaptively employs either random or nucleus sampling for each time step token prediction. This selection is based on the token repetition in the decoding history, enhancing the stability of the decoding process and circumventing the infinite loop issue encountered in VALL-E. Grouped code modeling, on the other hand, partitions the codec codes into groups, each of which is modeled in a single frame in the AR modeling process. This approach not only accelerates inference by reducing the sequence length but also improves performance by mitigating the long context modeling problem. Notably, VALL-E 2 requires only simple utterance-wise speech-transcription pair data for training, greatly simplifying the process of collecting and processing training data and facilitating potential scalability. -->
3. VALL-E 2 使用了两个关键修改：repetition aware sampling 和 grouped code modeling
    1. repetition aware sampling：根据解码历史中的 token 重复，自适应地使用随机或 nucleus sampling，增强了解码过程的稳定性，避免了 VALL-E 中遇到的无限循环问题
    2. grouped code modeling：将 codec codes 分组，每个组在 AR 建模过程中建模为一个 frame，减少了序列长度，加速了推理，改善了性能
<!-- VALL-E 2 is trained on the large-scale Libriheavy dataset [Kang et al., 2024]. Subsequent evaluations demonstrate that it achieves performance on par with human capabilities on both the in-domain LibriSpeech dataset [Panayotov et al., 2015] and the out-of-domain VCTK datasets [Veaux et al., 2016]. As illustrated in Figure 1, VALL-E 2 significantly outperforms VALL-E and other prior works on the LibriSpeech dataset in terms of robustness, naturalness, and similarity score, even achieving human parity performance. The numbers in Figure 1 are relative numbers (△ Score(Model) = Score(Model) − Score(GroundTruth)) based on the results reported in the paper. In this context, human parity indicates that the robustness, naturalness, and similarity metrics of VALL-E 2 surpass those of the ground truth samples (meaning that △ WERR(VALL-E 2) > 0, △ CMOS(VALL-E 2) > 0, and △ SMOS(VALL-E 2) > 0), meaning that VALL-E 2 can generate accurate, natural speech in the exact voice of the original speaker, comparable to human performance. It is important to note that this conclusion is drawn solely from experimental results on the LibriSpeech and VCTK datasets. Moreover, VALL-E 2 can accelerate the decoding process by multiple times with almost no performance degradation. To specifically evaluate the stability of VALL-E 2, we synthesize speech for complex sentences that are hard to read or contain many repeated phrases, and found that VALL-E 2 can always stably generate high-quality speech. The benefits of this work could support meaningful initiatives, such as generating speech for individuals with aphasia or people with amyotrophic lateral sclerosis. We encourage the reader to listen to our samples on the demo page https://aka.ms/valle2. -->
4. VALL-E 2 在大规模的 Libriheavy 数据集上训练，在 LibriSpeech 和 VCTK 数据集上达到了人类水平的性能（out-of-domain）
<!-- VALL-E 2 is purely a research project. Currently, we have no plans to incorporate VALL-E 2 into a product or expand access to the public. VALL-E 2 could synthesize speech that maintains speaker identity and could be used for educational learning, entertainment, journalistic, self-authored content, accessibility features, interactive voice response systems, translation, chatbot, and so on. While VALL-E 2 can speak in a voice like the voice talent, the similarity, and naturalness depend on the length and quality of the speech prompt, the background noise, as well as other factors. It may carry potential risks in the misuse of the model, such as spoofing voice identification or impersonating a specific speaker. We conducted the experiments under the assumption that the user agrees to be the target speaker in speech synthesis. If the model is generalized to unseen speakers in the real world, it should include a protocol to ensure that the speaker approves the use of their voice and a synthesized speech detection model. If you suspect that VALL-E 2 is being used in a manner that is abusive or illegal or infringes on your rights or the rights of other people, you can report it at the Report Abuse Portal. -->

## 相关工作（略）

本文采用 [EnCodec- High Fidelity Neural Audio Compression 笔记](../语音领域其他论文笔记/EnCodec-%20High%20Fidelity%20Neural%20Audio%20Compression%20笔记.md) 来 tokenize 语音信号，采用 [Vocos- Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis 笔记](轻量化/Vocos-%20Closing%20the%20gap%20between%20time-domain%20and%20Fourier-based%20neural%20vocoders%20for%20high-quality%20audio%20synthesis%20笔记.md) 来生成波形。

## VALL-E 2

### Grouped Codec Language Modeling
<!-- Following VALL-E, we use an off-the-shelf neural audio codec model to represent speech signals as discrete codec code sequence, and regard TTS as a conditional codec language modeling task. To improve the efficiency, VALL-E 2 introduce a grouped codec language modeling method, where we partition the codec code sequence into groups of a certain size, and model each group of codec codes as one frame. In this way, we can get rid of the frame rate constraint of the off-the-shelf neural audio codec model, and reduce the frame rate by integer multiples. It is not only beneficial for the inference efficiency but also the overall speech quality by mitigating the long context modeling problem. -->
将 TTS 视为条件 codec language modeling 任务，将 codec code sequence 分为一定大小的 groups，每个 group 的 codec codes 作为一个 frame 建模。这样可以摆脱之前模型的帧率约束，整数倍地减少帧率，好处：
1. 提高推理效率
2. 减少长序列建模问题

<!-- With TTS training objective, VALL-E 2 is optimized to maximize the likelihood of the grouped code sequence given the text condition. Specifically, given an audio sample y and its corresponding tokenized text transcription x = [x0, x1, . . . , x(L−1)], where L is the text sequence length, we first use a pre-trained neural audio codec model to convert the audio sample y into a codec code sequence CT×J = [c0,c1,...,c(T−1)], where T is the code sequence length, J (here J = 8) is the number of the quantizers in the codec model, and each ct represents the 8 codes for each time step. Then wepartitionitintothegroupedcodesequenceCG =[C0:G,CG:2G,...,C(T−G):T]withthegroup size G, and C0:G stands for the group [c0, c1, . . . , c(G−1)]. Due to the typical short silence at the start of an utterance, we can clip a few codes from the start of the code sequence to let the code sequence length T be the integer multiple of the group size without removing any speech information. Finally, we train the VALL-E 2 model θ to minimize the negative log-likelihood of the grouped code sequence CG conditioned on the text sequence x: -->
用的是 TTS 的目标函数，优化 grouped code sequence 的似然。给定音频样本 $\mathbf{y}$ 和对应的 tokenize 后的文本 $\mathbf{x}=[x_0,x_1,\ldots,x_{(L-1)}]$，其中 $L$ 是文本序列长度，首先用预训练的 codec 将 $\mathbf{y}$ 转换为 codec code sequence $\mathbf{C}^{T\times J}=[c_0,c_1,\ldots,c_{(T-1)}]$，其中 $T$ 是 code sequence 长度，$J$ 是 codec 模型中的量化器数量（这里的 $J=8$），每个 $c_t$ 表示每个时间步的 8 个 codes。然后将其分为 grouped code sequence $\mathbf{C}^G=[C_{0:G},C_{G:2G},\ldots,C_{(T-G):T}]$，其中 group size 为 $G$，$C_{0:G}$ 表示 group $[c_0,c_1,\ldots,c_{(G-1)}]$。最后，训练 VALL-E 2 模型 $\theta$，最小化给定文本序列 $\mathbf{x}$ 条件下 grouped code sequence $\mathbf{C}^G$ 的负对数似然：
$$\begin{aligned}\text{L}&=-\log p(\mathbf{C}^G|\mathbf{x};\theta)\\&=-\sum_{t=0}^{T/G-1}\log p(\mathbf{C}_{t\cdot G:(t+1)\cdot G}|\mathbf{C}_{<t\cdot G},\mathbf{x};\theta),\end{aligned}$$
<!-- where Ct·G:(t+1)·G is the t-th group of codec codes [ct·G, . . . , c((t+1)·G−1)], and C<t·G is all the codec codes in the previous (t − 1) groups. -->
其中 $C_{t\cdot G:(t+1)\cdot G}$ 是第 $t$ 个 group 的 codec codes $[c_{t\cdot G},\ldots,c_{((t+1)\cdot G-1)}]$，$C_{<t\cdot G}$ 是之前 $(t-1)$ 个 groups 中的所有 codec codes。
<!-- During inference, VALL-E 2 performs zero-shot TTS task via prompting. Given a text input (containing both the transcription of speech prompt and the text to synthesis) and grouped codec codes from an unseen speaker, serving as the condition and prompt, the model can generate the target grouped codec codes with the corresponding content and speaker’s voice. Specifically, given the text sequence x and the enrolled speech sample of the unseen speaker y′, we can obtain the corresponding grouped code sequence CP = CG<T′ = [C0:G,CG:2G,...,C(T′−G):T′]. Then, We generate the targetgroupedcodesequenceCT =CG≥T′ =[CT′:(T′+G),...,C(T−G):T]conditionedonthetext sequence x and code prompt CP : -->
推理的时候，给定文本输入（同时包含 speech prompt 和要合成的文本）和 unseen speaker 的 grouped codec codes，作为条件和 prompt，模型可以生成包含对应的内容和说话人声音的 grouped codec codes。具体来说，给定文本序列 $\mathbf{x}$ 和 unseen speaker 的 speech 样本 $\mathbf{y}'$，可以得到对应的 grouped code sequence $\mathbf{C}^P=\mathbf{C}^{G<T'}=[C_{0:G},C_{G:2G},\ldots,C_{(T'-G):T'}]$。然后，在给定文本序列 $\mathbf{x}$ 和 code prompt $\mathbf{C}^P$ 的条件下，生成目标的 grouped code sequence $\mathbf{C}^T=\mathbf{C}^{G\geq T'}=[C_{T'}:(T'+G),\ldots,C_{(T-G):T}]$：
$$\begin{aligned}
\text{CT}& =\underset{\mathbf{C}}{\operatorname*{\arg\max}}p(\mathbf{C}|\mathbf{C}^P,\mathbf{x};\theta)  \\
&=\arg\max_{\mathbf{C}}\sum_{t=T^{\prime}/G}^{T/G-1}\log p(\mathbf{C}_{t\cdot G:(t+1)\cdot G}|\mathbf{C}_{<t\cdot G},\mathbf{x};\theta).
\end{aligned}$$
<!-- Finally,wecanconvertthetargetcodesequenceCT tothetargetspeechwaveformusinganoff-the-
shelf neural codec decoder. -->
最后，可以使用 off-the-shelf neural codec decoder 将 code sequence $\mathbf{C}^T$ 转换为 speech waveform。

### VALL-E 2 架构
<!-- Building upon VALL-E,VALL-E 2 also use a hierarchical structure: an Autoregressive (AR) codec language model and a Non-Autoregressive (NAR) codec language model. The AR model generates sequence of the first codec code for each frame in an autoregressive manner, while the NAR model generates each remaining code sequence based on the preceding code sequences in a non-autoregressive manner. Both models utilize the same Transformer architecture with a text embedding layer, a code embedding layer, and a code prediction layer. We use distinct embeddings for the codes from different codec quantizers and share the parameters of the code prediction layer with the parameters of the code embedding layer. In addition, the AR model has a group embedding layer to project the code embedding to the group embedding, and a group prediction layer for the prediction of codes in one group . The NAR model has a code ID embedding layer to specify the ID of the code sequence to predict. The AR model and NAR model have different attention mask strategies: the AR model uses the causal attention strategy and the NAR model uses the full attention strategy, as shown in the right part of Figure 2. -->
VALL-E 2 也使用了 hierarchical 结构：
+ 一个 AR codec language model，以自回归方式生成每个 frame 的第一个 codec code 序列
+ 一个 NAR codec language model，以非自回归方式生成每个剩余的 code 序列

两个模型使用相同的 Transformer 架构，对于来自不同 quantizer 的 codes，使用不同的 embeddings，并且 code prediction 层的参数与 code embedding 层的参数共享。

AR 模型有一个 group embedding 层，将 code embedding 投影到 group embedding，以及一个 group prediction 层，用于预测一个 group 中的 codes。

NAR 模型有一个 code ID embedding 层，用于指定要预测的 code 序列的 ID。AR 模型和 NAR 模型有不同的 attention mask 策略：AR 模型使用 causal attention 策略，NAR 模型使用 full attention 策略。

### VALL-E 2 训练
<!-- Figure 2 shows the overview of VALL-E 2 model training. It is noteworthy that the training of VALL-E 2 requires only simple utterance-wise speech-transcription pair data, without any complex data such as force-alignment result or additional audio clips of the same speaker for reference. This greatly simplifies the process of collecting and processing training data. -->
VALL-E 2 的训练只需要简单的 utterance-wise speech-transcription pair 数据，训练过程如下：
![](image/Pasted%20image%2020240718155904.png)

<!-- Specifically, for each audio and corresponding transcription in the training dataset, we initially utilize the audio codec encoder and text tokenizer to obtain the codec codes C = [c0 , c1 , . . . , c(T −1) ] and the text sequence x = [x0 , x1 , . . . , x(L−1) ], respectively. These are then used for the AR model and the NAR model training. -->
具体来说，对于训练集中的每个音频和对应的文本，首先使用音频 codec encoder 和 text tokenizer 得到 codec codes $\mathbf{C}=[c_0,c_1,\ldots,c_{(T-1)}]$ 和文本序列 $\mathbf{x}=[x_0,x_1,\ldots,x_{(L-1)}]$，然后用于 AR 模型和 NAR 模型的训练。

#### 自回归模型训练
<!-- The AR model is trained to predict the first codec code sequence c:,0 = [c0,0,c1,0,...,c(T−1),0] conditioned on the text sequence x in an autoregressive manner. -->
AR 模型训练预测第一个 codec code 序列 $c_{:,0}=[c_{0,0},c_{1,0},\ldots,c_{(T-1),0}]$，条件是文本序列 $\mathbf{x}$。
<!-- As shown in the lower middle part of Figure 2, we first obtain the text embedding sequence Ex = [ex0,ex1,...,ex(L−1)]andthecodeembeddingsequenceEc =[ec0,ec1,...,ec(T−1)]usingthetext embedding matrix Wx and the code embedding matrix Wc. -->
首先使用 text embedding matrix $\mathbf{W}_x$ 和 code embedding matrix $\mathbf{W}_c$ 得到 text embedding sequence $\mathbf{E}_x=[e_{x0},e_{x1},\ldots,e_{x(L-1)}]$ 和 code embedding sequence $\mathbf{E}_c=[e_{c0},e_{c1},\ldots,e_{c(T-1)}]$：
$$\mathbf{e}_l^x=\mathbf{W}^x\odot x_l,\\\mathbf{e}_t^c=\mathbf{W}^c\odot c_{t,0},$$
<!-- where l and t denotes the indices of each item in the text sequence and code sequence, respectively, and ⊙ denotes index selection. Then, we partition the code embedding sequence into groups of size G, concatenate each group of the the code embeddings in the hidden dimension, and obtain the group embedding sequence Eg = [eg0,eg1,...,eg(T/G−1)] using the group embedding matrix Wg -->
其中 $l$ 和 $t$ 分别表示文本序列和 code 序列中的索引，$\odot$ 表示索引选择。然后将 code embedding sequence 分为大小为 $G$ 的 groups，将每个 group 的 code embeddings 在 hidden dimension 上连接起来，使用 group embedding matrix $\mathbf{W}_g$ 得到 group embedding sequence $\mathbf{E}_g=[e_{g0},e_{g1},\ldots,e_{g(T/G-1)}]$：
$$\mathbf{e}_t^g=\mathbf{e}_{t\cdot G:(t+1)\cdot G}^c\cdot\mathbf{W}^g$$
<!-- We concatenate the text embedding sequence Ex and the group embedding sequence Eg , inserting the embedding of special tokens < eos > and < bos > in between: -->
将 text embedding sequence $\mathbf{E}_x$ 和 group embedding sequence $\mathbf{E}_g$ 进行 concat，插入特殊 token eos 和 bos 的 embedding：
$$\mathbf{E}^0=\mathbf{E}^x\parallel[\mathbf{e}_{<\text{eos}>},\mathbf{e}_{<\text{bos}>}]\parallel\mathbf{E}^g,$$
<!-- where || indicates concatenation in the temporal dimension. We then separately add the learnable position embedding to the text embedding sequence and the group embedding sequence. The AR model is fed with E0 and trained to predict corresponding code sequence with a special token < eos > appended at the end using a linear mapping group prediction layer and softmax code prediction layer. Due to the causal attention mask strategy, the prediction of each code group ct·G:(t+1)·G,0 can only attend to the text sequence x and the preceding codes c<t·G,0, as demonstrated in the lower right part of Figure 2. -->
其中 $\parallel$ 表示在时间维度上的 concat。然后分别将可学习的 position embedding 加到 text embedding sequence 和 group embedding sequence 上。模型输入 $\mathbf{E}^0$，使用 group prediction 层和 softmax code prediction 层，预测对应的 code sequence 和末尾的特殊 token eos。由于 causal attention mask 策略，每个 code group $c_{t\cdot G:(t+1)\cdot G,0}$ 的预测只能关注文本序列 $\mathbf{x}$ 和前面的 codes $c_{<t\cdot G,0}$。
<!-- Overall, the parameters θAR of the AR model is optimized by minimizing the negative log likelihood of the first code sequence c:,0 conditioned on the text sequence x: -->
最终，AR 模型的参数 $\theta_{\text{AR}}$ 通过最小化给定文本序列 $\mathbf{x}$ 条件下第一个 code 序列 $c_{:,0}$ 的负对数似然进行优化：
$$\begin{aligned}
\mathcal{L}_{AR}& =-\log p(\mathbf{c}_{:,0}|\mathbf{x};\theta_{\mathrm{AR}})  \\
&=-\sum_{t=0}^{T/G-1}\log p(\mathbf{c}_{t\cdot G:(t+1)\cdot G,0}|\mathbf{c}_{<t\cdot G,0},\mathbf{x};\theta_{\mathrm{AR}}) \\
&=-\sum_{t=0}^{T/G-1}\sum_{t^{\prime}=t\cdot G}^{(t+1)\cdot G-1}\log p(c_{t^{\prime},0}|\mathbf{c}_{<t\cdot G,0},\mathbf{x};\theta_{\mathrm{AR}}).
\end{aligned}$$
<!-- 
In the AR model of VALL-E 2, the group sequence c:,0 = [c0:G, cG:2G,0, . . . , c(T −G):T,0] is modeled in an autoregressive approach, while the codec codes within each group ct·G:(t+1)·G,0 = [ct·G,0 , c(t·G+1),0 . . . , c((t+1)·G−1),0 ] are modeled in a non-autoregressive way. -->
在 VALL-E 2 的 AR 模型中，group sequence $c_{:,0}=[c_{0:G},c_{G:2G,0},\ldots,c_{(T-G):T,0}]$ 以自回归方式建模，而每个 group 中的 codec codes $c_{t\cdot G:(t+1)\cdot G,0}=[c_{t\cdot G,0},c_{(t\cdot G+1),0},\ldots,c_{((t+1)\cdot G-1),0}]$ 以非自回归方式建模。
> 其实就是批量预测。

#### 非自回归模型训练
<!-- Given the first code sequence generated by the AR model, the NAR model is trained to generate remaining code sequence c:,j for each codec code ID j conditioned on the text sequence x and the preceding code sequences c:,<j in a non-autoregressive manner, where j ∈ [1, . . . , 7]. -->
给定 AR 模型生成的第一个 code 序列，NAR 模型训练生成剩余的 code 序列 $c_{:,j}$，条件是文本序列 $\mathbf{x}$ 和前面的 code 序列 $c_{:,<j}$，以非自回归方式进行，其中 $j\in[1,\ldots,7]$。
<!-- As we have access to all 8 code sequences of the prompt during inference, to better model the speaker information of the prompt, during training, we explicitly split all the code sequences C into an acoustic condition C<T ′ and target code sequences C≥T ′ with a randomly sampled length T ′. The model is then optimized to predict each target code sequence c≥T ′ ,j conditioned on the text sequence x, all J = 8 code sequences in the acoustic condition C<T ′ and the preceding target code sequences C≥T′,<j inanon-autoregressivemanner. -->
在推理时， prompt 的所有 8 个 code 序列是已知的，为了更好地建模 prompt 的说话人信息，在训练时显式地将所有 code 序列 $\mathbf{C}$ 分为 acoustic condition $\mathbf{C}^{<T'}$ 和 target code sequences $\mathbf{C}^{\geq T'}$，长度 $T'$ 为随机采样。模型通过预测每个 target code sequence $c^{\geq T'}_j$ 来进行优化，条件包含三个：
+ 文本序列 $\mathbf{x}$
+ acoustic condition $\mathbf{C}^{<T'}$ 下的所有 $J=8$ 个 code sequences 
+ 前面的 target code sequences $c^{\geq T'}_{<j}$
> 因为训练的时候，数据只有 语音-文本 对，所以才需要这样手动选一部分作为 prompt。

<!--  we first obtain the text embedding sequence Ex = [ex0 , ex1 , . . . , ex(L−1)] using the text embedding matrix Wx, as denoted in Equation 5. Then, we obtain
thecodeembeddingsequenceEc =[ec0,ec1,...,ec(T−1)]byobtainingallthecodeembeddingsin the acoustic condition C<T′ and target code sequences C≥T′,<j with the code embedding matrix Wc, and summing them along with the code ID dimension: -->
首先使用 text embedding matrix $\mathbf{W}_x$ 得到 text embedding sequence $\mathbf{E}_x=[e_{x0},e_{x1},\ldots,e_{x(L-1)}]$，然后使用 code embedding matrix $\mathbf{W}_c$ 得到 code embedding sequence $\mathbf{E}_c=[e_{c0},e_{c1},\ldots,e_{c(T-1)}]$，将 acoustic condition $\mathbf{C}^{<T'}$ 和前面的 target code sequences $\mathbf{C}^{\geq T',<j}$ 中的所有 code embeddings 求和：
$$\mathbf{e}_t^c=\begin{cases}\sum_{k=0}^7\mathbf{W}^c\odot c_{t,k},&t<T'\\\sum_{k=0}^{j-1}\mathbf{W}^c\odot c_{t,k},&t\geq T'\end{cases},$$
<!-- where t is the time step and j is the codec code ID. Next, we obtain the codec code ID embedding ej with the code ID embedding matrix Wid. -->
其中 $t$ 是时间步，$j$ 是 codec code ID。然后使用 code ID embedding matrix $\mathbf{W}_{\text{id}}$ 得到 codec code ID embedding $e_j$：
$$\mathrm{e}^j=\mathbf{W}^{id}\odot j$$
<!-- We concatenate the text embedding sequence Ex, the code embedding sequence Ec, and the codec code ID embedding ej , inserting the embedding of the special token < eos > in the middle: -->
将 text embedding sequence $\mathbf{E}_x$，code embedding sequence $\mathbf{E}_c$ 和 codec code ID embedding $e_j$ 进行 concat，插入特殊 token eos 的 embedding：
$$\mathbf{E}^j=\mathbf{E}^x\parallel[\mathbf{e}_{<\text{eos}>}]\parallel\mathbf{E}^c\parallel[\mathbf{e}_{<\text{eos}>}]\parallel[\mathbf{e}^j].$$
<!-- We then separately add the learnable position embedding to the text embedding sequence and the code embedding sequence, similar to the AR model. The NAR model is fed with Ej and trained to predict the corresponding code sequence c:,j for each codec code id j using a code prediction layer. With the full attention mask strategy, the prediction of each token ct,j can attend to the entire input sequence, as depicted in the upper right part of Figure 2. -->
然后分别将可学习的 position embedding 加到 text embedding sequence 和 code embedding sequence 上，类似于 AR 模型。NAR 模型输入为 $\mathbf{E}^j$，使用 code prediction 层，预测每个 codec code id $j$ 的对应 code sequence $c_{:,j}$。由于 full attention mask 策略，每个 token $c_{t,j}$ 的预测可以关注整个输入序列。
<!-- Overall, the NAR model is optimized by minimizing the negative log likelihood of each j-th target code sequence c≥T′,j conditioned on the text sequence x, all the code sequences of the acoustic condition C<T′ and the preceding j target code sequences c≥T′,<j. -->
最终，NAR 模型通过最小化给定文本序列 $\mathbf{x}$，所有 acoustic condition $\mathbf{C}^{<T'}$ 的 code sequences 和前面的 $j$ 个 target code sequences $\mathbf{C}^{\geq T',<j}$ 条件下，每个第 $j$ 个 target code sequence $c^{\geq T',j}$ 的负对数似然进行优化：
$$\begin{aligned}
\mathcal{L}&_{NAR} =-\log p(\mathbf{C}_{\geq T^{\prime},\geq1}|\mathbf{x},\mathbf{C}_{<T^{\prime}},\mathbf{c}_{\geq T^{\prime},0};\theta_{\mathrm{NAR}})  \\
&=-\sum_{j=1}^7\log p(\mathbf{c}_{\geq T^{\prime},j}|\mathbf{x},\mathbf{C}_{<T^{\prime}},\mathbf{C}_{\geq T^{\prime},<j};\theta_{\mathrm{NAR}}).
\end{aligned}$$
<!-- In practice, to optimize computational efficiency during training, we do not calculate the training loss by iterating over all values of j and aggregating the corresponding losses, but randomly select a j ∈ [1, . . . , 7] and optimize the model using the training loss: -->
实际上，为了在训练时优化计算效率们不会遍历所有的 $j$ 值计算训练损失并求和，而是随机选择一个 $j\in[1,\ldots,7]$，使用训练损失优化模型：
$$\mathcal{L}_{\mathrm{NAR_j}}=-\log p(\mathbf{c}_{\geq T^{\prime},j}|\mathbf{x},\mathbf{C}_{<T^{\prime}},\mathbf{C}_{\geq T^{\prime},<j};\theta_{\mathrm{NAR}}).$$

### VALL-E 2 推理
<!-- Following VALL-E, we perform the zero-shot TTS task via prompting during inference. As depicted in Figure 3, given the text sentence and the enrolled speech sample of the unseen speaker along with its corresponding transcription, we first concatenate the speech transcription and the text sentence, encoded into the text sequence x using the text tokenizer to serve as the text condition. The speech sample is converted into the codes CP = C<T′ = [c0,c1,...,c(T′−1)] using the audio codec encoder to serve as the prompt. By prompting the conditional codec language model, we infer the AR model and NAR model to generate the target codes C≥T ′ = [cT ′ , . . . , c(T −1) ]. Finally, the target codes is used by the audio codec decoder to synthesize the target personalized speech signals. -->
在推理时，通过 prompting 实现 zero-shot TTS 任务，如图：
![](image/Pasted%20image%2020240718164240.png)

给定文本和 unseen speaker 的 speech 及其对应的文本，首先将 speech transcription 和 text sentence 进行 concat，得到文本序列 $\mathbf{x}$ 作为文本条件。speech 样本转换为 codes $\mathbf{C}^P=\mathbf{C}^{<T'}=[c_0,c_1,\ldots,c_{(T'-1)}]$ 作为 prompt。AR 模型和 NAR 模型基于这些 prompt 生成 codes $\mathbf{C}^{\geq T'}=[c_{T'},\ldots,c_{(T-1)}]$。最后将 codes 通过 audio codec decoder 合成 personalized 语音。

#### 自回归模型推理
<!-- We first infer the AR model to generate the first code sequence of the target codes c≥T ′ ,0 conditioned on the text sequence x and the code prompt c<T′,0. With the grouped codec language modeling method, we feed the grouped code sequence to the AR model and generate each group of target codes in an autoregressive way: -->
首先 AR 模型进行推理，生成第一个 target codes 的 code sequence $c^{\geq T'}_{0}$，条件是文本序列 $\mathbf{x}$ 和 code prompt $c^{<T'}_{0}$。使用 grouped codec language modeling 方法，将 grouped code sequence 输入 AR 模型，以自回归方式生成每个 group 的 target codes：
$$\begin{aligned}
\mathbf{c}\geq T^{\prime},0& \begin{aligned}&=\arg\max p(\mathbf{c}_{\geq T^{\prime},0}|\mathbf{x},\mathbf{c}_{<T^{\prime},0};\theta_{\mathrm{AR}})\\&\mathbf{c}_{\geq T^{\prime},0}\end{aligned}  \\
&=\arg\max_{\mathbf{c}_{\geq T^{\prime},0}}\sum_{t=T^{\prime}/G}^{T/G-1}\log p(\mathbf{c}_{t\cdot G:(t+1)\cdot G,0}|\mathbf{x},\mathbf{c}_{<t\cdot G,0};\theta_{\mathrm{AR}}) \\
&=\arg\max_{\mathbf{c}_{\geq T^{\prime},0}}\sum_{t=T^{\prime}/G}^{T/G-1}\sum_{t^{\prime}=t\cdot G}^{(t+1)\cdot G-1}\log p(c_{t^{\prime},0}|\mathbf{x},\mathbf{c}_{<t\cdot G,0};\theta_{\mathrm{AR}}).
\end{aligned}$$
<!-- Different from the random sampling method used in VALL-E, in this work, we propose a repetition aware sampling method to enhance nucleus sampling for the better decoding stability. As detailed in Algorithm 1, given the probability distribution p(ct′ |x, c<t·G,0 ; θAR ) predicted by the AR model, we first generate the target code ct′ by nucleus sampling with a pre-defined top-p value v. Then, we calculate the repetition ratio r of token ct′ in the preceding code sequence with a window size K. If the ratio r exceeds a pre-defined repetition threshold ratio tn, we replace the target code ct′ by random sampling from p(ct′ |x, c<t·G,0 ; θAR ). Although the codec codes in one group are modeled in a non-autoregressive way, they are predicted autoregressively so as to calculate the repetition ratio r and switch between these two sampling methods. With this repetition aware sampling method, the decoding process can not only benefit from the stability of nucleus sampling, but also avoid the infinite loop issue with the help of random sampling. It should be noted that this repetition aware sampling won’t increase the decoding latency since the runtime cost of the additional sampling operation is almost negligible compared to the model inference process. -->
与 VALL-E 中使用的随机采样方法不同，本文提出了一种 repetition aware sampling 方法，以增强 nucleus sampling 以获得更好的解码稳定性。

算法如下：
![](image/Pasted%20image%2020240718165602.png)

给定 AR 模型预测的概率分布 $p(c_{t',0}|\mathbf{x},\mathbf{c}_{<t\cdot G,0};\theta_{\mathrm{AR}})$，首先使用预定义的 top-p 值 $v$ 进行 nucleus sampling 生成 target code $c_{t'}$。然后计算窗口大小 $K$ 下 token $c_{t'}$ 在前面 code sequence 中的重复率 $r$。如果重复率 $r$ 超过预定义的重复阈值 $t_n$，则从 $p(c_{t'}|\mathbf{x},\mathbf{c}_{<t\cdot G,0};\theta_{\mathrm{AR}})$ 中随机采样替换 target code $c_{t'}$。尽管一个 group 中的 codec codes 是以非自回归方式建模的，但它们是自回归地预测的，所以才可以计算重复率 $r$ 并在这两种采样方法之间切换。通过这种 repetition aware sampling 方法，解码过程不仅可以从 nucleus sampling 的稳定性中受益，还可以避免无限循环问题。同时，这种 repetition aware sampling 不会增加解码延迟，额外采样操作的时间几乎可以忽略不计。

#### 非自回归推理
<!-- Given the first code sequence of the target codes c≥T ′ ,0 , we can infer the NAR model with the text condition x and the acoustic condition C<T ′ to generate the remaining code sequences of the target codes C≥T′,≥1: -->
给定第一个 target codes 的 code sequence $c_{\geq T',0}$，可以使用文本条件 $\mathbf{x}$ 和 acoustic condition $\mathbf{C}_{<T'}$ 推理 NAR 模型，生成剩余的 target codes 的 code sequences $\mathbf{C}_{\geq T',\geq1}$：
$$\begin{aligned}\mathbf{C}_{\geq T^{\prime},\geq1}&=\arg\max_{\mathbf{C}_{\geq T^{\prime},\geq1}}p(\mathbf{C}_{\geq T^{\prime},\geq1}|\mathbf{x},\mathbf{C}_{<T^{\prime}},\mathbf{c}_{\geq T^{\prime},0};\theta_{\mathbf{NAR}})\\&=\arg\max_{\mathbf{C}_{\geq T^{\prime},\geq1}}\sum_{j=1}^7\log p(\mathbf{c}_{\geq T^{\prime},j}|\mathbf{x},\mathbf{C}_{<T^{\prime}},\mathbf{C}_{\geq T^{\prime},<j};\theta_{\mathbf{NAR}}).\end{aligned}$$
<!-- To generate the 2-8 code sequence, we perform inference on the NAR model seven times, generating them one by one using a greedy decoding method. Together with the first codec codes generated by the AR model, the whole code matrix C≥T ′ is used for generating the target personalized speech waveform with the corresponding audio codec decoder. -->
为了生成第 2-8 个 code sequence，要对 NAR 模型进行七次推理，使用贪心解码方法逐个生成。将 AR 模型生成的第一个 codec codes 与 NAR 模型生成的所有 code matrix $\mathbf{C}_{\geq T'}$ 一起用于生成 target personalized 语音 waveform。
<!-- VALL-E 2 can not only use a reference utterance of an unseen speaker as prompt to generate the speech cloning his/her voice, but also be able to perform zero-shot speech continuation, in which, we use the complete transcription of the utterance as the text condition and the first 3-second prefix as the prompt for the target personalized speech generation.
 -->
VALL-E 2 不仅可以使用 unseen speaker 的 reference utterance 作为 prompt 生成 speech 来 clone 声音，还可以实现 zero-shot speech continuation，其中，使用 utterance 的完整 transcription 作为文本条件，前 3 秒的前缀作为 prompt 用于生成语音。

## 实验（略）
