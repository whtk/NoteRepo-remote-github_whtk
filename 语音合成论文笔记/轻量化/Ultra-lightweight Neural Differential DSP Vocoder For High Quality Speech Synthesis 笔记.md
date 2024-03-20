> preprint 2024, Meta AI
<!-- 翻译&理解 -->
<!-- Neural vocoders model the raw audio waveform and synthesize high- quality audio, but even the highly efficient ones, like MB-MelGAN and LPCNet, fail to run real-time on a low-end device like a smart- glass. A pure digital signal processing (DSP) based vocoder can be implemented via lightweight fast Fourier transforms (FFT), and therefore, is a magnitude faster than any neural vocoder. A DSP vocoder often gets a lower audio quality due to consuming over- smoothed acoustic model predictions of approximate representations for the vocal tract. In this paper, we propose an ultra-lightweight dif- ferential DSP (DDSP) vocoder that uses a jointly optimized acoustic model with a DSP vocoder, and learns without an extracted spec- tral feature for the vocal tract. The model achieves audio quality comparable to neural vocoders with a high average MOS of 4.36 while being efficient as a DSP vocoder. Our C++ implementation, without any hardware-specific optimization, is at 15 MFLOPS, sur- passes MB-MelGAN by 340 times in terms of FLOPS, and achieves a vocoder-only RTF of 0.003 and overall RTF of 0.044 while running single-threaded on a 2GHz Intel Xeon CPU. -->
1. 目前最高效的 vocoder 也很难在低端设备（如智能眼镜）上实时运行
2. DSP vocoder 速度快，但通常因为过度平滑的声学模型预测而导致音频质量低
3. 提出了一种 ultra-lightweight differential DSP (DDSP) vocoder，采用联合优化的声学模型和 DSP vocoder，无需提取声道的频谱特征
4. 效率与 DSP vocoder 相当，但质量与 neural vocoder 相当
5. c++ 实现，单线程运行在 2GHz Intel Xeon CPU 上，RTF 为 0.003，整体 RTF 为 0.044，比 MB-MelGAN 的 FLOPS 高 340 倍

## Introduction
<!-- Neural vocoders that directly model the audio waveform are computationally intensive because modeling the phase of a wave- form is challenging due to its stochastic nature. As described here [9], different phase waveforms can sound the same, whereas wave- forms with different magnitude spectrograms sound different. This observation motivates us to learn only the magnitude spectrograms well by comparing them against that of the true audio while proce- durally generating phase information for efficiency. -->
1. 波形的相位随机，且不同的相位波形听起来可能相同，而不同的幅度谱听起来不同
2. 从而可以只学习幅度谱，然后通过比较生成的幅度谱和真实音频的幅度谱，生成相位信息
<!-- In this paper, we propose a novel DDSP vocoder where we com- bine a simple and efficient DSP vocoder with the acoustic model, described in Section 2.2.1. The acoustic model is a neural net while DSP vocoder does not have any learnable parameters. Since the joint module is end-to-end differentiable, it can learn from the magnitude spectrogram of true audio. Our DDSP vocoder achieves audio qual- ity comparable to state-of-the-art neural vocoders, with the vocoder having a compute of 15 MFLOPS and vocoder-only RTF of 0.003 running single-threaded on a 2GHz Intel Xeon CPU. -->
3. 提出了一种新的 DDSP vocoder，将简单高效的 DSP vocoder 与声学模型结合，声学模型是神经网络，DSP vocoder 没有可学习参数，且模型是端到端可微的

## 提出的方法

通用的 TTS pipeline 如下：
![](image/Pasted%20image%2020240320153759.png)

### 前端

<!-- Linguistic Frontend: Responsible for converting input text into linguistic features. It first normalizes the text, predicts the phonetic transcription using the International Phonetic Alphabet (IPA) [11] and then converts phones, syllable stress, and other supra-segmental information such as phrase type into one-hot features [12]. It also adds pre-trained word embeddings [13] [14] to improve the natural- ness of the prosody. Features at phrase, word, or syllable rate are repeated for each phone to obtain one feature vector per phone. -->
Linguistic Frontend：将输入文本转为 lingustic feature，首先采用 IPA 预测音标，然后将音标、音节重音和其他超音段信息转为 one-hot feature，同时添加预训练的 word embeddings 来提高韵律的自然性
<!-- Prosody Model: Takes the linguistic features provided by the frontend and predicts the duration and the average fundamental fre- quency of each phone. The network architecture is an emformer[15], with a linear layer at input and two linear layers before the output, trained with an L2 loss on reference features estimated on the ground truth audio. -->
Prosody Model：输入 lingustic feature，预测每个音的持续时间和 F0，网络结构是 emformer，输入端有一个线性层，输出端之前有两个线性层，使用 L2 loss 训练，参考特征是基于真实音频估计的
<!-- Upsampler: Uses the phone-wise duration information to roll out the linguistic features into time synchronous frames by repeating them. It also includes the pitch and duration values, along with the positional information of the current frame within the current phone, syllable, word, and phrase. -->
Upsampler：使用 phone-wise 的 duration 将 lingustic feature 扩展，同时包括 pitch 和 duration 值，以及当前帧在当前音素、音节、单词和短语中的位置信息

### DDSP vocoder
<!-- The DDSP vocoder consists of an acoustic model and a differential DSP vocoder, which are trained end-to-end with losses on the final audio waveform. In this section, we first describe the DSP vocoder and the acoustic model architectures separately. We then explain the end-to-end joint training procedure. -->
DDSP vocoder 包括 acoustic model 和 differential DSP vocoder，端到端训练。

#### DSP vocoder
<!-- Our DSP vocoder is based on the source-filter model [16] for speech production, as shown in Figure 2 and takes three input features to generate the output speech signal s: -->
DSP vocoder 基于 source-filter 模型，如下：
![](image/Pasted%20image%2020240320153819.png)

输入下面三个特征，输出生成的语音信号 $s$：
<!-- 1. FundamentalFrequencyF0(1-dimHzvalue)
2. Periodicity P (12-dim mel band-wise ratio between periodic (only impulse train) and aperiodic excitation (only noise))
[17]
3. Vocal Tract Filter V (257-dim linear frequency log magni-
tude) -->
1. 基频 $F0$（1 维，Hz）
2. 周期性 $P$（12 维，mel band-wise ratio between periodic 和 aperiodic excitation）
3. 声道滤波器 $V$（257 维，线性频率对数幅度）
<!-- For the excitation signal E , it is either an impulse train Eimp (F 0) or white noise Enoise of the same energy [18] [19] [20]. Instead of combining them to get the mixed excitation signal, we split the vocal tract filter into the periodic and aperiodic parts by multiplying them with the periodicity feature. We then filter both excitation signals with their filters and add them to the final audio s. This approach allows us to optimize the algorithm used for each excitation type to avoid artifacts and make it computationally efficient. The equations describe the approach with uppercase denoting the variable in fre- quency domain vs lowercase denoting it in time domain. -->
激励信号 $E$ 是脉冲信号 $E_{\text{imp}}(F0)$ 或白噪声 $E_{\text{noise}}$。

这里没有将两者组合得到激励信号，而是将声道滤波器分为周期性和非周期性部分，通过周期性特征相乘得到混合激励信号。然后将两个激励信号滤波并相加得到最终音频 $s$。这种方法可以优化每种激励类型的算法，避免产生伪影，且计算效率高。下式描述了这种方法，大写表示频域变量，小写表示时域变量：
$$\begin{aligned}s&=iFFT(E\times V)\\s&=iFFT([P\times E_{imp}(F0)+(1-P)\times E_{noise}]\times V)\\s&=\underbrace{iFFT(P\times V)*e_{imp}(F0)}_{\text{Periodic signal}}+\underbrace{iFFT((1-P)\times V\times E_{noise})}_{\text{Aperiodic signal}}\end{aligned}$$
<!-- Our vocoder generates audio at a sample rate of 24000 Hz that is merged via overlap-and-add to get the final audio waveform [16]. We choose a frame shift of 128 samples and an FFT size of 512 points. 12-dim P is extrapolated to 257 linear coefficients. With 512 points, we can model frequencies down to 24000Hz/512 ≈ 47Hz, which is sufficient for human speech. We allocate a 512+128 sample buffer for the periodic signal and a 512 sample buffer for the aperiodic signal. The periodic and aperiodic signals for each frame i are then separately generated as follows: -->
vocoder 以 24KHz 的采样率生成音频，通过重叠相加得到最终音频波形。帧移为 128，FFT 大小为 512 点。12 维的 $P$ 扩展为 257 维的线性系数。512 点可以模拟 24000Hz/512 ≈ 47Hz 的频率，对人类语音足够。为周期信号分配 512+128 个样本的缓冲区，为非周期信号分配 512 个样本的缓冲区。每帧的周期和非周期信号分别生成如下：
<!-- Periodic Signal: We multiply the periodicity Pi with the vo-
cal tract filter Vi to get the periodic part Pi × Vi. Then, we con- vert Pi × Vi to the time domain using the inverse FFT and a phase of 180◦. This represents a single impulse filtered by the periodic part of the filter Pi × Vi. Then, we render the filtered impulse train by calculating the time stamps of the impulse within the frame by incrementing a running phase value by 1/F0i. It is multiplied by 1/sqrt(F0i) for energy normalization. Note that it is possible that no impulse falls within the frame at low F 0i values or the periodicity is entirely 0. In that case, we can skip the frame. -->
+ 周期信号：将周期性 $P_i$ 与声道滤波器 $V_i$ 相乘得到周期部分 $P_i \times V_i$。然后使用逆 FFT 和 180° 相位将 $P_i \times V_i$ 转为时域。这代表了一个被周期部分 $P_i \times V_i$ 滤波的单个脉冲。然后通过增加运行相位值 $1/F0_i$ 来计算帧内脉冲的时间戳。为了归一化能量，乘以 $1/\sqrt{F0_i}$。注意，当 $F0_i$ 较小时，可能没有脉冲落在帧内，或者周期性完全为 0。在这种情况下，可以跳过帧。
<!-- Aperiodic Signal: We shift the noise buffer by frame shift of 128 and fill the new 128 values with uniformly distributed pseudo- random numbers between −1... + 1, multiplied with 1/sqrt(24000) to scale the noise to the same energy level as the impulses. We then convert the noise buffer to the frequency domain using forward FFT without any windowing function to get the complex spectrum Enoisei . Note that windowing is not required for Enoisei , since each sample is uncorrelated, which makes Enoisei perfectly periodic withoutanydiscontinuities.WemultiplyEnoisei withtheaperiodic part of the filter Vi × (1 − Pi). The result is then converted back to the time domain using inverse FFT. We then apply a centered Hann window [16] of size 256 points to intermediate noise buffer, so that overlapped audio can sum upto 1.0. -->
+ 非周期信号：将噪声缓冲区移动 128 个帧移，并用均匀分布的伪随机数填充新的 128 个值，乘以 $1/\sqrt{24000}$ 以将噪声缩放到与脉冲相同的能量级。然后使用前向 FFT 将噪声缓冲区转为频域，得到复频谱 $E_{\text{noise}_i}$。注意，对 $E_{\text{noise}_i}$ 不需要窗函数，因为每个样本是不相关的，从而 $E_{\text{noise}_i}$ 具有周期性。将 $E_{\text{noise}_i}$ 与声道滤波器的非周期部分 $V_i \times (1 - P_i)$ 相乘。然后使用逆 FFT 将结果转回时域。然后对中间噪声缓冲区应用一个大小为 256 点的中心 Hann 窗口，以便重叠的音频可以相加到 1.0。

<!-- For each frame, both the intermediate audio buffers are then overlapped and added, with a frame shift of 128 samples to the final audio waveform. -->
对于每个帧，两个中间音频缓冲区重叠相加，帧移为 128，得到最终音频波形。

#### Acoustic Model
<!-- The acoustic model consumes a 512-dim input vector of linguistic features, repeated phone-level F0 and duration, and the positional information for each frame. It follows an emformer architecture[15]; see Table 1, and gives a 270-dim output, which corresponds to 1-dim F 0, 12-dim periodicity P and a 257-dim representation for the vocal tract V . -->
声学模型输入为 512 维的 lingustic feature，重复的 phone-level F0 和 duration，以及每个帧的位置信息。采用 emformer 结构，输出 270 维，对应 1 维的 F0，12 维的周期性 $P$ 和 257 维的声道滤波器 $V$。

#### 联合训练
<!-- As discussed in Section 2.2.1, the excitation signal E contains the phase information, while V is a linear filter on top of E. Since we only observe the speech signal s, we can’t accurately determine V, without knowing E. Methods to determine V in literature, based on cepstral smoothing, linear predictive coding (LPC) extraction or pitch synchronously extracted log mel spectrograms (lmelpsync), assume that V is responsible for slow changes throughout the mag- nitude spectrogram (formants), and create a smoothed version of the magnitude spectrogram of s [21][22]. When we train the acoustic model to lmelpsync features, the prediction errors also add up on top of approximate feature extraction. As a result, the audio sounds muf- fled and unnatural, and is penalized in subjective evaluations. Since the DSP vocoder is differentiable, we can combine it together with the acoustic model. The setup can be jointly optimized by compar- ing the predicted audio against the true audio. This ensures that the spectral feature driving the vocoder is learned instead of engineered, and is optimized via true audio. Figure 3 shows a comparison of two acoustic model outputs, one is the lmelpsync prediction from DSP Vocoder Adv (see Section 3.1, and the intermediate spectral repre- sentation learnt from DDSP vocoder, converted to 80-dim lmel for comparison. We can see that DDSP vocoder has learnt a detailed spectral representation with thinner formants and sharper plosives. -->
激励信号 $E$ 包含相位信息，而 $V$ 是 $E$ 之上的线性滤波器。由于我们只观察到语音信号 $s$，没有办法准确确定 $V$，除非知道 $E$。文献中确定 $V$ 的方法，基于 cepstral smoothing、线性预测编码（LPC）提取或基于 pitch 同步提取的 log mel 频谱图（lmel_psync），假设 $V$ 负责幅度谱中的慢变化（共振峰），并创建幅度谱的平滑版本。当我们训练 acoustic model 时，lmel_psync 特征的预测误差也会叠加在近似特征提取之上。结果，音频听起来沉闷和不自然。由于 DSP vocoder 是可微的，可以将其与 acoustic model 结合。通过比较预测音频和真实音频，可以实现联合优化。这确保了驱动 vocoder 的频谱特征是学习的，并且通过真实音频进行优化。下图显示了两个 acoustic model 输出的比较，一个是 DSP Vocoder Adv 的 lmel_psync 预测（见第 3.1 节），另一个是从 DDSP vocoder 学习的中间频谱表示，转换为 80 维的 lmel 进行比较。可以看到 DDSP vocoder 学习了一个细节丰富的频谱表示，共振峰更细，爆破音更尖：
![](image/Pasted%20image%2020240320154149.png)

#### 训练
<!-- We used three types of losses to train our DDSP vocoder. Window size is kept the same as the FFT size for all audio feature extractions and loss calculations. -->
用三种类型的损失来训练 DDSP vocoder。所有音频特征提取和损失计算的窗口大小都与 FFT 大小相同。

<!-- Reference MSE Loss (on acoustic model predictions): To get the training convergence, we apply an L2 loss for fundamental frequency prediction F 0 with reference F 0. For periodicity feature prediction P ̃, we found that the system could learn it without ex- plicit supervision from the reference P ; however, having an L2 loss with the reference P leads to improved quality with a less breathy voice. -->
在 acoustic model 预测结果中计算的 Reference MSE Loss：对基频预测采用 L2 loss，对周期性特征预测采用 L2 loss：
$$\begin{gathered}
L_{refmse} =L_{refmse\_F0}+L_{refmse\_P} \\
L_{refmse\_F0} =\mathbb{E}_{(F0,\tilde{F}0)}[\lambda_{F0}(F0-\tilde{F}0)^2] \\
L_{refmse\_P} =\mathbb{E}_{(P,\tilde{P})}\left[\frac{\lambda_P}{d_P}(P-\tilde{P})^2\right] 
\end{gathered}$$
<!-- Multi-window STFT loss (on vocoder output): We calculate L1 loss between the amplified log magnitude STFT spectrograms of the reference audio x and predicted audio x ̃ as follows: -->
在 vocoder 输出中计算的 Multi-window STFT loss：计算参考音频 $x$ 和预测音频 $\tilde{x}$ 之间的 log magnitude STFT spectrograms 的 L1 loss：
$$L_{mw\_stft}(G)=\mathbb{E}_{(x,\tilde{x})}\sum_{i=1}^C\frac{\lambda_{stft,i}||X_i-\tilde{X_i}||_1}{N_i}$$
