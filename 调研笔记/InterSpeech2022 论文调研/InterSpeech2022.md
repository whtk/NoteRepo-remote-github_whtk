# InterSpeech 2022 论文集

## 端到端语音合成
1. [SANE-TTS: Stable And Natural End-to-End Multilingual Text-to-Speech](https://isca-speech.org/archive/interspeech_2022/cho22_interspeech.html) —— 多语言TTS
2. [Enhancement of Pitch Controllability using Timbre-Preserving Pitch Augmentation in FastPitch](https://isca-speech.org/archive/interspeech_2022/bae22_interspeech.html)—— 提出两种算法来提高 FastPitch （一种可控音高的TTS模型）鲁棒性
3. [Speaking Rate Control of end-to-end TTS Models by Direct Manipulation of the Encoder's Output Embeddings](https://isca-speech.org/archive/interspeech_2022/lenglet22_interspeech.html)——通过无监督端到端 TTS Tacotron2 模型的编码器的嵌入来识别和控制升学参数
4. [TriniTTS: Pitch-controllable End-to-end TTS without External Aligner](https://isca-speech.org/archive/interspeech_2022/ju22_interspeech.html)——韵律建模、音高可控的端到端TTS模型
5. [JETS: Jointly Training FastSpeech2 and HiFi-GAN for End to End Text to Speech](https://isca-speech.org/archive/interspeech_2022/lim22_interspeech.html)——FastSpeech2 + HiFi-GAN 进行联合以实现端到端的TTS

## 自动构音障碍语音识别（ADSR）
1. [Interpretable dysarthric speaker adaptation based on optimal-transport](https://isca-speech.org/archive/interspeech_2022/turrisi22_interspeech.html)——基于最优传输理论，提出无监督、多源、域自适应算法来实现ADSR
2. [Dysarthric Speech Recognition From Raw Waveform with Parametric CNNs](https://isca-speech.org/archive/interspeech_2022/yue22_interspeech.html)——使用参数卷积神经网络对原始波形进行建模，同时也采取了多流声学建模进一步提高性能
3. [The Effectiveness of Time Stretching for Enhancing Dysarthric Speech for Improved Dysarthric Speech Recognition](https://isca-speech.org/archive/interspeech_2022/prananta22_interspeech.html)——研究基于GAN的语言转换，用于强构音障碍语音以改进构音障碍语音识别
4. [Investigating Self-supervised Pretraining Frameworks for Pathological Speech Recognition](https://isca-speech.org/archive/interspeech_2022/violeta22_interspeech.html)——研究自监督预训练框架在病态语言数据集中进行语言识别的性能
5. [Improved ASR Performance for Dysarthric Speech Using Two-stage DataAugmentation](https://isca-speech.org/archive/interspeech_2022/bhat22_interspeech.html)——使用静态和动态数据增强技术来来改进ADSR的性能
6. [Cross-lingual Self-Supervised Speech Representations for Improved Dysarthric Speech Recognition](https://isca-speech.org/archive/interspeech_2022/hernandez22_interspeech.html)——使用从 Wav2Vec、Hubert 和跨语言 XLSR 模型中提取的特征来训练声学模型从而实现自监督的ADSR。

## 用于ASR的神经网络训练方法
1. [Regularizing Transformer-based Acoustic Models by Penalizing Attention Weights](https://isca-speech.org/archive/interspeech_2022/lee22b_interspeech.html)——提出了一种新颖的正则化方法使Transformer模型对输入稀疏性具有鲁棒性。
2. [Content-Context Factorized Representations for Automated Speech Recognition](https://isca-speech.org/archive/interspeech_2022/chan22_interspeech.html)——引入一种无监督的、与编码器无关的方法，用于将语音编码器表征分解为显式内容编码表征和虚假上下文编码标准
3. [Comparison and Analysis of New Curriculum Criteria for End-to-End ASR](https://isca-speech.org/archive/interspeech_2022/karakasidis22_interspeech.html)——将课程学习用于ASR
4. [Incremental learning for RNN-Transducer based speech recognition models](https://isca-speech.org/archive/interspeech_2022/baby22_interspeech.html)——将RNN-T + 增量学习用于真实场景下的ASR
5. [Production federated keyword spotting via distillation, filtering, and joint federated-centralized training](https://isca-speech.org/archive/interspeech_2022/hard22_interspeech.html)——混合集中式联邦学习用于关键字识别

## 声学发音和韵律
1. [Use of prosodic and lexical cues for disambiguating wh-words in Korean](https://isca-speech.org/archive/interspeech_2022/song22b_interspeech.html)——使用韵律和词汇等信息来消除韩语wh词的歧义
2. [Autoencoder-Based Tongue Shape Estimation During Continuous Speech](https://isca-speech.org/archive/interspeech_2022/ribeiro22_interspeech.html)——使用autoencoder来估计声道（估计舌头的形状）
3. [Phonetic erosion and information structure in function words: the case of mia](https://isca-speech.org/archive/interspeech_2022/magistro22_interspeech.html)——研究在语法化过程中，功能词形成过程中的韵律相关性
4. [Dynamic Vertical Larynx Actions Under Prosodic Focus](https://isca-speech.org/archive/interspeech_2022/oh22_interspeech.html)——研究垂直喉部运动和韵律、重音的关系
5. [Fundamental Frequency Variability over Time in Telephone Interactions](https://isca-speech.org/archive/interspeech_2022/bradshaw22_interspeech.html)——研究人们在打电话时语言的F0的变换

## 口语机器翻译
1. [SHAS: Approaching optimal Segmentation for End-to-End Speech Translation](https://isca-speech.org/archive/interspeech_2022/tsiamas22_interspeech.html)——提出监督混合音频分割，可以从任何手动分割的语音语料库中学习最佳分割
2. [M-Adapter: Modality Adaptation for End-to-End Speech-to-Text Translation](https://isca-speech.org/archive/interspeech_2022/zhao22g_interspeech.html)——提出 M-Adapter，基于 Transformer，通过对语音序列的全局和局部依赖性进行建模，产生从语音到文本翻译所需的特征
3. [Cross-Modal Decision Regularization for Simultaneous Speech Translation](https://isca-speech.org/archive/interspeech_2022/zaidi22_interspeech.html)——提出跨模态决策正则化 (CMDR) 通过使用文本到文本翻译 (SimulMT) 任务来改进同步的语音到文本翻译
4. [Speech Segmentation Optimization using Segmented Bilingual Speech Corpus for End-to-end Speech Translation](https://isca-speech.org/archive/interspeech_2022/fukuda22b_interspeech.html)——提出一种语音分割方法使用分好的双语语音语料库训练的二进制分类模型，同时还可以结合 VAD 进行混合分割
5. [Generalized Keyword Spotting using ASR embeddings](https://isca-speech.org/archive/interspeech_2022/r22_interspeech.html)——通过在语音数据库中训练ASR模型得到的中间表征用作关键词检测的声学嵌入，结合三元组损失提高关键词的检测精度

## （多模态）情感语音识别SER-1
1. [Multi-Corpus Speech Emotion Recognition for Unseen Corpus Using Corpus-Wise Weights in Classification Loss](https://isca-speech.org/archive/interspeech_2022/ahn22_interspeech.html)——对语料库进行加权，从而联合多个情感语音识别语料库来预测未知的语料库的SER模型
2. [Improving Speech Emotion Recognition Through Focus and Calibration Attention Mechanisms](https://isca-speech.org/archive/interspeech_2022/kim22d_interspeech.html)——结合多头自注意力使用焦点注意力（FA）机制和新的校准注意力（CA）机制来提高情感语音识别准确率
3. [The Emotion is Not One-hot Encoding: Learning with Grayscale Label for Emotion Recognition in Conversation](https://isca-speech.org/archive/interspeech_2022/lee22e_interspeech.html)——考虑情绪之间的相关性自动构建灰度标签并将其用于学习（通过测量不同情绪的分数来构建灰度标签，而非简单的独热编码）
4. [Probing speech emotion recognition transformers for linguistic knowledge](https://isca-speech.org/archive/interspeech_2022/triantafyllopoulos22b_interspeech.html)——利用语言知识来提高情感语言识别的性能（基于Transformer）
5. [End-To-End Label Uncertainty Modeling for Speech-based Arousal Recognition Using Bayesian Neural Networks](https://isca-speech.org/archive/interspeech_2022/prabhu22_interspeech.html)——端到端的贝叶斯神经网络来捕捉情绪表达的唤醒维度中固有的主观性
6. [Mind the gap: On the value of silence representations to lexical-based speech emotion recognition](https://isca-speech.org/archive/interspeech_2022/perez22_interspeech.html)——通过对语音中的静音段建模来提高SER的性能
7. [Exploiting Co-occurrence Frequency of Emotions in Perceptual Evaluations To Train A Speech Emotion Classifier](https://isca-speech.org/archive/interspeech_2022/chou22_interspeech.html)——认为情绪类别是不独立的，从训练集中的感知评估中计算同时出现的情绪的频率
8. [Positional Encoding for Capturing Modality Specific Cadence for Emotion Detection](https://isca-speech.org/archive/interspeech_2022/dhamyal22_interspeech.html)对Transformer中的每种模态使用单独的位置编码来模拟音频、音素序列、单词序列的模态的情感“节奏”，其效果优于共享编码

## 去混响、降噪、说话人抽取
1. [Speak Like a Professional: Increasing Speech Intelligibility by Mimicking Professional Announcer Voice with Voice Conversion](https://isca-speech.org/archive/interspeech_2022/vuho22_interspeech.html)——提出一种在嘈杂环境中通过对非专业人士的语音应用语音转换方法来增强语音清晰度的方法
2. [Vector-quantized Variational Autoencoder for Phase-aware Speech Enhancement](https://isca-speech.org/archive/interspeech_2022/ho22_interspeech.html)——提出一种相位感知语音增强方法，来估计复数自适应维纳滤波器的幅度和相位
3. [iDeepMMSE: An improved deep learning approach to MMSE speech and noise power spectrum estimation for speech enhancement](https://isca-speech.org/archive/interspeech_2022/kim22i_interspeech.html)——提出一种改进的 DeepMMSE（iDeepMMSE），使用 DNN 估计语音 PSD 和语音存在概率以及先验 SNR，用于语音和噪声 PSD 的 MMSE 估计，最终效果优与DeepMMSE
4. [Boosting Self-Supervised Embeddings for Speech Enhancement](https://isca-speech.org/archive/interspeech_2022/hung22_interspeech.html)——使用用跨域功能来解决自监督学习的嵌入向量可能缺乏细粒度信息来重新生成语音信号的问题
5. [Monoaural Speech Enhancement Using a Nested U-Net with Two-Level Skip Connections](https://isca-speech.org/archive/interspeech_2022/hwang22b_interspeech.html)——提出NUNet-TLS，在大型 U-Net 结构的每一层中的残差 U-Block 之间具有两级skip connection，显著提高了语音增强（SE）的性能
6. [CycleGAN-based Unpaired Speech Dereverberation](https://isca-speech.org/archive/interspeech_2022/muckenhirn22_interspeech.html)——基于 CycleGAN使去混响模型能够在未配对的数据上进行训练，且性能和配对数据训练的模型相当
7. [Attentive Training: A New Training Framework for Talker-independent Speaker Extraction](https://isca-speech.org/archive/interspeech_2022/pandey22_interspeech.html)——提出一种新的训练框架（注意力训练）实现说话人无关的说话人抽取，训练了一个DNN来为第一个说话人创建表征，并利用它从多说话人混合音中提取或跟踪该说话者
8. [Improved Modulation-Domain Loss for Neural-Network-based Speech Enhancement](https://isca-speech.org/archive/interspeech_2022/vuong22_interspeech.html)——通过自监督语音重建任务来学习一组频谱时间感受野用于计算调制域中的加权均方误差，以训练语音增强系统
9. [Perceptual Characteristics Based Multi-objective Model for Speech Enhancement](https://isca-speech.org/archive/interspeech_2022/peng22d_interspeech.html)——提出了一种基于感知特征的多目标语音增强（SE）算法，结合音高和音色相关特征，模型包括基于 LSTM 的 SE 模型和基于 CNN 的多目标模型
10. [Listen only to me! How well can target speech extraction handle false alarms?](https://isca-speech.org/archive/interspeech_2022/delcroix22_interspeech.html)——了解目标语音抽取（TSE） 系统如何处理不活跃说话者（ IS） 案例
11. [Monaural Speech Enhancement Based on Spectrogram Decomposition for Convolutional Neural Network-sensitive Feature Extraction](https://isca-speech.org/archive/interspeech_2022/shi22e_interspeech.html)——根据第一阶段的掩码值将特征图与包含明显语音成分的频谱图组合在一起，将大于特定阈值的掩码对应的位置提取为特征图，从而使 CNN 对输入特征敏感
12. [Neural Network-augmented Kalman Filtering for Robust Online Speech Dereverberation in Noisy Reverberant Environments](https://isca-speech.org/archive/interspeech_2022/lemercier22_interspeech.html)——提出了一种基于加权预测误差 (WPE) 方法的卡尔曼滤波变体的抗噪声在线去混响神经网络增强算法

## 声源分离2
1. [PodcastMix: A dataset for separating music and speech in podcasts](https://isca-speech.org/archive/interspeech_2022/schmidt22_interspeech.html)——数据集，分离播客中的音乐和语言
2. [Independence-based Joint Dereverberation and Separation with Neural Source Model](https://isca-speech.org/archive/interspeech_2022/saijo22_interspeech.html)——提出了一种具有神经源模型的基于独立的联合去混响和分离方法
3. [Spatial Loss for Unsupervised Multi-channel Source Separation](https://isca-speech.org/archive/interspeech_2022/saijo22b_interspeech.html)——提出了一种用于无监督多通道源分离的空间损失，利用了波达方向 (DOA) 和波束成形的对偶性
4. [Effect of Head Orientation on Speech Directivity](https://isca-speech.org/archive/interspeech_2022/bellows22_interspeech.html)——研究头部方向对语音方向性的影响，低频下的头部方向和身体衍射对方向性的影响很小
5. [Unsupervised Training of Sequential Neural Beamformer Using Coarsely-separated and Non-separated Signals](https://isca-speech.org/archive/interspeech_2022/saijo22c_interspeech.html)——提出了一种使用粗分离和非分离监督信号的顺序神经波束形成器 (Seq-BF) 的无监督训练方法
6. [Blind Language Separation: Disentangling Multilingual Cocktail Party Voices by Language](https://isca-speech.org/archive/interspeech_2022/borsdorf22_interspeech.html)——引入盲语分离 (BLS)任务，通过语言解开多种语言的重叠声音
7. [NTF of Spectral and Spatial Features for Tracking and Separation of Moving Sound Sources in Spherical Harmonic Domain](https://isca-speech.org/archive/interspeech_2022/guzik22_interspeech.html)——提出一种新颖的基于非负张量分解 (NTF) 的方法来跟踪和分离移动声源
8. [Modelling Turn-taking in Multispeaker Parties for Realistic Data Simulation](https://isca-speech.org/archive/interspeech_2022/deadman22_interspeech.html)——提出了一种基于有限状态的生成方法，该方法对语音语料库中的时间信息进行训练
9. [An Initialization Scheme for Meeting Separation with Spatial Mixture Models](https://isca-speech.org/archive/interspeech_2022/boeddeker22_interspeech.html)——展示了通常只有一个说话者处于活动状态的情况下可以用于巧妙地初始化使用时变类先验的空间混合模型（SMM）
10. [Prototypical speaker-interference loss for target voice separation using non-parallel audio samples](https://isca-speech.org/archive/interspeech_2022/mun22_interspeech.html)——提出了一种原型说话人干扰 (PSI) 损失，它利用来自目标说话人、干扰说话人以及干扰噪声的代表性样本，以更好地利用任何可能可用的非并行数据

## 用于说话人识别的嵌入和网络架构（略）

## 语音表征2
1. [SpeechFormer: A Hierarchical Efficient Framework Incorporating the Characteristics of Speech](https://isca-speech.org/archive/interspeech_2022/chen22_interspeech.html)——提出了一种层次化的高效框架 SpeechFormer，它考虑了语音的结构特征，可以作为认知语音信号处理的通用主干
2. [VoiceLab: Software for Fully Reproducible Automated Voice Analysis](https://isca-speech.org/archive/interspeech_2022/feinberg22_interspeech.html)——自动声学分析并自动记录分析参数软件
3. [TRILLsson: Distilled Universal Paralinguistic Speech Representations](https://isca-speech.org/archive/interspeech_2022/shor22_interspeech.html)——通过自监督学习和副语言语音模型进行知识提炼来实现模型
4. [Global Signal-to-noise Ratio Estimation Based on Multi-subband Processing Using Convolutional Neural Network](https://isca-speech.org/archive/interspeech_2022/li22b_interspeech.html)——提出了一种基于多子带的 gSNR 估计网络（MSGNet），将嘈杂的语音波形分割成巴克尺度子带，CNN  用于学习非线性函数
5. [A Sparsity-promoting Dictionary Model for Variational Autoencoders](https://isca-speech.org/archive/interspeech_2022/sadeghi22_interspeech.html)——通过稀疏促进字典模型来构造潜在空间，该模型假设每个latent code都可以编写为字典列的稀疏线性组合
6. [Deep Transductive Transfer Regression Network for Cross-Corpus Speech Emotion Recognition](https://isca-speech.org/archive/interspeech_2022/zhao22h_interspeech.html)——提出深度转导转移回归网络（DTTRN）学习一个语料库不变的深度神经网络来桥接源和目标语音样本及其标签信息
7. [Audio Anti-spoofing Using Simple Attention Module and Joint Optimization Based on Additive Angular Margin Loss and Meta-learning](https://isca-speech.org/archive/interspeech_2022/hansen22_interspeech.html)——提出了一种基于加权加性角边距损失的二元分类联合优化方法，使用元学习训练框架来开发一个高效的系统，且对各种欺骗攻击具有鲁棒性
8. [PEAF: Learnable Power Efficient Analog Acoustic Features for Audio Recognition](https://isca-speech.org/archive/interspeech_2022/bergsma22_interspeech.html)
9. [Hybrid Handcrafted and Learnable Audio Representation for Analysis of Speech Under Cognitive and Physical Load](https://isca-speech.org/archive/interspeech_2022/elbanna22_interspeech.html)——用于语音压力检测的数据集
10. [Generative Data Augmentation Guided by Triplet Loss for Speech Emotion Recognition](https://isca-speech.org/archive/interspeech_2022/wang22w_interspeech.html)——利用由三元网络引导的基于 GAN 的增强模型，在训练数据不平衡和不足的情况下提高 SER 性能
11. [Learning neural audio features without supervision](https://isca-speech.org/archive/interspeech_2022/yadav22_interspeech.html)——结合自监督学习和可学习的时频表征神经模块
12. [Densely-connected Convolutional Recurrent Network for Fundamental Frequency Estimation in Noisy Speech](https://isca-speech.org/archive/interspeech_2022/zhang22ca_interspeech.html)——将 F0 估计视为多类分类问题，并训练频域密集连接卷积神经网络 (DC-CRN) 从嘈杂的语音中估计 F0
13. [Predicting label distribution improves non-intrusive speech quality estimation](https://isca-speech.org/archive/interspeech_2022/faridee22_interspeech.html)——研究了几种整合MOS的方法以提高 MOS 估计性能
14. [Deep versus Wide: An Analysis of Student Architectures for Task-Agnostic Knowledge Distillation of Self-Supervised Speech Models](https://isca-speech.org/archive/interspeech_2022/ashihara22_interspeech.html)——研究应用知识蒸馏 (KD) 等压缩方法来生成紧凑的自监督学习模型的情况下，模型架构（深度还是宽度）对下游语音任务的性能的影响
15. [Dataset Pruning for Resource-constrained Spoofed Audio Detection](https://isca-speech.org/archive/interspeech_2022/azeemi22_interspeech.html)——提出一个新的度量标准-忘记范数，进一步提高反欺骗模型在修剪数据上的性能

## 语音合成：语言处理、范式和其他主题 II（略）
## 语音识别中的其他主题（略）
## 鲁棒说话人识别
1. [Extended U-Net for Speaker Verification in Noisy Environments](https://isca-speech.org/archive/interspeech_2022/kim22b_interspeech.html)——出了一个基于 U-Net 的集成框架，可以同时优化说话人识别和特征增强损失
2. [Domain Agnostic Few-shot Learning for Speaker Verification](https://isca-speech.org/archive/interspeech_2022/yang22j_interspeech.html)——提出一个few-shot泛化框架，该框架学习解决新用户和新域的分布变化
3. [Scoring of Large-Margin Embeddings for Speaker Verification: Cosine or PLDA?](https://isca-speech.org/archive/interspeech_2022/wang22r_interspeech.html)——研究最适合说话人验证的评分后端
4. [Training speaker embedding extractors using multi-speaker audio with unknown speaker boundaries](https://isca-speech.org/archive/interspeech_2022/stafylakis22_interspeech.html)——使用弱注释训练说话人嵌入提取器的方法：通过将不需要训练或参数调整的基线说话人分类算法、具有分段聚合的修改损失和两阶段训练方法相结合，我们能够训练一个有竞争力的基于 ResNet 的嵌入提取器
5. [Investigating the contribution of speaker attributes to speaker separability using disentangled speaker representations](https://isca-speech.org/archive/interspeech_2022/luu22_interspeech.html)
6. [Joint domain adaptation and speech bandwidth extension using time-domain GANs for speaker verification](https://isca-speech.org/archive/interspeech_2022/kataria22_interspeech.html)——学习将窄带对话电话语音映射到宽带麦克风语音，利用配对和非配对数据开发了并行和非并行学习解决方案
## 语音生成（略）
## 语音质量评估（略）
## ASR 的语言建模和词法建模
1. 

## 声源分离1
1. [A Hybrid Continuity Loss to Reduce Over-Suppression for Time-domain Target Speaker Extraction](https://isca-speech.org/archive/interspeech_2022/pan22b_interspeech.html)——提出一种用于时域说话人提取算法的混合连续性损失函数，在频域中引入了多分辨率 delta 频谱损失，以解决过度抑制问题
2. [Extending GCC-PHAT using Shift Equivariant Neural Networks](https://isca-speech.org/archive/interspeech_2022/berg22_interspeech.html)——提出一种扩展广义互相关与相位变换 (GCC-PHAT)的方法，将收到的信号使用移位等变神经网络进行过滤
3. [Heterogeneous Target Speech Separation](https://isca-speech.org/archive/interspeech_2022/tzinis22_interspeech.html)——引入一种用于单通道目标源分离的方法，使用非互斥概念来区分感兴趣的源
4. [Separate What You Describe: Language-Queried Audio Source Separation](https://isca-speech.org/archive/interspeech_2022/liu22w_interspeech.html)——介绍了语言查询音频源分离 (LASS) 的任务，提出 LASS-Net 联合处理声学和语言信息，并将与语言查询一致的目标源从音频混合中分离出来
5. [Implicit Neural Spatial Filtering for Multichannel Source Separation in the Waveform Domain](https://isca-speech.org/archive/interspeech_2022/markovic22_interspeech.html)——提出一个单级随机波形到波形的多通道模型，可以根据动态声学场景中的广泛空间位置分离移动声源

## ASR 技术和系统
1. [End-to-end Speech-to-Punctuated-Text Recognition](https://isca-speech.org/archive/interspeech_2022/nozaki22_interspeech.html)——提出端到端模型将语音作为输入，输出带标点的文本
2. [End-to-End Dependency Parsing of Spoken French](https://isca-speech.org/archive/interspeech_2022/pupier22_interspeech.html)——引入 wav2tree 端到端的[依赖解析](https://zhuanlan.zhihu.com/p/198207874)模型来进行句法分析
3. [Turn-Taking Prediction for Natural Conversational Speech](https://isca-speech.org/archive/interspeech_2022/chang22_interspeech.html)——提出端到端的模型用于自然情况下的轮流会话查询语音识别
4. [Streaming Intended Query Detection using E2E Modeling for Continued Conversation](https://isca-speech.org/archive/interspeech_2022/chang22b_interspeech.html)——提出一种流式端到端 (E2E) 预期查询检测器，可识别指向设备的话语并过滤掉其他不指向设备的话语
5. [Exploring Capabilities of Monolingual Audio Transformers using Large Datasets in Automatic Speech Recognition of Czech](https://isca-speech.org/archive/interspeech_2022/lehecka22_interspeech.html)——展示了从包含超过 8 万小时未标记语音的大型数据集预训练捷克单语音频转换器方面取得的进展，结果表明单语 Wav2Vec 2.0 模型是强大的 ASR 系统
6. [SVTS: Scalable Video-to-Speech Synthesis](https://isca-speech.org/archive/interspeech_2022/schoburgcarrillodemira22_interspeech.html)——视频到语音合成（通过口型合成语音），引入一个可扩展的视频到语音框架，包括一个视频到频谱图的预测器和一个预训练的神经声码器

## 语音感知
1. [One-step models in pitch perception: Experimental evidence from Japanese](https://isca-speech.org/archive/interspeech_2022/kishiyama22_interspeech.html)——研究虚元音感知
2. [Generating iso-accented stimuli for second language research: methodology and a dataset for Spanish-accented English](https://isca-speech.org/archive/interspeech_2022/perezramon22_interspeech.html)——数据集
3. [Factors affecting the percept of Yanny v. Laurel (or mixed): Insights from a large-scale study on Swiss German listeners](https://isca-speech.org/archive/interspeech_2022/leemann22_interspeech.html)——研究人们对同一个音频片段的不同感知问题
4. [Effects of laryngeal manipulations on voice gender perception](https://isca-speech.org/archive/interspeech_2022/zhang22ba_interspeech.html)——喉部对语言性别感知的影响
5. [Why is Korean lenis stop difficult to perceive for L2 Korean learners?](https://isca-speech.org/archive/interspeech_2022/lee22n_interspeech.html)——韩语学习感知
6. [Lexical stress in Spanish word segmentation](https://isca-speech.org/archive/interspeech_2022/zurita22_interspeech.html)——研究词汇重音在分词中的作用

