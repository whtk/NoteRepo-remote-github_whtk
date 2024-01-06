1. 本文提出了一种新的语音反欺诈模型：ResWavegram Resnet（RWResnet），模型包含两个部分，Conv1D Resblocks 和 backbone Resnet34。
	1. Conv1D Resblocks 输入为原始波形，输出为 ResWavegram，相比于传统方法，ResWavegram 保留了音频信号的所有信息，具有更强的特征提取能力
	2. 第二部分中，提取的特征输入 Resnet34 网络进行二分类
2. 在 ASVspoof2019 LA 中进行评估，比其他模型性能更好


## Introduction
1. 很多研究者直接使用原始波形作为输入。如 [[End-to-End Anti-Spoofing with RawNet2 笔记]]、[[Replay attack detection with complementary high-resolution information using end-to-end DNN for the ASVspoof 2019 Challenge 笔记]] 在重放攻击中应用修改的 CNN-GRU，但是以上方法无法与传统特征进行比较，因为只使用了一维卷积。
2. Wavegram 是一种基于神经网络的时频表示，类似于Melspectrogram，基于此的模型 Wavegram CNN和Wavegram-Logmel CNN 也显示出强大的竞争力。
3. 本文贡献如下：
	1. 使用 Conv1D块从原始波形中获得二维波形特征
	2. 提出了一种新的反欺骗模型，名为ResWavegram Resnet，进行端到端训练，效果超过了最先进的反欺骗方法。


## 模型结构
整个模型包含两个部分：特征提取（Wavegram或ResWavegrams）和主干网络（Resnet34）。

训练用的音频长度为固定 8s（剪切拼接），同时波形数值进行归一化（除以32768）。

### Wavegram
由于时域CNN模型无法捕获音频信号中的频率关系，因此从基于一维卷积的构建特征频率非常重要。下表给出了波形图提取结果：
![[Pasted image 20221024094006.png]]
图中关键是 Conv1d 块，同时包含 Maxpooing 层对输入波形进行下采样。

输出结果的维度从原始的 $1\times 12800$ 变成 $C_3 \times 400(T)$，然后 reshape 到 $\frac CF \times T \times F$ ，即将 $C_3$ 分成了 $\frac CF$ 组，每组都有 $F$ 个 frequency bin，最终每个音频被转换为 $C_g \times T \times F$ 大小的特征图，然后输入到 Resnet34 中进行分类。

### ResWavegram
受 Resnet 网络的影响，在每个 Conv1D 之间加入一个 ResBlock，修改后如下图 a2 所示。
![[Pasted image 20221024100312.png]]
且添加的 ResBlock 由 Conv1d 和 BN 层组成。其中，Conv1D(1) 和 Conv1D(3) 的 dilation 为1，Conv1D(2) 为2，kernel size 都为3，在每个 ResBlock 中，感受野由于 kernel size = 3 而变大。BN 层则用来加速训练过程和减缓过拟合。

所以 ResWavegram = Wavegram - Conv1D block + Conv1D Resblock。通过残差连接特征提取过程中感知信息范围加深。梯度也能更好地传播。

### Resnet
1. Resnet 广泛用于语音和反欺诈中。
2. 采用了论文 "Deep residual learning for image recognition" 论文中的resnet结构，但是通道数改为原来的 1/4，在池化层之后添加了两个全连接层，结构如下表：![[Pasted image 20221024101423.png]]
3. 池化层和F2之间还有 skip connection，确保梯度传递。F1 之后还有一个 RELU。


## 实验
数据集：ASVspoof 2019 LA

将训练和开发集用于联合训练，评估集用于测试评估。

参数：
+ epoch : 50
+ batch_size : 16
+ Adam 优化器
+ lr : 1e-4
+ weight_decay : 0
+ ce 损失
+ 余弦热退火重启算法进行梯度更新
+ Kaiming 初始化

评估指标：
+ EER
+ t-DCF
+ 得分计算采用对数似然比：$\operatorname{score}(U)=\log (p(\text { bonafide } / U ; \theta))-\log (p(\text { spoof } / U ; \theta))$

## 结果
和 baseline 以及其他模型的对比：
![[Pasted image 20221024102331.png]]
同时设置不同的通道组合进行对比：
![[Pasted image 20221024102601.png]]
可以看到 ，$C_g=1$ 的效果始终最好，设置 $C_g$ 再次进行结果比较：
![[Pasted image 20221024102850.png]]
ResWavegram 获得了更好的性能。

