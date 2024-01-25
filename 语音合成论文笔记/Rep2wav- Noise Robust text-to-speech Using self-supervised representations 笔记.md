> preprint，腾讯 AI LAB，中科大、南洋理工

1. 从真实场景下收集的数据通常包含噪声，需要通过增强模型进行降噪。噪声鲁棒的 TTS 则通常使用增强后的语音进行训练，会受语音失真和背景噪声影响
2. 自监督预训练模型有很好的噪声鲁棒性，提出采用预训练的模型来提高 TTS 的噪声鲁棒性
3. 基于 HiFi-GAN 提出 representation-to-waveform vocoder，学习将预训练模型表征映射到波形
4. 基于 FastSpeech2 提出 text-to-representation 模型，学习将文本映射到预训练模型表征
5. 在 LJSpeech 和 LibriTTS 数据集实验结果表明，相比于语音增强方法效果更好

> 就是 HiFi-GAN + FastSpeech2，然后 intermediate feature 改为 自监督表征。。。

## Introduction

1. 用带噪语音训练 TTS 模型的主流方法是采用语音增强模型降噪，然后用增强后的语音训练；也有直接使用带噪数据来训练的
2. 这些模型用的都是 mel 谱特征
3. 自监督模型在 ASR 效果很好，也可以提高 ASR 模型在噪声场景下的准确率
4. 本文第一个提出使用自监督模型来提高 TTS 模型的噪声鲁棒性
5. 结果表明：
	1. 表征的层数越高，上下文信息越多，模型的噪声鲁棒性越好，但说话人信息丢失
	2. 采用不同层的平均表征可以平衡模型的噪声鲁棒性和说话人信息
	3. 基于表征的 TTS 模型在性能优于基于 mel 谱的 TTS 模型

## 方法

### Representation-to-waveform: Vocoder

vocoder 基于 [HiFi-GAN- Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis 笔记](HiFi-GAN-%20Generative%20Adversarial%20Networks%20for%20Efficient%20and%20High%20Fidelity%20Speech%20Synthesis%20笔记.md)，包含一个 generator 和两个 discriminator（ multi-scale 和 multi-period discriminator），generator 和 discriminator 都采用多层卷积网络：
+ generator 采用多层转置卷积，输入为预训练模型不同层的表征，然后上采样到原始波形的维度
+ discriminator 用于区分语音信号中不同周期的信号模式

训练过程如下图 a：
![](image/Pasted%20image%2020240125102441.png)

选择来自其他领域的公开的多说话人 clean speech 数据集，语音 $x$ 输入到预训练模型中提取不同层的表征 $c$，然后输入到 vocoder 中重构 clean waveform。给定 generator $G$ 和 discriminator $D$，vocoder 的损失函数为：
$$\begin{gathered}\mathcal{L}_G=\mathcal{L}_{adv}(G;D)+\alpha\mathcal{L}_{fm}(G;D)+\beta\mathcal{L}_{mel}(G),\\\mathcal{L}_D=L_{adv}(D;G)\end{gathered}$$
其中生成损失 $\mathcal{L}_{adv}(G;D)$ 和判别损失 $\mathcal{L}_{adv}(D;G)$ 分别为：
$$\begin{aligned}\mathcal{L}_{adv}(D;G)&=\mathbb{E}_{(x,c)}\left[\left(D(x)-1\right)^2+\left(D(G(c))\right)^2\right],\\\mathcal{L}_{adv}(G;D)&=\mathbb{E}_{(c)}\left[\left(D(G(c))-1\right)^2\right]\end{aligned}$$
其中特征匹配损失 $\mathcal{L}_{fm}(G;D)$ 和 mel 谱损失 $\mathcal{L}_{mel}(G)$ 与 HiFi-GAN 中的保持一致，$\alpha$ 和 $\beta$ 是超参数。

### Text-to-representation: FastSpeech 2

采用 FastSpeech 2 [FastSpeech 2- Fast and High-Quality End-to-End Text to Speech 笔记](FastSpeech%202-%20Fast%20and%20High-Quality%20End-to-End%20Text%20to%20Speech%20笔记.md) 来学习文本到表征的映射。FastSpeech 2 包含 phone embedding、encoder、variance adaptor 和 decoder 模块：
+ encoder 采用多层 feed-forward transformer，将 phoneme 序列转为 hidden state 序列
+ variance adaptor 采用多层卷积网络预测 duration、pitch、energy 等
+ decoder 采用线性投影层，将网络输出映射到表征

训练过程如上图中的 b：
+ 将带噪语音输入到 speech enhancement model 中得到增强语音，然后输入到预训练模型中提取表征
+ 使用文本和增强表征训练 FastSpeech 2 模型
+ 训练后，输入文本，连接 FastSpeech 2 和 vocoder 模型来合成语音波形（如上图 c）

## 实验（略）
