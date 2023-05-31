1. 提出 Mockingjay，采用双向 Transformer Encoder 在大量未标记语音中进行预训练，通过联合过去和未来，来预测当前
2. 能够在音素分类、说话人识别和口语内容情感分类等任务中提高性能
3. 在只有 0.1% 标记数据的低资源中，其表现优于使用所有的 Mel 特征的结果

> 其实这篇文章可以说是直接把 BERT 用于 音频特征，区别在于，BERT 输入的是 token，而音频的输入一般是帧（注意，这里的 帧并不是 raw waveform，其实还是 mel spectrogram），但是 token 是会被转换成 embedding的，而 frame 在这里就可以直接看成是 embedding了，比 BERT 更简单。

## Introduction

1. 提出 Mockingjay，基于 Masked Acoustic Model 进行特征学习，在训练的时候，mask 一定数量的帧，模型重构和预测原始的帧，结构如图：![[image/Pasted image 20230324194516.png]]

## 相关工作

1. CPC 和 wav2vec 使用多层 CNN 编码过去的文本，通过在预测隐空间中预测未来以学习语音表征，这个过程是基于一个对比的二分类任务
2. APC 采用自回归模型编码过去的声学序列的时间信息，通过类似于 RNN 的方法来预测未来
3. 这些模型都是单方向的，但是这种约束限制了模型的能力
4. vq-wav2vec 在连续语音中引入 BERT，输入语音离散化为 k-way 的量化 embedding 空间，然后将这些离散值看成是 NLP 中的 token
5. 和 vq-wav2vec 不同，本文可以看成是一个 BERT 的修改版，可以直接用在连续语音中（没有量化的过程）
6. 提出的方法在多个任务的 fine tune 的效果都很好，而且在低资源下提升也很大。

## 方法
![[image/Pasted image 20230325100820.png]]
> 这张图应该从右边的 Real Frame 开始向下，然后看左边的模型，最后到右上角的 predicted frame，然后计算损失。

### 模型架构

采用多层 Transformer Encoder 和 MHA 进行双向编码，每个 encoder layer 包括：
+ MHA
+ FF

每层的输出维度都相同，记为 $H_{dim}$，FF 层的大小为 $F_{dim}$，attention head 为 $A_{num}$，总层数为 $L_{num}$，最终提取的表征是图中的 $Hidden$。

由于模型不包含回归结构，所以使用 positional encoder 编码位置信息，但是直接相加不太好，所以先将输入帧投影到 $H_{dim}$ 维再相加。对输入特征进行了下采样使其适用于长序列，将 $R_{factor}$ 个连续的帧 stack 成一个。

### 掩码声学模型（MAM）

随机选择 15% 的输入帧进行 mask，训练时添加一个 prediction head（两层 FF+layer norm），其输入为最后一层的 encoder，使用 L1 loss 来最小化预测和GT之间的重构误差（只在被 mask 的那 15% 的帧上进行），模型训练完之后就不用 prediction head 了。

训练时，对于选择的那 15% 的帧，有 80% 的次数将 mask 为0，有 10% 的次数替换为随机帧，有10% 的次数保持不变。这里和 BERT 不一样，BERT 是在第 $i$ 个 token 上进行随机，而 Mockingjay 则是在 utterance 上进行随机的，也就是他 mask 的时候的那些概率是指整段音频的。
> 更直白地说，有 80% 的音频在 mask 的时候是 mask 0，有 10% 的音频是随机mask，有 10% 的音频不进行 mask，这个时候 training 和 valid 是匹配的。

为了避免模型学习插值，提出额外的连续 masking，mask 连续的 $C_{num}$ 帧为0，同时采用动态 masking，mask 的值是从均匀分布中随机选择的（而不是在特征处理阶段就确定的）。

BERT 里面还有 sentence prediction 的任务，但是发现其效果不好甚至可能变坏，所以还是没用。

### 结合下游任务

可以从最后一层提取表征，也可以将内部层的特征进行加权混合。fine tune的时候，Mockingjay 和 下游模型一起更新。

## 实现

采用了两种特征，Mel-scale spectrogram 和 linear-scale spectrogram，有两个模型，Base 和 Large，两个模型的 $H_{dim}=768, F_{dim}=3072,A_{num}=12$，但是 $L_{num}, R_{factor}, C_{num}$ 不同：![[image/Pasted image 20230325104905.png]]

模型在 LibriSpeech  train-clean-360 的数据集上进行预训练，使用 Adam，训练 500k step，前 7% 的 step 中 lr warm up 到 4e-4，然后线性衰减。所有的层使用 0.1 的 dropout，下游任务 fine tune 的时候，lr 变成 4e-3，训练 2 个 epoch（50K  step），batch size 为 6，在单卡 1080 上训练。 

## 实验

发现随机初始化的模型+下游模型很难从零开始训练，这说明预训练过程非常重要。

然后在因素分类、说话人识别 和 语义分类任务中进行了实验。具体结果参看原论文。