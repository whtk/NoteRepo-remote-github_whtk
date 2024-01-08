> IJCAI 2022，renyi，ZJU

1. NAR-TTS 通常采用 phoneme 序列作为输入，无法理解树结构的语义信息（tree-structured syntactic informatio）
2. 提出 SyntaSpeech，一种 syntax-aware 的 light-weight NAR-TTS 模型，将树结构的语义信息集成到 PortaSpeech 的韵律建模中：
	1. 基于输入句子的依赖建立语法图，然后采用语法图来进行 text encoding 来提取语义信息
	2. 将提取到的 syntactic encoding 引入 PortaSpeech 提高韵律预测
	3. 引入 multi-length discriminator 来替换 PortaSpeech 中的 fow-based post-net
3. 不仅可以合成 expressive prosody 的音频，而且可以泛化到多语种、多说话人 TTS

## Introduction

音频和依赖树的关系如下：
![](image/Pasted%20image%2020240108111205.png)

之前的 syntax-aware 的 TTS 都是在 AR-TTS 下的，从而可以把语义信息作为一个额外的特征引入，而 NAR-TTS 通常采用外部的 predictor 提取 prosody，目前没有 NAR-TTS 可以有效地利用树结构的语义信息来提高韵律。

提出 SyntaSpeech，采用 graph encoder 从文本中发掘依赖关系：
+ 基于依赖树为输入的句子建立语法树，将 phoneme-level latent encoding 表征到图中的 word node 来得到 word level 的语义编码，然后用 graph encoder 进行聚合
+ 将 graph encoder 引入 PortaSpeech，引入到 duration predictor 和variational generator 中，来分别提升其建模能力
+ 采用 multi-length adversarial training 来替代 ﬂow-based post-net 实现轻量化的架构

## 相关工作（略）

## SyntaSpeech

结构如图：
![](image/Pasted%20image%2020240108113132.png)

图 a，采用 syntactic graph encoder 来得到语义信息用于 duration 预测和其他韵律分布建模，步骤如下：
+ 文本通过基于 transformer 的 phoneme encoder 得到 phoneme encoding，然后基于 单词边界 通过 平均池化 得到 word-level representation
+ syntactic graph builder 采用依赖关系建立语法图，然后 word encoding 通过采用 gated graph convolution 构建的图进行聚合
+ 得到的 word-level syntactic encoding 拓展到 phoneme level 和 frame level，分别用于嵌入到 duration prediction 和pitch-energy prediction

### 基于依赖关系的语法树

Dependency parse tree 可以视为有向图，每个边都表示两个节点（word）之间的依赖关系，引入  syntactic graph builder 将 dependency tree 转为 syntactic graph。

但是原始的依赖图是单向的，从而叶子结点在聚合是不能在其他节点中获得信息，于是为每条边增加了一个 reverse edge，使得图中的信息流是双向的。

下面分别介绍不同语言下，用 node embedding 构造语义图的过程。

对于英语，在双向图中添加 BOS 和 EOS，分别将其和第一个和最后一个单词连接，一个示例如下：
![](image/Pasted%20image%2020240108115147.png)
其中，前向边为实线，后向边为虚线，通过 word-level average pooling 得到 word-level node embedding。

对于中文，并没有提取 word-level encoding，而是采用 character-level average pooling 得到 character encoding，然后拓展语法图，将 word node 拓展到 character node，把每个 word 的第一个 character 和其他内部的 character 进行依赖连接，其他 character 则根据顺序依次连接，相当于增加了两个额外的边来表示 intra-word 连接（下图绿色部分的线），如图：
![](image/Pasted%20image%2020240108171222.png)

### 用于韵律预测的 Syntax-Aware Graph Encoder

为了从文本中学习 syntax-aware word representation，设计了一个 syntactic graph encoder，如上图 b。

前面得到语义图之后，phoneme embedding 通过 word-level average pooling 得到 图中的 node embedding，然后通过以下方式的图聚合得到语义信息：
+ 采用两层的 Gated Graph Convolution layers 提取图中的长时依赖
+ 把前面得到的所有的输出求和，得到最终的 syntactic word-level encoding

然后 PortaSpeech 包含：
+ Transformer-based linguistic encoder 来提取 frame-level semantic representations
+ VAE-based variational generator 来合成 mel 谱

且韵律预测分为两个任务：
+ linguistic encoder 中的 duration predictor 用于控制时间
+ variational generator 用于生成 pitch 和 energy

那么，提取得到的 syntactic word encoding 一方面会拓展到 phoneme level 然后输入到 duration predictor，另一方面拓展到 frame level 作为 flow 的额外特征。

### Multi-Length Adversarial Training

简单来说，就是输入 discriminator 的那段音频的 window length 是随机选取的（而非之前固定长度的）。

## 实验（略）