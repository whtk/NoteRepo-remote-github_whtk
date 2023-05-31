> 2019 年

1. 提出 GTP-2，1.5B 参数的 Transformer 模型
2. 做了一个新的数据集：WebText
3. 主要的优势是 zero shot

## Introduction
1. 目前的AI系统只是在某个特定领域的专家，而非所谓的通用性AI，作者希望转向更为通用的系统，可以实现多个任务而无需为每个子任务创建特定的标注数据集
2. 怀疑是在单个领域的数据集进行训练导致的模型泛化性差
3. 多任务学习可以改善性能并且有很大的前景，但是在 NLP 用的不多
4. 本文将多任务和自监督+fine tune的模式结合起来，探索更为通用的迁移方法，可以在 zero shot 的情况下实现下游任务，且不需要进行任何架构或者参数方面的修改（也就是不需要下游任务数据集，不进行 fine tune），最终得到了 “还可以” 的结果

## 方法

核心是语言模型，从一系列的样本 $\left(x_1, x_2, \ldots, x_n\right)$ 中进行无监督分布估计，每个样本都包含可变长度的符号 $\left(s_1, s_2, \ldots, s_n\right)$，则：$$p(x)=\prod_{i=1}^n p\left(s_n \mid s_1, \ldots, s_{n-1}\right)$$
系统的输出不应该仅仅取决于输入，还应该和任何相关，即我们需要建模 $p(\text { output } \mid \text { input,task) }$，其中的任务条件，通常可以在架构级实现或者在算法级实现，但是，语言文字本身就可以当作这样一种条件，例如：
+ 对于翻译任务，可以将序列写成：translate to french, english text, french text
+ 对于 阅读理解任务，可以写成：answer the question, document, question, answer

作者推测，一个足够强大的语言模型可以学习去 推理和执行 语言文本序列中的任务，以便更好地进行预测，而无论其采用的方法。做到这一点就可以实现无监督的多任务学习。作者通过分析语言模型在各种任务的 zero shot 下的性能来进行测试。

### 训练数据集

GPT-2 使用了尽可能大且多样的数据集，从而可以用在尽可能多的领域和任务中。

采用网络爬虫，只爬取人类策划和过滤的网页来提高数据质量，最终得到 WebText 数据集， 经过清理之后得到超过 800万 个文档，总计 40G 的数据。

### 输入表征

1. 作者发现，标准的 byte level 的 LM 性能不如 word level LM
2. 于是采用 BPE 作为输入来减少 vocabulary 的大小，但是由于 BPE 使用基于贪婪频率的启发式方法来构建 vocabulary，直接将 BPE 应用于字节序列会导致次优合并，于是禁止 BPE在任何字节序列的字符类别之间合并
3. 从而可以将 word level 和 byte level 的优点相结合，且可以为任意 Unicode 编码的字符串分配概率而不用管前面的预处理、tokenization等

### 模型

基于 Transformer架构，和 OpenAI 的 GPT 差不多，修改在于 Layer normalization 被放到每个 sub-block 的输入之前，且在最后的注意力层之后添加了一个额外的 Layer normalization 层。

初始化的时候采用了修正的初始化方法，将上下文的 token 数从 512 增加到 1024，batch size 增加到 512。

## 实验（略）

![[Pasted image 20230302162510.png]]
参数最小的相当于原始的 GPT，参数最大的那个称为 GPT-2。