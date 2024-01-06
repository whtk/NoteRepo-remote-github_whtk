
> 2014 年google的论文，提出了 d-vector，本质就是多层感知机。

1. 研究 DNN 在 small footprint text-dependent speaker verification 任务中的应用
2. 在 development stage，训练 DNN 进行说话人分类（frame level）
3. 在 enrollment 阶段，从训练好的 DNN 的最后一层提取 features，做平均之后得到所谓的 d-vector,
4. 在 evaluation stage，通过比较 d-vector 做决策
5. 相比于 i-vector 实现了较好的性能，并且对噪声更为鲁棒

## Introduction 

1. 之前的系统都是基于 i-vector 和 PLDA balabala
2. 本文采用 DNN 提取说话人特征，称为 d-vector

## 之前的工作（略）

## 基于 DNN 的 SV

提出的模型如图：![](./image/Pasted%20image%2020221227104703.png)

### DNN 作为特征提取器

构建了一个在 frame level 工作的 DNN 来分类 development set 中的说话人。

训练时将每个 frame 左右的上下文 frames 进行 stack，输出的数量对应于set中speaker 的数量 $N$。target label 是一个 one hot label。

完成训练后，将最后的隐藏层的激活输出作为 说话人表征，也就是所谓的 d-vector（不是 softmax 的输出）。

DNN 训练完成后，最后一层隐藏层的输出学习到了说话人的紧凑表征，并且可以泛化到未知说话人。

### Enrollment 和 evaluation

给定一组来自说话人 $s$ 的 utterance $X_s=\left\{O_{s_1}, O_{s_2}, \ldots, O_{s_n}\right\}$ 和 observations $O_{s_i}=\left\{o_1, o_2, \ldots, o_m\right\}$，enrollment 过程表述如下：

把每个 observation $o_j$ 及其上下文输入到 DNN 模型中，最后一层隐藏层的输出进行 L2 归一化，然后把所有 $O_{s_i}$ 中的结果进行累计，把最后累计的结果称为 d-vector。

最后求出所有的 utterance 的 d-vector 然后进行平均得到说话人表征。

### DNN 训练过程

使用 dropout，训练 DNN 为 maxout DNN。

其他具体参数略。


## 实验 & 结果（略）