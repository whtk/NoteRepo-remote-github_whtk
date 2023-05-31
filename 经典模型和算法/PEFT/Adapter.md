> ICML 2021

1. fine tune 预训练模型在很多下游任务中效率通常都很低，因为需要为每个任务独立训练一个全新的模型
2. 提出使用 adapter 模块进行迁移学习，即在原模型的参数保持不变的条件下，对于每个任务只要添加很少的可训练的参数
3. 将 BERT 模型迁移到 26 个不同的文本分类任务中，可以获得接近 SOTA 的性能

	## Introduction

1. 从预训练模型中进行迁移学习可以在 NLP 任务中获得很好的性能
2. 本文的目的是，建立一个在所有方面任务都表现良好的系统而不用为每个任务训练一个全新的模型，提出的迁移学习策略可以获得 compact model，即在每个任务中只有少量的附加参数，通过增量训练这些附加的参数而不改变原始模型的参数，且可以获得较好的性能
3. NLP 中两个常见的迁移学习方法是，基于特征的迁移和 fine tune，这两个方法都需要重新训练模型的权重
4. 提出的 adapter 参数效率很高，其本质是**在预训练的网络层之间添加一个新的模块**，相比于前面两种迁移方法：
	1. 考虑以 $\boldsymbol{w}$ 为参数的神经网络模型 $\phi_{\boldsymbol{w}}(\boldsymbol{x})$
	2. 基于特征的迁移学习将 $\phi_{\boldsymbol{w}}$ 和一个新的模型 $\chi_{\boldsymbol{v}}$ 进行组合，得到 $\chi_{\boldsymbol{v}}\left(\phi_{\boldsymbol{w}}(\boldsymbol{x})\right)$，训练的时候只训练其中的 task specific 的参数 $\boldsymbol{v}$
	3. 而 fine tune 则更简单，直接调整整个原始模型的参数 $\boldsymbol{w}$
	4. adapter 定义了一个新函数 $\psi_{\boldsymbol{w}, \boldsymbol{v}}(\boldsymbol{x})$，其中 $\boldsymbol{w}$ 来自于预训练的模型参数，初始参数 $\boldsymbol{v}_0$ 满足 $\psi_{\boldsymbol{w}, \boldsymbol{v}_0}(\boldsymbol{x}) \approx \phi_{\boldsymbol{w}}(\boldsymbol{x})$，训练的时候，只更新参数 $\boldsymbol{v}$，这里的 $\boldsymbol{v}$ 通常是在原来的模型上添加新的模块来实现，而且参数 $\boldsymbol{v}$ 的数量远小于 $\boldsymbol{w}$
5. 基于 adapter的 fine tune 有点类似于多任务学习和连续学习
	1. 多任务学习也是要一个 compact model，但是需要同时访问多个任务，而 adapter 不需要
	2. 连续学习就是从连续不断的任务流中一直学学学，其缺点就是可能会忘掉之前的任务
	3. adapter 不同之处在于，各个任务互不相干，且原始模型的参数不会更新（也就是不会忘记预训练学到的知识）
6. 本文的关键创新就在，设计了一个有效的 adapter 模块，将他和 BERT 集成起来，效果几乎可以和直接进行 fine tune 相当，但是只要调整原始参数 3% 的参数大小，是一个即插即用的模块

## Adapter

核心原理：adapter tuning 策略在原始的网络中添加新的层，训练过程中，原始模型层的参数保持不变，新添加的 adapter  参数随机初始化。

Adapter模块 的两个特点：
+ 参数少
+ 近乎恒等的初始化（为了实现稳定训练）

考虑 Transformer 的标准模型：![](image/Pasted%20image%2020230416094446.png)
Adapter 位于每个 sub layer 的输出，通过残差连接之后的输出直接传递到 layer norm 中。

为了限制参数的大小，还提出了一个 bottleneck 的结果，即 adapter 首先将 FFN 输出的 $d$ 维的特征投影到一个较小的维度 $m$，然后通过一个非线性层，再投影回 $d$ 维。那么包含 bias 的总的模块参数为 $2md+d+m$，如果 $m\ll d$ ，则可以极大地限制参数量，实际用的时候只用了原始模型 $0.5-8\%$ 左右的参数，由于 adapter 内部本身也有一个残差连接，可以很方便地实现恒等初始化（所有的参数都接近 0）。

同时对于每个任务也单独训练了 layer norm 层的参数，类似于 batch norm 的 parameter-efficient，但是效果不太好。

## 实验

采用预训练的 BERT 作为 base 模型，分类的时候将第一个 token 的输出作为 classification token，然后训练一个 linear 层实现分类。

4太 TPU 上训练，batch 为 32。

### GLUE benchmark

采用 BERT-LARGE 模型，330M 参数训练 adapter的时候，通过超参数扫描获得最佳的超参配置。

结果：
![](image/Pasted%20image%2020230416100127.png)

参数和性能的 trade-off：![](image/Pasted%20image%2020230416101051.png)


### 消融实验
![](image/Pasted%20image%2020230417144001.png)
纵轴-横轴表示删掉位于其中层的 adapter 的效果，对角线表示删掉单独的这一层的 adapter，右上角表示全删掉（也就是不用 adapter的结果）。
表明：
1. 删除单层对效果影响较小
2. 全删会导致性能大幅下降
3. 删掉低层的影响更小，删掉高层的影响更大