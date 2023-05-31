> 2023 SLT

1. 首先指出训练一种能够泛化到各种欺诈的对策是很难的
2. 数据增强和自监督学习等方法可能有帮助，但不够
3. 本文研究使用 active learning（AL） 从 large pool set 选择有用的训练数据进行训练，同时提出了一种新的 AL 方法，从 pool 中主动删除无用的数据
4. 结果表明，一种 energy-score-based AL 和作者提出的 data-removing 方法优于 strong baseline，能够把错误率相对降低 40%，并且使用了更少的数据

## Introduction

1. CM 需要泛化到未知的数据！
2. 尽管很多好的 CM 在特定的数据库中表现出色，但是在不同的数据库中错误率会增大
3. 这种退化可能是由不同的语言、信道变化甚至训练集中的伪影引起
4. 一个好的解决办法是数据增强，通过使用 codec 或其他信号处理算子来处理波形
5. 也有使用 SSL 作为特征提取器，从而对不同的信道条件、语言和说话人都更为鲁棒
6. 尚未探索的一种方法是增加更多的数据，但是又不能简单地直接添加，而且不是所有的数据都对结果有用，有些数据可能是有害的，因此如果选择有用的训练数据避免有害的数据是一个不错的研究问题
7. 本文提出将 AL 用于 CM，迭代地选择有用的数据然后 fine tune CM 模型：![[Pasted image 20230402165324.png]]候选数据被称为 pool set，AL 要求 CM 评估每条数据的有用性，选择最有用的数据来扩充训练集，然后迭代对模型进行 fine tune。
8. 本文比较了一些现有的算法来衡量这种“有用性”，同时提出 AL 框架的一个变体，即在随机采样新数据扩充训练集之前从 pool set 中删除无用的数据
9. 在 ASVspoof 2019、2021和 WaveFake 的数据库进行实验，能够在使用不到 1/4 的数据下得到和 top line 相当的 EER

## 用于 CM 的基于 AL 的框架

### 有用数据的主动选择
> AL 的总体目标是允许模型选择自己的训练数据。对于 CM，从一个大的 pool set 中选择有用的子集进行训练比全用更有效。

#### 框架

如下算法：![[Pasted image 20230402170306.png]]
从较小的 seed set $\mathcal{U}_{seed}$ 开始，开始训练 CM，然后进入 AL 循环，每次迭代从 pool set 中选择 $L$ 个数据，然后把选中的数据从 pool set 中删除，移到 train set 中，然后在新的 train set 中 fine tune 模型。

这里的 $\mathcal{U}_{seed}$ 可以假设来自一个精心选择的数据库，而 $\mathcal{U}_{pool}$ 来自各种语音数据的合集。AL 的目的就是从 $\mathcal{U}_{pool}$ 中选择有用的数据。

关键在于第五行，常见的是 certainty scoring ，将 certainty 较小的加入到 train set 中，从而 CM 可以学习对困难的数据进行分类。
> 关于 AL 的更多原理，见 [[主动学习调研]]。

本文比较了几种不同的 score 计算方法，最后都会为每个语音得到一个 score $c_{m}\in \mathbb{R}$，具体见下一节。 

####  certainty scoring 方法

##### Negative energy-based certainty score

因为最终是二分类，模型最后得到的 logits（输入到 softmax 之前）为 $\left\{l_{m, 1}, l_{m, 2}\right\}$，则 certainty score 计算为：$$c_m=\mathcal{F}_{\mathrm{CM}}\left(\boldsymbol{o}_{1: T_m}\right)=-\log \sum_{j=1}^2 \exp \left(l_{m, j}\right)$$
这个也被成为 negative energy score，当 CM 对输入的确定性较低时，score 较小。这个也会优于经典的交叉熵 score。

##### Adversarial-sample-based distance

该方法假设可以在CM的对抗性样本附近找到有用的数据，在 图像的 AL 中效果不错。从 $\mathcal{U}_{train}$ 中随机采样一个 mini-batch 的数据，根据这些数据产生对应的对抗样本，然后计算 pool set 和生成的对抗样本之间的距离，最近的那个距离被用作得分 。

##### Random scoring

随机产生 score，相当于从 pool set 中随机采样，这个就不是 AL 了，而是 PL（passive learning）。

### 无用数据的主动去除
> 这一部分是作者的创新。

每次选择最有用 的数据的采样可能引入采样偏差，导致局部最小值，探索选择不是最有用的数据可能反而带来更多的数据多样性。

于是提出在每次迭代中主动删掉无用的数据：![[Pasted image 20230402174109.png]]
和前面的算法不同之处用红色标明了。采用  negative energy score 来衡量数据的无用性，然后采用 random sampling 选择数据。

## 实验

### 数据

首先就是收集数据得到 seed 和 pool set：![[Pasted image 20230402213454.png]]

### 模型和配置

包含以下几种 CM 模型：![[Pasted image 20230402213743.png]]
其中 $AL_{PosE}$ 应该是效果最差的，因为他每次选的是最不有用的数据加入 train set，然后还有两个：![[Pasted image 20230402213905.png]]
所有的都基于 wav2vec2.0 前端，最终通过 temporal pooling 得到一个向量，然后通过 linear 层和 softmax 得到二分类的输出。训练的时候 SSL 的前端也会更新。

每次迭代选择的数据量 $L=2560$，最大迭代 $8$ 次，基于 AL 的模型每次迭代 fine tune 5 个 epoch。

采用 Adam 优化器，mini batch 为 16，学习率 1e-6，用三个随机种子训练和评估了三次。

### 结果

![[Pasted image 20230402214611.png]]

![[Pasted image 20230402214558.png]]

图是每次AL迭代后基于AL的CM的EER，表是最后一次AL迭代后的EER。

结论：
1. pool set 会影响 AL 的性能，随着更多的数据从 pool set 选择，EER 减少；使用 pool set B 效果更好，这说明 pool set 越大其数据的多样性越高。但是包含的无用数据也越多，从而导致 pool set B 的方差更大；极端一点是图中的绿色曲线，B 的效果反而更差
2.  哪一种 AL CM 更有效：在 pool set B 中，ALNegE 和 ALRem 性能最好；对于 2021 DF test set，只使用 37% 的数据，效果都不比top差（其实 top 也不是最好的性能，因为如果 pool set 有偏差，top 的效果反而会差）；不建议使用 $AL_{Adv}$
3. 关于数据选择的问题，![[Pasted image 20230402220811.png]]作者发现， AL 会在选择数据的时候试图增加真实数据的数量，以平衡训练集
> 关于第三点，这算不算给出了一个灵感，就是我只真对真实样本做数据增强，同时在数据增强的过程中增加样本的数量，而对于虚假的语音其实可以不用那么多的数据，一方面可以增加真实样本的数量以平衡正负样本数据，另一方面可以提高模型对正样本的分类学习能力（或者说增加正样本空间的裕度）
> 因为既然是 AL 自动选择的，而且会增加正样本的数量，这说明模型对正样本的 uncertainty 是比较低的，可能分类的 accuracy 也会比较低。