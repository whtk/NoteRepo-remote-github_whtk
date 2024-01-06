
1. 使用 Mockingjay 自监督学习模型提取特征，来保护反欺诈模型在黑箱场景中免受对手攻击
2. 提出了一种分层噪声信号比（LNSR）来量化和测量模型在对抗 对抗噪声 中的有效性
3. ASVspoof 2019数据集结果表明，Mockingjay提取的 high-level representations 能够 prevent 对抗样本的 transferability ，成功抵御黑箱攻击

## Introduction

1. 对抗样本：通过向输入样本添加不可察觉的扰动（人类无法从视觉或听觉上区分）而生成的样本
2. 有很多研究表明，ASR、ASV等系统都容易受到对抗攻击，本文则主要研究防御对策
3. 对抗攻击的两种防御手段：
	1. 主动防御：建立新模型来对抗攻击，如对抗训练，但是要已知对手的对抗攻击算法
	2. 被动防御：不修改模型进学校防御，如 滤波等。
4. 本文提出利用自监督学习进行被动防御。自监督学习模型可以作为深度滤波器，从污染的谱图中提取关键信息来对抗 对抗攻击。同时提出分层噪信比，以量化和测量模型在对抗噪声的性能。

![[Pasted image 20221115094408.png]]
## 对抗攻击

微小扰动  $\delta$（小到人类无法察觉）添加到原始样本 $x$ 中，新的样本 $\tilde{x}=x+\delta$ 将导致模型的错误预测。给定反欺诈模型 $f(\cdot)$ ，$f(x),f(\tilde{x})$ 分为别原始样本和对抗样本的预测输出。

对抗攻击相当于，找到合适的 $\delta$ 进行优化：$$\max _{\|\delta\|_{\infty} \leq \epsilon} \operatorname{Dif} f(f(x), f(\tilde{x}))$$其中，$\operatorname{Diff}(f(x), f(\tilde{x}))$ 是表示 $f(x), f(\tilde{x})$ 之间的差异且此差异是可微的。$\delta$ 的不同搜索策略导致不同算法，本文使用了 FGSM 和 PGD 算法。

存在两种对抗攻击场景：黑盒攻击和白盒攻击；每种攻击都有两种模型：目标模型和攻击模型：
+ 在黑盒攻击中，目标模型和攻击模型是不同的模型
+ 在白盒攻击中，目标模型也是攻击模型
在黑盒攻击中，攻击者无法获取目标模型的内部参数，而他们可以通过查询目标模型来收集目标模型的输入和输出。然后攻击者将训练一个替代模型，并使用该替代模型作为攻击模型来生成具有可转移性的对抗样本。

## 方法

### Mockingjay

Mockingjay 通过 $L_1$ 重构损失 解决自监督 masked-prediction （掩码预测）问题 学习语音表征。模型基于 BERT， 具体原理详见原论文。

自监督训练结束后，模型的输出表征做为反欺诈模型的输入。

### 对抗防御

Mel 谱 这类的特征通常会掩蔽丰富的语音信息，而 Mockingjay 提取的高阶表征更适用于下游任务。

黑盒攻击时，攻击者不知道内部使用了 Mockingjay，只知道输入是语谱图。

Mockingjay 将有助于减轻添加到输入频谱图中的噪声，并避免对抗性噪声的transferability。由于目标模型有 Mockingjay 而攻击模型没有，所以很显然攻击信号不能传递到目标模型。

实验表明，预训练非常重要，在没有预训练的情况下，仅仅使用网络架构的失配不能避免对抗性噪声的 transferability。

Mockingjay 可以对抗攻击的两种可能解释：
1. 从自监督角度看，掩膜训练已经引入了噪声，Mockingjay 可以训练来减弱噪声并使用主要信息来重构原始的语谱。  
2. 从损失函数的角度看，在黑盒攻击场景中，目标模型和攻击模型执行相同的任务，并通过分类损失进行训练。而 Mockingjay通过重构损失来训练，并执行与攻击模型不同的任务。

### 层级噪信比（LNSR）

LNSR 用来估计 Mockingjay 不同层的 噪声强度：
$$\begin{array}{r}
L N S R_i=\sum_{n=1}^N \frac{\left\|\hat{h}_i^n-h_i^n\right\|_2}{\left\|h_i^n\right\|_2} \\
\text { for } i=0,1, \ldots, K,
\end{array}$$
其中，$K$ 为 Mockingjay 的总层数 。$N$ 为 对抗-原始样本对 的数量。$\hat{h}_i^n, h_i^n$ 分别表示第 $i$ 层的对抗样本和原始样本的特征。如果 $L N S R_i$ 随着 $i$ 的增加而减少，说明 Mockingjay 确实可以减弱攻击噪声。

## 实验

数据集：ASVspoof 2019 LA

Mockingjay：12 层 transformer encoder

后端模型：LCNN 和 SENet（baseline 的输入是mel谱）

1. 不同方法的对比：![[Pasted image 20221115104009.png]]可以看到，灰色的线（Mel 谱）容易受到攻击，红色的线（Mock）在所有的攻击和模型中效果都是最优的（当然也优于滤波法）。橙色的线是随机初始 Mockingjay 参数的，可以看到效果有，但不明显（甚至会更差）。scratch 表示从零开始训练的 Mockingjay 和 LCNN/SENet 模型，显然效果很烂。
2. 对抗噪声的减少：![[Pasted image 20221115104858.png]]预训练的 Mockingjay 成功降低了 LNSR。模型越深时，LNSR 越低。而随机参数化的 Mockingjay 只能在一定程度上降低LNSR。



