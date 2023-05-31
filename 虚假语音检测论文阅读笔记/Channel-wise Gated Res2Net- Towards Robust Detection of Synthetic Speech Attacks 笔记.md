
> [[Res2Net]] 最早来源于南开博士的一篇论文

1. 作者认为，不同信道的信息重要程度不一样，通过抑制一些不相关的通道，系统对未知攻击有着更好的泛化性
2. 因此提出了 channel-wise gated Res2Net （CG-Res2Net）在信道维度添加门控机制以抑制不相关通道。
3. 提出了三种门控机制，在 ASVspoof 2019 LA 进行实验，优于 [[Res2Net]]，也优于 其他最先进的单系统


## Introduction
1. 现有的 SOTA 系统很难泛化到未知的攻击（如 LA 中的 A17）中，这项工作旨在增强LA攻击检测的泛化性能。
	1. [[Data augmentation with signal companding for detection of logical access attacks 笔记]] 中基于信号拓展方法增强训练数据提高其泛化性
	2. [[End-to-End Anti-Spoofing with RawNet2 笔记]] 直接在原始语音波形上检测合成语音
	3. [[Replay and Synthetic Speech Detection with Res2Net Architecture 笔记]] 利用Res2Net架构改进了模型的泛化性，并证明了其对未知LA攻击的卓越检测精度
2. Res2Net在一个block内的不同特征组之间设计了一种 residual-like connection，这增加了可能的感受野，提高对未知攻击的泛化性，但是直接相加忽略了信道的优先权，而信道信息对欺骗的贡献程度是不一样的，通过抑制不相关的信道特征可以提高泛化性。
3. 提出了 CG-Res2Net，在特征组之间的 residual-like connection 中启用通道门控机制：选通相关的，抑制不相关的。比较了以下三中门控机制，三个系统都优于当时的 SOTA单系统：
	1. single-group channel-wise gate (SCG)
	2. multi-group channel-wise gate (MCG)
	3. multi-group latentspace channel-wise gate (MLCG)

## 方法

### channel-wise gated Res2Net

本节给出了 CG-Res2Net 的结构。原始模型和修改后的模型结构如下图的 a 和 b：
![[Pasted image 20221031192201.png]]
在 $1\times 1$ 卷积之后，两个模型都将特征图 $X$ 均匀分成 $s$ 个子集，记为：$x_i \text {, 其中 } i \in\{1,2, \ldots, s\}$, 假设 $X\in \mathbb{R}^{s\cdot C \times D \times T}$，则 $x_i \in \mathbb{R}^{C \times D \times T}$，分别表示 channel、spectrum、time 三个维度。Res2Net 允许在 $3\times 3$ 卷积之前在特征组之间进行相加。每个 $y_i$ 定义为：
$$y_i= \begin{cases}x_i, & i=1 \\ K_i\left(x_i\right), & i=2 \\ K_i\left(x_i+y_{i-1}\right), & 2<i \leq s\end{cases}$$
$K()$ 表示卷积函数。CG-Res2Net 在此基础上采取了一个门控机制：
$$\begin{aligned}
y_i &= \begin{cases}x_i, & i=1 \\
K_i\left(x_i\right), & i=2 \\
K_i\left(x_i+z_{i-1}\right), & 2<i \leq s\end{cases} \\
z_{i-1} &=y_{i-1} \otimes a_{i-1}
\end{aligned}$$
其中，$a_i \in \mathbb{R}^C$ 为 channel-wise 的门控系数。这种门控机制可以优先考虑对欺诈有帮助的通道，抑制不相关的通道，增强模型对未知攻击的泛化性。

### 三种门控机制

上图的 c-e 给出了三种不同的门控机制，分别描述如下：

1. Single-group channel-wise gate（图 c）：上一个通道的结果 $y_i$ 首先在 channel 通道上进行 avgpool，通过全连接层后进行维度转换，最后通过一个 sigmoid 函数得到 $a_i$，$$\begin{aligned}
F_{a p}\left(y_i\right) &=\frac{1}{D \times T} \sum_{d=1}^D \sum_{t=1}^T y_i(:, d, t) \\
a_i &=\sigma\left[W_{f c}^T F_{a p}\left(y_i\right)\right]
\end{aligned}$$
2. Multi-group channel-wise gate（图 d） ，不仅考虑 $y_i$ 还考虑 $x_{i+1}$：$$a_i=\sigma\left\{W_{f c}^T\left[F_{a p}\left(y_i\right) \oplus F_{a p}\left(x_{i+1}\right)\right]\right\}$$ 其中，$\oplus$  是 concatenate 操作。
3. Multi-group latent-space channel-wise gate（图 e）：$y_i$ 和 $x_i$ 的信息对应的功能是不对称的，所以可以在拼接之间分别进行处理，即在连接之间将两个特征子图分别投影到各自的 latent-space 中：$$\begin{aligned}
&L_1\left(y_i\right)=\delta\left(W_{f c 1}^T F_{a p}\left(y_i\right)\right) \\
&L_2\left(x_{i+1}\right)=\delta\left(W_{f c 2}^T F_{a p}\left(x_{i+1}\right)\right) \\
&a_i=\sigma\left\{W_{f c 3}^T\left[L_1\left(y_i\right) \oplus L_2\left(x_{i+1}\right)\right]\right\}
\end{aligned}$$其中，$\delta$ 代表 RELU 函数。

## 实验

数据集：ASVspoof 2019 LA

baseline：Res2Net50 with squeeze-and-excitation (SE) block （[[Replay and Synthetic Speech Detection with Res2Net Architecture 笔记]] 中的结构）

特征：CQT，16ms 帧长、Hanning 窗、9 octaves with 48 bins per octave，固定 400 帧

训练：bce 损失、Adam 优化器、$\beta_1=0.9, \beta_2=0.98, lr=0.0003, epochs=20$.

## 结果

1. 门控机制：![[Pasted image 20221101100551.png]]总结：SCG-Res2Net50的性能略好于Res2Net 50，MCG-Res2Net50和MLCG-Res2Net50都表现出了比Res2Net50更大的改进
2. 未知攻击：![[Pasted image 20221101100311.png]]对于最困难的A17攻击，ResNet50的检测准确率低于50%，Res2Net50仅达到81.48%的准确率。三种提出的CG-Res2Net50模型都大大优于Res2Net50。MLCG-Res2Net50实现了87.63%的最高检测精度；对于其他容易检测到的攻击，CG-Res2Net50模型的性能与Res2Net50相当。
3. 和 SOTA 进行比较：![[Pasted image 20221101100749.png]]CGRes2Net模型优于其他SOTA系统，此外，所提出的CG-Res2Net模型可以用作主干网络，与其他有效的策略相结合，进一步提高对未知攻击的泛化性。
