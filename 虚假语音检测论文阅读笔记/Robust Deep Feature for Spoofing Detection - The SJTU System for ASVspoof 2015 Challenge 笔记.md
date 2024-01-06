1. 首次在欺诈检测中应用和开发 DNN 方法
2. 提出了一种基于 DNN 的用于欺诈检测的表征，提取所谓的欺诈向量 s-vector，同时研究了马氏距离和归一化方法以获取最佳的系统性能

## Introduction（略）

## 先前的工作

### 特征

特征对欺诈检测非常重要，HMM 生成的语音中的低方差对高阶的 MCEP 特征敏感，使用此特征足以检测 HMM 生成的语音。

当检测合成语音时，合成滤波器会导致相位谱的伪影，因此基于相位谱的特征比基于幅度谱的特征更好进行区分。

本文的Baseline 使用三种不同的特征：
+ MCEP，25 维
+ Band-Aperiodicity (BAP)，5 维
+ pitch (LF0)，1 维

### 模型

使用两个 GMM 模型，然后计算得分：$$\operatorname{score}(\mathbf{x})=\log \left(P\left(\boldsymbol{M}_n \mid \mathbf{x}\right)\right)-\log \left(P\left(\boldsymbol{M}_s \mid \mathbf{x}\right)\right)$$
其中，$\boldsymbol{M}_n$ 代表真实语音，$\boldsymbol{M}_s$ 代表虚假语音，$P(\boldsymbol{x})$ 代表高斯分布，结合给定的阈值进行区分。

### baseline

使用 two class GMM-UBM 作为评估模型，在 dev 数据集的结果如下：![[Pasted image 20230217150921.png]]

## 基于 DNN 的欺诈检测

采用 带 delta 的 FBANK 特征作为 DNN 模型训练的输入。

### spoofing-vector
> 在 d-vector 之后提出，那这不就是和 d-vector 的思想差不多，只不过分类问题变了而已。


作者试图找到一种可直接用于距离度量或分类器的，紧凑、鲁棒和抽象的特征表示，使用 DNN 来提取这种表征。

模型如图：![[Pasted image 20230217151811.png]]
是一个多分类的任务，包含一个真实语音标签和多个欺诈类型的标签。

完成训练后，将最后一个隐藏层的输出 $\boldsymbol{X}_{s, 1}, \boldsymbol{X}_{s, 2}, \ldots, \boldsymbol{X}_{s, n}$ 作为特征，其中 $s$ 表示某段音频，$n$ 表示帧数。最终的表征是这些特征的均值，称为 spoofing vector，s-vector：$$\text { s-vector }(s)=\frac{\boldsymbol{X}_{s, 1}+\boldsymbol{X}_{s, 2}+\ldots+\boldsymbol{X}_{s, n}}{n}$$
### 得分：马氏距离

如果 eval set 中的所有欺骗方法都出现在 train set 中，那么可以将重点放在在选择合适的分类器上。

然而，两个集合中的欺骗算法并不完全相同，因此目的是构建一个健壮的系统。

我们假设一个类中的所有 s-vector 都是正态分布。马氏距离用于估计测试语音段和不同的类之间的距离：$$l_c(\mathbf{x})=-\frac{1}{2}\left(\mathbf{x}-\boldsymbol{\mu}_c\right)^{\top} \boldsymbol{\Sigma}^{-1}\left(\mathbf{x}-\boldsymbol{\mu}_c\right)$$
其中，$\Sigma$ 是不同类别的协方差矩阵的平均值。展开之后忽略第一项，最终得分计算为：$$\operatorname{score}_c(\mathbf{x})=\mathbf{x}^{\top}\left(\boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_c\right)+\left(-\frac{1}{2} \boldsymbol{\mu}_c^{\top} \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_c\right)$$
也可以用传统的 PLDA，对未知方法的泛化性更好。

### 归一化

分类任务时，使用概率作为类别的可能性：$$P(\text { class }=c \mid \boldsymbol{X}=\mathbf{x})=\frac{\pi_c P(\mathbf{x} \mid c)}{\sum_{k=1}^K \pi_k P(\mathbf{x} \mid k)}$$
但是这只针对于训练集，测试集中有未知算法。

在 ASV 中，使用 T-Norm 和 Z-Norm 可以改善性能，本文使用 T-Norm，不同的欺诈算法的得分可能会有一个得分接近训练集中的，T-Norm 计算为：$$\operatorname{score}_{\text {TNorm }}(\mathbf{x})=\frac{\left(\text { score }_{\text {human }}(\mathbf{x})-\operatorname{mean}(\mathbf{x})\right)}{\operatorname{std}(\mathbf{x})}$$
其中，均值和方差由所有的欺诈算法的得分给出，下标 $human$ 表示真实的语音的得分。

还有一种 P-Norm ：$$\begin{aligned}
\operatorname{score}_{\text {PNorm }}(\mathbf{x}) & =\log \left(\frac{\exp \left(l_{\text {human }}(\mathbf{x})\right)}{\sum_k \exp \left(l_k(\mathbf{x})\right)}\right) \\
& \approx \text { score }_{\text {human }}(\mathbf{x})-\max _{k \neq \text { human }}\left(\operatorname{score}_k(\mathbf{x})\right)
\end{aligned}$$
其中 $k$ 属于类的索引。这里假设每个类具有相同的先验概率。

## 实验

在 dev set 中的结果：![[Pasted image 20230217154256.png]]

在 eval 数据集中的结果：
![[Pasted image 20230217154319.png]]


