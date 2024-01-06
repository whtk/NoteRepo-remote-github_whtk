
> [IEEE Signal Processing Letters](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=97)

1. 提出 Time-domain Synthetic Speech Detection Net（TSSDNet）具有ResNet或Inception风格的结构
2. 不需要手工的特征 ，是完全端到端的

## Introduction

1. 前后端和端到端框架对比：![[Pasted image 20230103105708.png]]
2. 手工提取的信息通常有一些信息的丢失（如相位信息）
3. 本文表明，基于 DNN 的合成语音检测可以不用手工制作的特征，仅输入语音波形的端到端轻量级神经网络可以获得更好的结果
	1. 提出 TSSDNet，考虑了两种类型的 CNN：ResNetstyle skip connection with 1 × 1 kernels 和 Inceptionstyle parallel convolutions

## 方法
> 在合成语音检测中，关键特征是数据伪造后留下的伪影，且可能不包含任何语义信息，因此可以不用深层的网络

### 模型架构

提出的两种模型 Res-TSSDNet 和 Inc-TSSDNet 如图：
![[Pasted image 20230103110737.png]]
为了控制模型的复杂度和增加感受野，Inc-TSSDNet 使用了 dilation。

所有的卷积层用的都是 “SAME” padding，stride = 1。

### 训练策略

语音长度固定为 6s，直接输入网络进行训练。

采用加权交叉熵损失，设训练集 $\left\{x_i, y_i\right\}$，其中 $y_i \in\{0,1\}$，则损失定义为：$$\operatorname{WCE}\left(\mathbf{z}, y_i\right)=-w_{y_i} \log \left(z_{y_i}\right)$$
这里的 $\mathbf{z}=\left[z_0, z_1\right]$ 为softmax 二分类的概率输出，$w_{y_i}$ 为标签 $y_i$ 占比的倒数。 

采用 Adam 优化器的默认配置进行训练。

使用混合正则化进一步提高泛化能力，具体来说，使用混合样本和标签来进行训练而不是原始的样本和标签，即定义新的训练对：$$\tilde{x}_i=\lambda x_i+(1-\lambda) x_j, \quad \tilde{y}_i=\lambda y_i+(1-\lambda) y_j$$
其中，$\left\{x_i, y_i\right\}$ 和 $\left\{x_j, y_j\right\}$ 为从原始训练对中随机选的两个对，$\lambda \sim \operatorname{Beta}(\alpha, \alpha),\alpha \in(0, \infty)$ 为超参数，可以通过以下损失函数实现混合正则化：$$\mathrm{CE}_{\text {mixup }}\left(\tilde{\mathbf{z}}, y_i, y_j\right)=\lambda \mathrm{CE}\left(\tilde{\mathbf{z}}, y_i\right)+(1-\lambda) \mathrm{CE}\left(\tilde{\mathbf{z}}, y_j\right)$$

## 结果

实验结果对比：![[Pasted image 20230103114047.png]]
Res-TSSDNet 效果最好，并且是端到端的网络，没有模型融合没有特征工程，且模型也很小。

Inc-TSSDNet 是最小的模型，但是效果也挺好。

消融实验（不同网络深度的结果）：![[Pasted image 20230103135923.png]]
太深了或者太浅了都不行。

固定所有的参数多次实验，结果表明，模型越小，最终的结果波动越小。（论文实验了30次，EER 的波动还挺大的emmm）

还有一个有意思的发现是，语音样本的时间对结果也影响很大，5s 会使得性能降低，2s EER 会急剧增加。

跨数据集实验表明，提出的模型 **不能** 泛化到15年的数据集，但是使用 混合正则化，Res-TSSDNet 能够显著减少跨数据集的 EER。Inc-TSSDNets 泛化性能也不错。实验结果如图：![[Pasted image 20230103161158.png]]