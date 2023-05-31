
1. 提出一个新的可解释框架，衡量数据质量对反欺诈效果的影响
2. 选择长期频谱信息（long-term spectral information）、说话人（基于 [[x-vector、i-vector]]）embedding 和 SNR 作为质量评估指标


## Introduction

1. 许多工作专注于设计声学特征，如 [[cqcc]]、[[Modulation Dynamic Features for the Detection of Replay Attacks 笔记]] ；或者 DNN 架构，如 [[Audio replay attack detection with deep learning frameworks 笔记]]、[[A Gated Recurrent Convolutional Neural Network for Robust Spoofing Detection 笔记]]；或者融合不同的模型，如 [[Ensemble models for spoofing detection in automatic speaker verification 笔记]]、[[ResNet and model fusion for automatic spoofing detection 笔记]]；但是一个好不代表大家好，模型在单个语料库的结果并不能作为一般性的衡量标准
2. 在不同的数据集中的效果差太多了：![[Pasted image 20221102163334.png]]
3. 下图可以看到，不同语料库数据的频谱差异：![[Pasted image 20221102163644.png]]
4. 本文旨在量化语料库级的声学不匹配因子对语音反欺骗性能的影响


##  方法

### 数据集设置

假设有 $M$ 个不同的数据集，对于每一个数据集，划分一个训练集，$N_{test}$ 个测试集，然后在每个数据集的所有测试集中进行测试，这样对每个数据集就可以得到 $N_{test}$ 个集内实验，$(M-1)\times N_{test}$ 个集外实验，然后一共有 $M\times N_{test}$ 个集内实验，$M\times (M-1)\times N_{test}$ 个集外实验。

实际实验时，$M=7,N_{test}=20$


### 多元线性回归

定义数据对 $\left\{\left(\boldsymbol{d}_t, E_t\right): t=1, \ldots, T\right\}$，其中 $E_t$ 为 评估指标 EER，下标 $t$ 代表第几个训练-测试对，$d_t=\left(d_t^{(1)}, \ldots, d_t^{(R)}\right)^{\top} \in \mathbb{R}^R$ 为一系列的预测因子。

### 预测因子定义

$\mathcal{D}_{train}, \mathcal{D}_{test}$  分别表示训练和测试集（可以来自相同或不同的数据集），其中 $\mathcal{D}_{\text {train }}=\left\{\left(\mathcal{X}_m, y_m\right)\right\}^{\substack{N_{\text {train }}}}_{j=1},\mathcal{D}_{\text {test }}=\left\{\left(\mathcal{X}_m, y_m\right)\right\}^{\substack{N_{\text {test }}}}_{j=1}$ 为数据样本，$y \in\{0 \equiv \text { spoof, } 1 \equiv \text { bonafide }\}$，第 $j$ 个波形 $\mathcal{X}_j$ 可以用一系列的质量特征来描述 $\phi_j^{(1)}, \ldots, \phi_j^{(Q)}$，例如，$\phi_j^{(1)}$ 可以表示 SNR，$\phi_j^{(2)}$ 可以表示 512 维的说话人 embedding 等，并认为 $Q$ 个特征独立。

训练/测试、真/假语音两两组合得到四个条件概率：![[Pasted image 20221102172038.png]]基于上图可以得到 6 个预测因子。$d_{13},d_{24}$ 好理解，其他的四个是为了防止系统学习到一些和欺诈检测无关的东西（但是确把他当作真实和虚假语音的特征区别了）。

## 实验

采用 GMM 和 CNN 作为分类器进行实验，数据质量特征包括五种：
+ LTAS 时间维度进行平均的频谱信息
+ SNR
+ Noise spectrum 噪声功率谱密度
+ X-vector
+ Acoustic descriptors：包括 F0、F1 和响度

## 结果

1. 下图给出了 EER 的分布： ![[Pasted image 20221102173322.png]]
2. LTAS特征距离与CNN分类器的EER的皮尔森相关性：![[Pasted image 20221102173513.png]]显然，域内的相关性强于域外；四个类间距离（d12；d14；d23；d34）与 EER 的相关性比类内距离（d13；d24）更强。
3. 下表给出了语料库间和跨语料库数据的每个分类器的特征模型的 adjusted-$R^2$![[Pasted image 20221102173858.png]]可以看到，在域内数据集，LTAS、x-vector等特征 和 EER 的相关性很大，但是在域外的相关性就很小了，这也解释了为什么跨语料库的效果会变差。同时还表明：
	1. LFCC-GMM受LTAS影响最大，响度影响最小
	2. CQCC-GMM受SNR影响最大，响度影响最小
	3. CNN受x-vector的影响最大，受F0（语料库内）或噪声谱（语料库间）的影响最小
		从而可以推测，CQCC-GMM系统可能对噪声敏感，CNN系统可能受到说话人特征选择的更强烈影响。


