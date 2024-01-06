1. 检测欺骗所需的特征通常位于特定的频谱子带或时间段，本文使用GAT来对这些关系进行建模以提高检测性能。
2. 图由 ResNet 生成的表征构建，图中的节点代表特定子带或者时间段信息。ASVspoof2019 LA 中的表现由于所有单系统的baseline，融合模型相比于单GAT，性能提高了 47%


## Introduction
1. CNN 可以在谱-时间 图中提取局部伪影，2017 和 2019 的ASVspoof 中，基于 CNN 的方法是表现最好的系统之一
2. 作者之前使用了 RawNet2 [[End-to-End Anti-Spoofing with RawNet2 笔记]] 进行反欺骗检测，其中使用了 FMS，可以看作是在频域使用注意力机制。但是他不会学习或者建模不同滤波器或子带之间的关系。
3. 本文假设，一种能够对不同伪影之间的关系进行建模的注意力机制有可能提高反欺骗性能
4. GNN 可以对跨越不同子带和时间段的非欧几里德数据进行建模；而 GAT 可以学习到哪些子带或者时间段信息相对于其邻域信息最丰富，然后对那些最具有辨别力的子带或时间段分配较高的权重。因此本文探索了使用 GNN 来建模频谱和时间关系。

## 相关工作
1. 相比于 Graph SAGE 和 GCN，GAT 使用自注意力机制学习权重，根据节点相对于相邻节点提供的信息来加权。
2. GNN 已经被用于语音的多个方面：
	1. GCN 用于语音的 few-shot audio classification
	2. GNN 可以用于神经语音合成
	3. GAT 可以用于学习话语级别的关系，使用GAT来计算ASV的话语级别的相似性分数

## 使用 GAT 进行反欺诈
本文使用的 GAT 框架如图：
![[Pasted image 20221023152006.png]]
### 高维特征提取
使用 ResNet-18 从声学特征中学习高级表征，最终得到一个三维向量，三个维度分别代表 CNN 的kernel数量、频率轴和时间轴，然后可以分别在时间或者频率轴上进行平均。

### GAT
高层表征输入到 [[Graph Attention Networks for Speaker Verification 笔记]] 中的 GAT 模型中，图是全连接的，同时还包括自连接，GAT 使用自注意力机制学习到的权重来聚合相邻节点。在GAT中，信息量更大的节点使用更大的权重进行聚合，其中权重反映了给定节点对之间关系的强度。GAT 的输出为：
$$G A T(\mathcal{G})=\frac{1}{N} \sum_{n \in \mathcal{G}} \boldsymbol{o}_n W_{\text {out }}$$
其中，$W_{\text {out }}$ 为投影矩阵，$\boldsymbol{o}_n$ 是节点 $n$ 的输出特征，基于 GAT 的节点传播定义如下：
$$o_n=B N\left(W_{a t t}\left(\boldsymbol{m}_n\right)+W_{r e s}\left(\boldsymbol{e}_n\right)\right)$$
相邻节点通过自注意力进行聚合：
$$m_n=\sum_{v \in \mathcal{M}(n) \cup\{n\}} \alpha_{v, n} e_v$$
其中，$\mathcal{M}(n)$ 表示节点 $n$ 的相邻节点，$\alpha_{v, n}$ 表示节点 $v,n$ 时间的注意力权重，根据以下公式进行计算：
$$\alpha_{v, n}=\frac{\exp \left(W_{m a p}\left(e_n \odot e_v\right)\right)}{\sum_{w \in \mathcal{M}(n) \cup\{n\}} \exp \left(W_{\operatorname{map}}\left(e_n \odot e_w\right)\right)}$$
更多细节见 [[Graph Attention Networks for Speaker Verification 笔记]]。

### 谱-时间 注意力
心理声学研究表明，人类听觉系统可以选择信息最丰富的频谱带，并根据相邻帧之间的时间相关性执行自相关。受此启发，分别在谱和时间级别应用 GAT模型，分别对应上图中的 GAT-S 和 GAT-T。

## 实验
数据集：ASVspoof 2019 LA
指标：t-DCF 和 EER

三个 baseline：
1. LFCC-GMM
2. ResNet-18 但 不同的注意力机制（statistics pooling、self-attentive pooling、attentive statistical pooling）
3. RawNet2

系统最后使用 readout layer聚合节点特征以输出得分，采用 BCE 损失函数，lr = 0.0001，weight_decay = 0.0001，Adam 优化器，batch size = 64，epoch = 300。

系统性能比较：
![[Pasted image 20221023161108.png]]
1. 在攻击级别上，性能存在显著差异。虽然频谱注意力的使用会使某些攻击的性能更好，但时间注意力对其他攻击的效果更好，反之亦然。
2. 对其他攻击和ResNet-18、GAT-T和GAT-S系统的结果进行比较，显示了基于图的方法建模时间或频谱关系的好处。
3. 不同的攻击表现出不同的伪像，其中没有一种可以单独使用单个分类器捕获。

> 所以效果也不是那么好？。。。