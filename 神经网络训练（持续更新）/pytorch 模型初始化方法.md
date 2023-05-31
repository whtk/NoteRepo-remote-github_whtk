> 都包含在 torch.nn.init 这个模块下，里面所有的方法都用来初始化神经网络的参数，因此都在 [`torch.no_grad()`](https://pytorch.org/docs/stable/generated/torch.no_grad.html#torch.no_grad "torch.no_grad") 模式下运行，也就是不会进行梯度计算。
> 截止到 pytorch 1.13 版本。


## torch.nn.init.calculate_gain
返回给定非线性函数的增益值

## torch.nn.init.uniform_
以均匀分布的随机数填充输入 tensor

## torch.nn.init.normal_
以正太分布的随机数填充输入 tensor

## torch.nn.init.constant_
以常数 c 填充输入 tensor

## torch.nn.init.ones_
以常数 1 填充 输入 tensor

## torch.nn.init.zeros_
以常数 0 填充 输入 tensor

## torch.nn.init.eye_
以单位阵填充输入的 二维 tensor（可以不用是方阵）

## torch.nn.init.dirac_
以 狄拉克 delta 函数填充 3或4或5 维的 tensor

## torch.nn.init.xavier_uniform_
根据Understanding the difficulty of training deep feedforward neural networks 论文中的方法，以均匀分布填充 tensor，得到的值是从 $\mathcal{U}(-a, a)$ 采样的，其中：$$a=\text{gain}\times\sqrt{\frac{6}{\text{fan\_in}+\text{fan\_out}}}$$
也被称为 Glorot initialisation。
> 这篇论文的重点是，在线性的人工神经网络中，BP 算法的梯度的方差取决于层数，因此可以通过对神经网络分层初始化来提高网络的收敛速度和泛化能力。
> 这里的 fan in 和 fan out 是指上一层和下一层的神经元的个数（tensor的维度） 

## torch.nn.init.xavier_normal_
和上面一个差不多，只不过采用的是正太分布（均值为0），标准差为：$$\operatorname{std}=\text { gain } \times \sqrt{\frac{2}{\text { fan\_in }+\text { fan\_out }}}$$

## torch.nn.init.kaiming_uniform_
> 是目前 pytorch linear、conv 层默认的初始化方式

根据 Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification 论文中的方法，以均匀分布填充 tensor，得到的值是从 $\mathcal{U}(-\text{bound}, \text{bound})$ 采样的，其中：$$\text { bound }=\text { gain } \times \sqrt{\frac{3}{\text { fan\_mode }}}$$
也被称为 He initialization。
> 这篇文章初始化的主要思想是，xavier 的初始化推到是基于线性对称的激活函数的，而针对于 ReLU 这种非线性不对称的还没有一个合适的初始化（因为 ReLU 把负数都变成了0，整体的方差变了）。
> 这里的 fan mode 只能选 fan in 或者 fan out，选择“fan_in”将保留前向传递中权重方差的大小。选择“fan_out”将保留后向传递中的大小，默认是 fan in（因为负的都变成0了，所以继承前面的？）。

## torch.nn.init.kaiming_normal_
和上面一个差不多，只不过采用的是正太分布（均值为0），标准差为：$$\text { std }=\frac{\text { gain }}{\sqrt{\text { fan\_mode }}}$$
## torch.nn.init.trunc_normal_（略）

## torch.nn.init.orthogonal_（略）

## torch.nn.init.sparse_（略）

## 重点
如何选择 xavier 或 kaiming 初始化：emmm 直接用默认的 kaiming 就行了，但是如果模型中没有 类似 ReLU 这种激活的话可以用 xavier。