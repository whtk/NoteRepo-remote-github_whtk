
如果 dataloader 使用自己的 sampler，一定要确保  use_distributed_sampler 这个参数设置为 false！！！否则多卡的时候 batches 计算错误。

Pytorch Lightning 的  resume 有点问题，中途断掉之后，lr 不会被保存，重新跑的时候 lr 是新给的。

log 函数中，参数 sync_dist 需要小心配置，对于多机多卡，这个参数会影响（减慢）10%-15%的性能。

关于 Pytorch Lightning 的自动优化：
1. automatic_optimization 参数默认为 True，这样会自动进行优化，不需要手动进行优化
2. 设置 automatic_optimization=False 时，可以手动进行优化，此时需要完成以下步骤
    1. 在 LightningModule 的 `__init__` 中设置 `self.automatic_optimization = False`
    2. 在 `training_step` 中手动进行优化，如下：
        1. 通过 `self.optimizers()` 获取优化器
        2. 通过 `optimizer.zero_grad()` 清空梯度
        3. 通过 `self.manual_backward(loss)` 进行反向传播
        4. 通过 `optimizer.step()` 更新参数
        5. 通过 `optimizer.step()` 更新学习率（可选）


