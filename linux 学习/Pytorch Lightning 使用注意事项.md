
如果 dataloader 使用自己的 sampler，一定要确保  use_distributed_sampler 这个参数设置为 false！！！否则多卡的时候 batches 计算错误。

Pytorch Lightning 的  resume 有点问题，中途断掉之后，lr 不会被保存，重新跑的时候 lr 是新给的。

