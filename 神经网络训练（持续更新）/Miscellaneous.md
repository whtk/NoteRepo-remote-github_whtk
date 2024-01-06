1. add_argument 不支持多个 type 类型
2. 梯度累加 变相增加 batch_size，但是当代码中存在 batch norm 的时候，两者并不完全等价
3. Dataloader 中的  shuffle 为 True 时，是先打乱 index，然后再依次输出，也就是说只要 drop last 不为 True 时，每个 epoch 是恰好迭代完整个数据集的所有数据且不重复不遗漏。
4. 关于 num_workers 的选择，踩坑自闭了一周了，不是越大越好，最佳的做法是固定训练的 batch size，然后测试1，2，4，8，12，16这几个数字选择最好的，不要听信网上说的所谓的 设置为cpu核心数大小！！！已下是实测结果：
| num_workers | cpu 占用情况 | 训练用时           | pytorch lightning warning |
| ----------- | ------------ | ------------------ | ------------------------- |
| 1           | 30不到       | 48s                | 有                        |
| 2           | 40-60之间    | 28s                | 有                        |
| 4           | 能到 85      | 24s                | 无                        |
| 8           | 能跑满       | 26s                | 无                        |
| 16          | 占满         | 服务器直接重启。。 | 无                        |

测试条件：python pl_main_asv.py --max_epoch 3 --device 4 --log_dir asv_test_log --limit_train_batches 0.3 --limit_val_batches 0.5
从表里面可以看出，**不是 num_workers 越大越好！！！不是 num_workers 越大越好！！！不是 num_workers 越大越好！！！**

5. IO 的读取速度也可能是性能瓶颈，毕竟大部分的数据集都放在机械硬盘上，监控磁盘的性能的命令为：
	1. iostat -x /dev/sdc1 3 5
6. 关于 model.train() 和 model.eval() 会导致模型的性能差异，原因是模型中存在 batch norm 和 dropout 层
7. 李宏毅老师讲的一个故事：对于 sequence to sequence 的模型，在 inference 的时候加上 dropout 会有奇效
8. 模型的参数中，如果涉及乘法运算（加法除外），其**永远不能初始化为 0**，不然这个参数是不会被更新的！！！
9. pytorch 中，torch.stft 两个不同的版本差异还挺大的：
	1. 对于 1.6.0 的版本，torch.stft 返回的维度是 $*\times N\times T \times 2$
	2. 而对于 1.13 的版本，则通过 return_complex 这个参数来控制：![](image/Pasted%20image%2020231020120537.png)
	3. 两种不同的返回维度影响 spectrogram 的计算，也就是最后要不要执行 sum(-1)
10.  


## 音频基础

1. 音频的 container 就是指文件的类型，而 codec 是指编解码器，对应于文件的压缩类型：
	1. 一种 container 可能支持多种 codec，如 MPEG-4 container 同时支持 FLAC、AAC、Opus 等多种编码
	2. 也有不少的 codec 和 container 是一一对应的，如 FLAC
	3. .wav 是一种比较特殊的 container，因为他直接存储的是原始的音频比特流，所以很少有 codec 支持 wav，wav 通常采用 PCM 直接编码，可以看成是未经过压缩的原始音频
	4. .flac 也比较常见，但是他是通过无损压缩得到的，也就是此时里面存储的不是原始的波形文件，而是需要 decoder 进行解码才能得到原始的波形
	5. ALAC codec 也是一种无损压缩格式，但是受众较小，但是 iOS 不支持
	6. .ogg 是一种容器，可以包含很多的音频格式，大多都是 ogg的vorbis 的格式，效果和 mp3 相媲美，但是也可以包含其他的 codec 的音频，如 flac、mp3 等
	7. opus 可以无缝调节编码的比特率，可以从低比特率窄带语音扩展到非常高清音质的立体声音乐，且是一个免费的开源格式
	8. 然后像 PCM、PCM_16 这种是指编码格式，编码算法和压缩算法是不一样的