
## 特点

zero shot TTS：5s 参考音频

few shot TTS：1min 训练数据

webui 支持

语言：中日英，可以跨语言转换

## 原始的训练的数据来源

> 训练数据来源：
> 数据是从我做TTS以来陆续清洗采集的，很难给到你一个完整的list，需要你自己采集。统计层面大概是1000小时左右中文，700小时左右英文和300小时左右日文，一共约2000小时。清洗层面，sovits侧重下音质，gpt侧重文本和停顿正确性（比如口吃、复读的要过滤，标点ASR错误的要过滤，长句中间说话人多次停顿但是文本里又没有标注停顿的要过滤，等等）

> 请问作者，基础模型的数据对采样率有要求吗？
> 录音质量好就可以，采样率没有要求，反正预处理脚本都会统一

## 微调

1. 下载预训练模型
2. 下载 ASR 模型
3. 数据预处理：
	1. 用的数据为 干声（即无伴奏人声，有伴奏的话用 UVR5 去除）
	2. 填路径、分割语音，得到 slice 文件
	3. 进行 ASR 识别，得到 list 文件
	4. 校对 list
	5. 填入 list 和 音频 文件路径，预处理，得到文件：
		1. 2-xxx：name2text.txt
		2. 3-xxx：SSL 特征，pt 后缀
		3. 4-xxx：中文 SSL 特征，pt 后缀
		4. 5-xxx：音频
        5. 6-xxx：name2semantic.tsv，SSL 特征通过量化器得到的离散的 code

> 里面有两个自监督模型：
> 1. chinese-hubert-base 用于从音频中提取自监督特征
> 2. chinese-roberta-wwm-ext-large 用于从文本中提取自监督特征

## 从 SoVITS 到 GPT-SoVITS
SoVITS 做的是 VC，不做 TTS，因此输入只有两段音频而没有文本。

采用 SoftVC 作为 先验编码器，目的是去除音色信息（speaker）保留内容或语义信息。
得到的语义特征再通过 VITS 进行语音合成
> 其实就是用这个特征作为 VITS 中的 condition，即替换了原始的文本特征

推理的时候，需要提供目标说话人的音色信息。
> 训练的时候当然也会，只不过训练的时候因为没有并行数据，所以用的是同一个人的语音（同一个人的音色）。推理的时候，需要提供目标说话人的音色信息，这样就可以实现说话人转换。

但是，采用 SoftVC 作为 先验编码器 来去除音色的时候，并不能完全去除。
> 这也是后续更新为什么用 ContentVec 这个模型的原因。

对 GPT-SoVITS 的启发：
1. 如果泄漏的音色较多（或者说 SSL 特征包含的音色特征较多），那是不是干脆用一个泄漏音色很大的模型（如 CN_HuBERT）作为一种 “引导”，从而缺点变成优点，这个 SSL 特征保留了丰富的音色（所谓的 音色丰富的 语义 token）
然后用 GPT 模型来 “补全” 这些特征。

考虑 VC 任务，此时只有目标语音作为输入：
+ 从目标语音的 SSL 特征中提取带有音色信息的 SSL 特征
+ 用 ContentVec 模型从源语音中提取语义特征
+ 然后把这两个特征通过 GPT 模型进行预测，得到预测的 token，此时的 token 同时包含了目标语音的音色信息和源语音的语义信息

然后考虑 TTS 任务，此时会给定文本输入：
+ 此时我们有两个特征，
    + 一个是前面说的带有音色信息的 SSL 特征
    + 另一个是 从要合成的文本得到的文本语义特征（注意：TTS 任务中，前面说的 ContentVec 模型就不需要了）。
    + （可选）实际使用时，还会加上参考音频的文本（参考音频就是用于提取 SSL 的那个音频）
+ GPT 模型的输入就是前面的两个或者三个特征，预测得到的输出是，目标音色下，对应于输入文本的 token，也就是同时包含一定的目标音色信息和要合成的文本的语义信息的 token。
+ 这样子模型就相当于有两部分的音色，一个是 GPT 生成的 SSL 中 “继承” 的音色，另一个是在 z_p 特征中手动给模型的音色特征。
> 这样可以缓解 VITS 重构音色的压力，之前的音色都是手动给的，现在相当于通过 SSL 又多给了。

具体来说：
整体模型框架：ar-vits
GPT 模型用的是 SoundStorm
音色编码器用的是 TransferTTS
中文的 SSL 特征用的是腾讯游戏的 CN_HuBERT
VC 任务中用上了 ContentVec

VITS 中的量化器：
1. 用的是 RVQ
2. codebook 大小为 1024
3. 但是 residual 的数量为 1 。。。

## 模型结构
微调包括两个部分，且两个部分是无序的
1. SoVITS，对应代码 train_s2
2. GPT 模块，对应代码 train_s1

微调的时候，先数据预处理得到上述需要的特征：
然后：
ssl 特征通过量化器得到 code，这些 code 后续会用于训练 GPT 模块。
> 这里的 code 包含了 内容信息 和 speaekr 信息

code 和 文本特征 输入到 先验编码器TextEncoder 得到分布的均值和对数方差。

然后通过 flow 得到 z_p，剩下的和 VITS 一样。
对于 后验编码器PosteriorEncoder，输入的是音频的 mel 谱 特征

## 代码

### 前端

phoneme：
1. 对于中文，用的是 python 的 [pypinyin](https://pypi.org/project/pypinyin/) 库，将输入的文本转为 phoneme
2. 对于英文，用的是 python 的 [g2p_en](https://pypi.org/project/g2p-en/) 库，将输入的文本转为 phoneme
3. 对于中英文混合输入，首先用 LangSegment 将文本分为中文和英文，然后分别用上述两个库转为 phoneme

bert feature：
1. 中文的 bert 特征是用的 huggingface 的 CN_HuBERT 模型（[chinese-roberta-wwm-ext-large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)）
2. 对于其他语种，不用 bert 这个特征（Bert = torch.zeros）

混合语种：
1. 对于混合

### 数据预处理

### 微调

train_s1：训练 GPT
输入：text1 + text2 + wav1-hubert-token
输出：wav2-hubert-token
> 本质就是一个 VALLE
> 实际执行代码：/data/miniconda3/envs/GPTSoVits/bin/python" GPT_SoVITS/s1_train.py --config_file "/workspace/user_code/GPT-SoVITS/TEMP/tmp_s1.yaml；简化后的执行代码：python GPT_SoVITS/s1_train.py --config_file tmp_s1.yaml
> tmp_s1.yaml 来源：加载s1longer.yaml，然后用 data 变量修改其中的一部分参数


train_s2：训练 VITS
输入：wav2-hubert-token + wav1-audio（用于提取音色信息的） + text2
输出：wav2-audio
> 这里还多加了一个 s1 得到的 wav2-hubert-token，也就是前面反复提到的泄漏了音色的 SSL 特征。

> 注意：wav1 reference wav，即用来给音色的，wav2 为 target wav，即要合成的

> 实际执行代码：/data/miniconda3/envs/GPTSoVits/bin/python" GPT_SoVITS/s2_train.py --config "/workspace/user_code/GPT-SoVITS/TEMP/tmp_s2.json；简化后的执行代码：python GPT_SoVITS/s1_train.py --config tmp_s2.json
> tmp_s2.json 来源：加载s2.json，然后用 data 变量修改其中的一部分参数


train_s2：
SynthesizerTrn：来自 VITS 的 generator
freeze_quantizer = True

输入：
+ ssl
+ y：mel 谱
+ y_lenght
+ text：文本
+ text_length


train_s1：
用 Pytorch Lightning 框架
模型：Text2SemanticLightningModule
+ 输入：phoneme_ids、phoneme_ids_len、semantic_ids、semantic_ids_len、bert_feature
+ 内部的模型为 Text2SemanticDecoder
    + 输入为 phoneme_ids、phoneme_ids_len、semantic_ids、semantic_ids_len、bert_feature
    + forward 中，make_input_data 用于准备输入和输出数据：
        + xy_pos：就是对齐之后的用于 GPT 的输入，来自 phoneme_ids 和 bert_feature
        + target：GPT 模型的目标，来自于 semantic_ids
    + infer 的时候，semantic_ids 就变成了 prompt，
+ forward 函数在 AR/models/t2s_model.py 中




数据：Text2SemanticDataModule
+ 包含 train 和 val
+ 用的是 DistributedBucketSampler
+ dataset 用的是 Text2SemanticDataset，有 collate_fn
+ __getitem__ 返回的是
    "idx": idx,
    "phoneme_ids": phoneme_ids,
    "phoneme_ids_len": phoneme_ids_len,
    "semantic_ids": semantic_ids,
    "semantic_ids_len": semantic_ids_len,
    "bert_feature": bert_feature（文本中提取的 feature）
+ 

## 推理

1. 推理对应的代码为：inference_webui 为推理的窗口，里面包含了从数据预处理到推理的所有的代码！如果要流程化，建议基于这个来改。
2. 推理的时候，参考音频会被重采样到 16k
> TODO：所以 32k 音频的作用是什么？

## 其他

1. 关于从零训练：https://github.com/RVC-Boss/GPT-SoVITS/wiki/%E4%BB%8E%E9%9B%B6%E8%AE%AD%E7%BB%83(train-from-scratch)
    1. 从零训练和微调的唯一的区别在于：webui微调训练微操降低了sovits模块文本编码器的lr，底模训练没有
    2. 当然，要改一下 webui，用命令行训练
2. 训练的一些 trick：训练的时候是一个两阶段的训练，先训 VITS，然后用 VITS 得到的量化后的 token 再去训练 GPT
2. 关于训练其他语种的模型：
3. 时间估算：
    1. C0936 估计总时长（16k采样率、16bit、单通道）：大小为 62705054 字节，计算时长 62705054/32000 = 1,959.5329375 s = 32.6588822917 min = 0.54431470486 h
    2. 但是训练的时候会上采样到 32k（不过总时长理论上不变）：大小变为 115600344，计算时长为 115600344/64000 = 1,806.255375 s = 30.10425625 min = 5.01737604167
4. *个人的想法*：
    1. VITS 可能更偏向于负责音质，当然，训练 VITS 的时候，音色也是有的（mel 谱那块给的），所以训好了也可以提升音质
    2. GPT 模块负责文本的正确性（读得对不对）、音色的相似度（音色好不好）




5. ckpt的两种加载方式，一种是加载 log_s2 里面的模型，然后进行 fine tune；另一种是默认的，当 los_s2 里面没有时，按照 except 的代码加载，此时 加载的文件 要和 默认的底模 或者 fine tune 得到的那一系列的模型（weight 的那种） 是一样的。

> 首先加载默认的路径，根据 list 自动找（默认路径是 exp_name/log_s2），找到了就忽略输入的路径，没有的话，则按照 weight 那种格式读取给定路径的模型。



6. 关于 GPT 模块的训练：
> 来自作者的经验：训练acc太高不是件好事，train的acc和loss曲线对最终合成效果的意义不大。


## 模型增训要点

GPT 每次训练要注意调的参数：
+ output_dir：这个是 tensorboard、log、checkpoint 的保存路径，eg: logs/0p5h_data/logs_s1_without_sovits_training
+ exp_name：这个是保存在 GPT_weights 下的文件名，eg: 0p5h_data_without_sovits_training
+ train_phoneme_path：一般保持不变
+ train_semantic_path：一般也保持不变

SoVITS 每次训练要注意调的参数：
+ exp_dir：这个别调了（这个目前仅用于给出 2- 3- 4- 特征文件的路径）
+ s2_ckpt_dir：和 output_dir 作用类似了，eg: logs/0p5h_data/logs_s2_fix_quantizer
+ name：这个是保存在 SoVITS_weights 下的文件名，eg：0p5h_data_fix_quantizer
+ gpu_numbers


目前增训的模型配置：

1. 原始底模
2. 不训 SoVITS，只增训 GPT
    1. SoVITS 可以直接用底模
    2. SoVITS 可以用 3 中训练的 SoVITS
3. fix quantizer 增训 SoVITS：
    1. GPT 可以直接用底模
    2. 也可以用 2 中训练的 GPT（等于2.2）
4. 从零开始训两个模型

5. 还有一个，tune quantizer 训练 SoVITS，然后基于这个 tune 的 SoVITS 训 GPT



关于增训后 loss 异常的尝试：
1. quantizer 是否 fine tune：尝试训练 fix 版本的 quantizer，发现不能解决问题
1. text model 的 lr 的系数的问题：尝试在增训和微调的时候，将两者的 lr 系数调整为一致（都是 1），发现不能解决问题
2. 增训或者微调的时候，没有保存 enc_q（PosteriorEncoder）的问题，尝试用小数据集训练一个 epoch 的模型，保存下来后进行微调：
    1. 保持源代码的配置，不保存 enc_q 的参数，loss 依然异常
    2. 保存 enc_q 的参数，**loss 正常**！
> 为什么源代码保存模型的时候不存 enc_q ？？？（可能是为了节省保存文件的大小。。。） 导致增训的模型也不会加载 enc_q 这个模块的参数，从而在微调的时候进行随机参数初始化，从而微调的时候不匹配。


关于引入静音 token 的想法：
1. 第一种，最简单粗暴：对于文本中的静音 token，直接将其转为 逗号
2. 第二种，修改 bert 特征，记录文本中的静音 token，在提取 bert 特征后插入一个新的 全零 的向量

### 静音 token
目前代码里，有 sp、sp2、sp3 三个静音 token，分别对应于：
+ sp：%（注释了）
+ sp2：￥
+ sp3：^

即，对于输入的文本，检测到上述三个字符，就会将其转为对应的静音 token。
而为了提取对应的文本特征，会将上述三个字符转为 ',' 即逗号。
所以实际输入模型的时候，文本是用逗号分隔的，但是 phoneme 不是逗号，而是上述的静音 token。
如，对于文本：
> 这几十年皇帝换了好几任。

对应的 phoneme 序列为：
> ['zh', 'e4', 'j', 'i3', 'sh', 'ir2', 'n', 'ian2', 'h', 'uang2', 'd', 'i4', 'h', 'uan4', 'l', 'e5', 'h', 'ao2', 'j', 'i3', 'r', 'en4', '.']

而如果引入静音 token（以￥为例），对应的输入文本为：
> 这几十年￥皇帝￥换了￥好几任。

但是实际输入提取 bert 特征是，用的文本是：
> 这几十年,皇帝,换了,好几任。

而此时对应的 phoneme 序列为：
> ['zh', 'e4', 'j', 'i3', 'sh', 'ir2', 'n', 'ian2', 'SP3', 'h', 'uang2', 'd', 'i4', 'SP3', 'h', 'uan4', 'l', 'e5', 'SP3', 'h', 'ao2', 'j', 'i3', 'r', 'en4', '.']

## 推理

1. 原始的推理核心（GPT 部分）：

得到 xy_pos，输入为 transformer decoder 中，得到 xy_dec，通过 ar_predict_layer 得到 logits，对于第一次推理，删除 EOS token 防止一开始就结束。然后进入采样函数，得到采样后的 token，将其拼到 y（y 是 token 序列），同时将当前时刻预测得到的 y_emb 拼接到 xy_pos 中，然后继续下一时刻的预测。直到 EOS token 出现则 break。

采样过程中，首先对输入的 logits 应用某些采样策略，得到 probs 概率，然后采用 gumbel softmax 采样得到 token index。

采样策略：
1. 对应函数 logits_to_probs，输入为 logits、previouse_token、top_k、top_p、repetition_penalty、temperature
2. 首先进行重复惩罚，从 logits 中找出 previouse_token 的位置，然后将对应位置的 logits：
    1. 如果 logits 大于 0，除以 repetition_penalty（惩罚项大于 1）
    2. 如果 logits 小于 0，乘以 repetition_penalty（惩罚项小于 1）
3. top_p 采样
4. 引入 temperature
5. top_k 采样
6. softmax 得到概率

关于原始 vits 的 loss：
![](image/Pasted%20image%2020250208121747.png)