
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
SoVITS 做的是 VC，不做 TTS，因此输入只有两端音频而没有文本。

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

> GPT-SoVITS 约等于 VALLE+VITS

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



## 记录

1. gradio 端口上报：
. /data/bore_run_script/common/util.sh
report "gradio" "${ENV_IP}" "9874" "gradio_token"
2. 占卡代码：nohup /group/30106/yinlinguo/envs/GPTSoVits/bin/python -u /group/30106/goodli/keepworking_v4/run.py > /dev/null 2>&1 &
3. 将公司数据集格式转为 GPT-SoVITS 能够读取的 list 文件的代码：/group/30106/yinlinguo/code/preprocess.py
4. 特征提起的命令行代码：/group/30106/yinlinguo/GPT-SoVITS/kevinmo/get_audio_feature.py、/group/30106/yinlinguo/GPT-SoVITS/kevinmo/get_text_feature.py
5. 现在的 lr 是 0.0001，然后微调的时候，文本模块的学习率乘以了一个 0.4 的系数
6. 训练 VITS 的代码：/group/30106/yinlinguo/GPT-SoVITS/

ls -l | grep "^-" | wc -l
find ./ -type f | wc -l 这个速度比上面的快得多

10k_data_005_lin 是修复命名 bug 之前，用 500 h 提取的数据，但是

/group/30106/yinlinguo/lzy/raw.list
lzy_70h_ft_v1

ckpt的两种加载方式，一种是加载 log_s2 里面的模型，然后进行 fine tune；另一种是默认的，当 los_s2 里面没有时，按照 except 的代码加载，此时 加载的文件 要和 默认的底模 或者 fine tune 得到的那一系列的模型（weight 的那种） 是一样的。

首先加载默认的路径，根据 list 自动找（默认路径是 exp_name/log_s2），找到了就忽略输入的路径，没有的话，则按照 weight 那种格式读取给定路径的模型。


70h 底模：
+ /group/30106/yinlinguo/GPT-SoVITS/logs/10k_data_005_lin/logs_s2/D_233333333333.pth
+ /group/30106/yinlinguo/GPT-SoVITS/logs/10k_data_005_lin/logs_s2/D_233333333333.pth
+ /group/30106/yinlinguo/GPT-SoVITS/SoVITS_weights/10k_data_005_lin_e1_s1177.pth
+ 

150h_500h：
+ VITS：
    + /group/30106/yinlinguo/GPT-SoVITS/SoVITS_weights/500h_data_D_e1_s1457.pth
    + /group/30106/yinlinguo/GPT-SoVITS/SoVITS_weights/500h_data_e1_s1457.pth
    + /group/30106/yinlinguo/GPT-SoVITS/GPT_weights/500h_data-ei.ckpt

关于 GPT 模块的训练：
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
+ gpu_numbers：

取名规则：
+ SoVITS：
    + {name}_fix_quantizer_ft_from_pretrained_{version}
    + {name}_tune_quantizer_train_from_scratch_{version}
    + {name}_tune_quantizer_ft_from_pretrained_{version}
    + eg: 0p5h_data_fix_quantizer_ft_from_pretrained_v1

+ GPT：
    + {name}_train_gpt_from_pretrained_with_sovits==pretrained_s2G488k=={version}
    + {name}_train_gpt_from_scratch_with_sovits=={sovits_name}=={version}
    + {name}_train_gpt_ft_from_pretrained_with_sovits=={sovits_name}=={version}
    + eg: {name}_train_gpt_from_scratch_with_sovits==tune_quantizer_train_from_scratch_v1==v1


模型配置：

1. 原始底模
2. 不训 SoVITS，只增训 GPT
    1. SoVITS 可以直接用底模
    2. SoVITS 可以用 3 中训练的 SoVITS
3. fix quantizer 增训 SoVITS：
    1. GPT 可以直接用底模
    2. 也可以用 2 中训练的 GPT（等于2.2）
4. 从零开始训两个模型

5. 还有一个，tune quantizer 训练 SoVITS，然后基于这个 tune 的 SoVITS 训 GPT


需要测试的：
1. 原始底模（c1）
2. 不训 SoVITS，只增训 GPT，SoVITS 直接用底模（c2）
3. fix quantizer 增训 SoVITS，GPT 则用 2 中增训的 GPT（c3）
4. 从零开始训两个模型（c4）

目前已出的 demo：
A: 在配置 1 下测试所有的说话人：《御姐》、《李泽言》、《范闲》、《妲己》、《吕布》、《猴哥》、《四郎》
B: 在配置 2.1 下，用小说文本测试：《御姐》、《猴哥》

今晚准备出的 demo：
C: 在配置 3.2 下，用小说文本测试：《御姐》、《猴哥》

> 只要 VITS 不训练 或者 训练的时候不改变量化器，得到的 semantic 特征是一样的。


yj_300h_ft_gpt_v1: 2.1 的配置进行 fine tune
yj_300h_ft_scratch_v1: 4 的配置进行 fine tune
yj_300h_ft_v1: 5 的配置进行 fine tune
yj_orig_ft_v1: 1 的配置进行 fine tune
