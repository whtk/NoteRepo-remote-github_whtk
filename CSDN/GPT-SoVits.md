
## 特点

zero shot TTS：5s 参考音频

few shot TTS：1min 训练数据

webui 支持

语音：中日英，且跨语言

## 微调

1. 下载预训练模型
2. 下载 ASR 模型
3. 数据预处理：
	1. 用的数据为 干声（即无伴奏人声，有伴奏的话用 UVR5 去除）
	2. 填路径、分割语音，得到 slice 文件
	3. 进行 ASR 识别，得到 list 文件
	4. 校对 list
	5. 填入 list 和 音频 文件路径，预处理，得到文件：
		1. 2-xxx：name2text
		2. 3-xxx：SSL 特征，pt 后缀
		3. 4-xxx：中文 SSL 特征，pt 后缀
		4. 5-xxx：音频