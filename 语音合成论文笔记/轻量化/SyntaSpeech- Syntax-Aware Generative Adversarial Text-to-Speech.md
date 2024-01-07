> IJCAI 2022，renyi，ZJU

1. NAR-TTS 通常采用 phoneme 序列作为输入，无法理解树结构的语义信息（tree-structured syntactic informatio）
2. 提出 SyntaSpeech，一种 syntax-aware 的 light-weight NAR-TTS 模型，将树结构的语义信息集成到 PortaSpeech 的韵律建模中：
	1. 基于输入句子的依赖建立语法图，然后采用语法图来进行 text encoding 来提取语义信息
	2. 将提取到的 syntactic encoding 引入 PortaSpeech 提高韵律预测
	3. 引入 multi-length discriminator 来替换 PortaSpeech 中的 fow-based post-net
3. 不仅可以合成 expressive prosody 的音频，而且可以泛化到多语种、多说话人 TTS

## Introduction

