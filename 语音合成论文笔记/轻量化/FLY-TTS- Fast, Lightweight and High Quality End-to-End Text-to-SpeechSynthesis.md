
PortaSpeech 

LightSpeech 采用 NAS，搜索的时候非常耗时，且无法泛化到其他语种的数据中。

Nix-TTS 采用知识蒸馏来减少参数，但是 hinder 了端到端的训练。

最接近的工作是 AdaVITS，采用 PPG 作为特征来提高模型稳定性，但是训练的时候需要一个额外的 ASR 模型来从 phoneme 中提取 PPG，推理的时候需要 PPG predictor 

## 测试

一般会测：
+ 参数
+ RTF
+ 加速比
+ FLOPS