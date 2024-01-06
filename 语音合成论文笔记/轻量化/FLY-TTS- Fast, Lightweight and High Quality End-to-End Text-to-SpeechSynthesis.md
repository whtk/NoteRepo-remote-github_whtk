
PortaSpeech 

LightSpeech 采用 NAS，搜索的时候非常耗时，且无法泛化到其他语种的数据中。

Nix-TTS 采用知识蒸馏来减少参数，但是 hinder 了端到端的训练。


## 测试

一般会测：
+ 参数
+ RTF
+ 加速比
+ FLOPS