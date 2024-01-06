> InterSpeech 2022，Kakao Enterprise Corporation，韩国

1. TTS 中的 two-stage 效果很好，但是训练流程过于复杂
2. 提出端到端的 TTS，联合训练  FastSpeech2 和 HiFi-GAN，通过在联合训练框架中引入一个对齐学习目标函数，移除了外部的 speech-text 对齐工具
3. 