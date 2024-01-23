> InterSpeech 2020，三星电子

1. 提出端到端 TTS，可以在 CPU 上实现实时 TTS
2. 包含 attention-based 自回归 seq2seq 声学模型 +  LPCNet vocoder 来生成波形
3. 声学模型采用 Tacotron 1 和 2，用了 purely location-based attention 来确保稳定性
4. 推理时，decoder 