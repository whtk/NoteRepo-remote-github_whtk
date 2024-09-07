> ICASSP 2024，厦门大学

1. DDPM 和 score-based 模型效率很低，导致采样耗时很大
2. 提出 ReFlow-TTS，本质为 ODE 模型，以尽可能直的 path 将高斯分布转为 GT mel 谱 分布
	1. 模型可以在单个采样步下实现高质量的合成，不需要 teacher model
3. 在 LJSpeech 数据集上实验，效果超过 diffusion based 模型

## Introduction

