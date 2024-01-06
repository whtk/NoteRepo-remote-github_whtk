> ICLR 2023，University of Waterloo，Baidu Research

1. 提出 Alignment-Aware Acoustic-Text Pretraining (A3T)，在训练时，将 文本 和 声学-文本 对齐 作为输入来重构掩码声学信号
2. 从而预训练的模型可以用于生成高质量的重构 spectrogram，从而可以直接用于语音编辑或未知说话人的 TTS
3. 在语音编辑任务上超过了 SOTA，提升了多说话人的语音合成性能

## Introduction

1. 之前的语音表征学习都是用在 speech understanding 的任务上，其输入为语音，但是没有用在合成任务中的
2. 提出 A3T，引入 cross-modal alignment embeddings，从而可以在多莫模态预训练中更好地学习 acoustic 和 phoneme 之间的对齐

## 之前的工作（略）

## A3T 预训练



## 实验