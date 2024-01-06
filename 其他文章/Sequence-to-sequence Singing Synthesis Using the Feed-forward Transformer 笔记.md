> ICASSP 2020

1. 提出 sequence-to-sequence 的歌声合成，不需要序列数据包含  pre-aligned phonetic and acoustic features 
2. 采用了一个适用于 feed-forward synthesis 的注意力机制
3. 由于 phonetic timings 受乐谱限制，采用一个 duration model 得到近似的 initial alignment，然后采用基于  feed-forward 的 decoder 来 refines  这些 initial alignment 得到目标声学特征
4. 优点是，推理快，没有自回归模型的 exposure bias 问题

## Introduction

## 系统