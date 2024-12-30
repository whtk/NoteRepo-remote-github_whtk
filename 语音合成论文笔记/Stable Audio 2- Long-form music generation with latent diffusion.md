> preprint 2024.4，Stability AI
<!-- 翻译 & 理解 -->
<!-- Audio-based generative models for music have seen great
strides recently, but so far have not managed to produce
full-length music tracks with coherent musical structure
from text prompts. We show that by training a generative
model on long temporal contexts it is possible to produce
long-form music of up to 4m 45s. Our model consists of a
diffusion-transformer operating on a highly downsampled
continuous latent representation (latent rate of 21.5 Hz).
It obtains state-of-the-art generations according to met-
rics on audio quality and prompt alignment, and subjective
tests reveal that it produces full-length music with coherent
structure. -->
1. 目前没有可以根据文本提示生成连贯完整音乐的音频生成模型
2. 提出了一个基于 long temporal contexts 的生成模型，可以生成长达 4 分 45 秒的音乐，由 diffusion-transformer 和连续的 latent 表征组成（latent rate 为 21.5 Hz）
3. 模型实现了质量和对齐的 SOTA

## Introduction
<!-- Generation of musical audio using deep learning has been
a very active area of research in the last decade. Initially,
efforts were primarily directed towards the unconditional
generation of musical audio [1, 2]. Subsequently, attention
shifted towards conditioning models directly on musical
metadata [3]. Recent work has focused on adding natural
language control via text conditioning [4–7], and then im-
proving these architectures in terms of computational com-
plexity [8–11], quality [12–15] or controlability [16–19]. -->
