> Meta，2022

1. 提出一种 audio codec，包含 streaming encoder-decoder 架构，latent space 是量化的，训练方式是端到端的
2. 使用一个 multiscale spectrogram adversary 来加速训练，减少伪影
3. 引入一个新的 loss balancer mechanism 来稳定训练
4. 研究用 lightweight Transformer 来进一步压缩得到的表征
