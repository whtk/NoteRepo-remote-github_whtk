
# Denoising Diffusion Probabilistic Models 笔记

原论文 [[Denoising Diffusion Probabilistic Models.pdf]]
代码在  https://github.com/hojonathanho/diffusion

1. 使用扩散模型合成高质量的图像
2. 通过结合扩散概率模型和朗之万动力学动力学匹配的降噪分数，设计加权变分界
3. 256 \* 256 的LSUN上，效果类似于 ProgressGAN 

## Introduction
1. GAN、VAE、自回归、flow 等模型在图像和音频方向的合成能力很强大，基于能量的模型（EBM）和得分匹配产生的图像也和GAN相当。
2. 本文介绍了扩散概率模型（DPM），是使用变分推理训练的参数化马尔科夫链，以在有限的时间后生成与数据匹配的样本。
3. 当扩散模型包含少量高斯噪声时，可以直接将转换链设置为条件高斯。
4. 扩散模型的优点在于：定义简单，训练效率高，而且可以生成高质量的样本