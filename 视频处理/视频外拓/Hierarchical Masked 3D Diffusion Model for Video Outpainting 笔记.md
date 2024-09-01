> ACM MM 2023，中科院大学


1. video outpainting 目标是充分填补视频帧边缘的缺失区域
2. 与 image outpainting 相比，模型需要保持填充区域的时间一致性
3. 本文提出用于 video outpainting 的 Hierarchical Masked 3D Diffusion 模型，使用 mask modeling 技术来训练 3D Diffusion，从而可以使用多个 guide frame 来连接多个 video clip 以确保时间一致性
4. 提取视频的 global frame 作为 prompt，采用 cross attention 引导模型获取 除当前 video clip 之外的信息
5. 引入 hybrid coarse-to-fine 的推理 pipeline 来缓解 artifact 积累问题，受益于 mask modeling 的双向学习，在生成 sparse frames 时可以采用 infilling 和 interpolation 的混合策略
6. 实验表明，方法取得了 SOTA 的 video outpainting 结果

## Introduction

1. video outpainting 根据上下文信息（视频的中间部分）扩展视频的边缘区域。需要对运动信息进行建模，以确保视频帧之间的时间一致性（temporal consistency）。且长视频额外的挑战：
    + 由于长时间和 GPU 内存限制，视频会被分成多个 clip。确保同一视频的不同 clip 之间生成内容的时间一致性很难
    + 长视频 outpainting 存在 artifact 积累问题，同时需要大量的计算资源。

2. 现有方法：
    + Dehan 使用 video object segmentation 和 video inpainting 方法形成背景估计，并通过引入 optical flow 来确保时间一致性。但是在相机运动复杂和前景对象离开的情况下结果很差
    + MAGVIT 提出基于 mask 的 video 生成模型，也可用于 video outpainting。引入 3D-Vector-Quantized（3DVQ）tokenizer 来量化视频，把 transformer 用于 multi-task conditional masked token 建模。这种方法能够生成合理的短视频 clip，但对于长视频的多个 clip 组成完整结果，效果会变差。因为无法实现时间一致性，且在多个 clip 推理存在 artifact 累积问题。

3. 本文提出 masked 3D diffusion model（M3DDM）和 hybrid coarse-to-fine 推理 pipeline。方法基于 latent diffusion models（LDMs），LDMs 的好处有：
    + 其在 latent space（而非 pixel space） 中编码视频帧，需要内存更少，效率更高
    + 预训练的 LDMs 可以提供图像先验，帮助模型在 video outpainting 任务中快速收敛

4. 为了确保单个 clip 和同一视频的不同 clip 之间的高时间一致性，采用两种技术：
    + Masked guide frames，可以生成语义一致、与相邻 clip 之间 jitter 更少的 clip。训练时，随机用 raw frames 替换上下文信息，作为 guide frames。模型不仅可以基于上下文信息预测边缘区域，还可以基于相邻 guide frames。推理时，iteratively 和 sparsely 地去 outpaint frames，从而可以使用先前生成的 frames 作为 guide frames。mask modeling 的双向学习模式使模型更好地感知上下文信息，从而更好地推理 single clip。也可以用上 hybrid coarse-to-fine 的推理 pipeline。hybrid pipeline 不仅使用 infilling 策略，还使用 interpolation 策略，多个中间 frames 作为 guide frames。
    + Global video clips 作为 prompts，从完整视频中均匀提取 global frames，使用轻量级 encoder 将其编码为 feature map，然后通过 cross-attention 与当前 video clip 的上下文交互。从而模型在生成当前 clip 时可以看到一些全局信息（输入的视频的 global frames 不包括要填充的边缘区域，以避免泄漏）。
5. 实验表明，在相机运动复杂和前景对象来回移动的场景中，提出的方法可以生成更具时间一致性的完整视频

6. hybrid coarse-to-fine 推理 pipeline 可以缓解长视频 outpainting 中的 artifact 积累问题。由于推理阶段使用 guide frames 迭代生成，前一步生成的不好的结果会污染后面生成的结果。而在长视频生成任务中的 coarse-to-fine 推理 pipeline 中，coarse 阶段首先稀疏生成视频的关键帧，然后根据关键帧密集生成每一帧。与直接密集生成视频相比，coarse 阶段需要更少的迭代，从而缓解了长视频中 artifact 积累问题。现有的 coarse-to-fine 推理 pipeline 使用了三级层次结构，但只使用 infilling 策略，导致了 coarsest 阶段生成的关键帧之间的时间间隔较大，从而降低了生成结果的质量。本文使用 coarse-to-fine 推理 pipeline 进行 video outpainting，由于训练阶段的 masking 策略，可以将 infilling 策略和 interpolation 策略混合在一起，从而不仅可以使用第一帧和最后一帧作为 three-level 的 coarse-to-fine 的 guide，还可以使用多帧插值生成视频
7. 贡献如下：
    1. 首次使用 masked 3D diffusion model 进行 video outpainting，并取得了 SOTA 结果
    2. 提出了双向学习方法，使用 mask modeling 训练 3D diffusion model。同时，使用 guide frames 连接同一视频的不同 clip，可以有效生成具有高时间一致性和低 jitter 的 video outpainting 结果
    3. 从视频的 global frames 中提取全局时间和空间信息作为 prompt，并以 cross-attention 的形式输入网络，引导模型生成更合理的结果
    4. 提出了 hybrid coarse-to-fine 生成 pipeline，在生成 sparse frames 时结合 infilling 和 interpolation

## 相关工作

Diffusion 模型：与 GAN 相比，Diffusion 模型可以生成更丰富多样、质量更高的样本。LDMs 是 latent space 中的 diffusion 模型。

Mask modeling：BERT 随机 mask 句子中的 token，并根据上下文预测 mask 的 token。MAE 在 CV 领域证明了 mask modeling 可以有效用于无监督图像表示学习；mask modeling 也被用于 video generation；mask modeling 和 diffusion model 的结合也可用于 image 和 video generation 。本文不在图像或整个视频帧上应用 mask，而是应用于需要填充的视频周围区域

Coarse-to-Fine Pipeline：在生成长视频时，模型往往会因为自回归策略而遭受 artifact 积累问题。最近的研究采用 coarse-to-fine 生成 pipeline，首先生成稀疏的视频 key frame，通过减少迭代次数来缓解 artifact 积累问题。本文采用 coarse-to-fine 推理 pipeline，并使用 infilling 策略和 interpolation 策略来帮助缓解长视频中的 artifact 积累问题

## 方法

### 预备知识

diffusion 学习数据分布 $p_{data}$。首先向原始分布添加噪声，然后逐渐去噪以恢复原始分布。在 forward 过程中，从 $t=0$ 到 $t=T$ 使用以下公式从 $x_0$ 进行加噪：
$$q_t(x_t|x_{t-1})=\mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_tI).$$

$x_t$ 可以直接从 $x_0$ 中采样：
$$x_t=\sqrt{\widetilde{\alpha}_t}x_0+\sqrt{1-\widetilde{\alpha}_t}\epsilon,$$

其中 $\widetilde{\alpha}_t=\prod_{i=1}^t(1-\beta_i)$，$\epsilon\sim\mathcal{N}(0,1)$。去噪时，训练模型预测 $x_t$ 中的噪声。损失函数为：
$$L_{DM}=\mathbb{E}_{x,\epsilon\sim\mathcal{N}(0,1),t}[\|\epsilon-\epsilon_\theta(x_t,c,t)\|_2^2],$$

其中 $c$ 是条件输入，$t$ 从 $1$ 到 $T$。

LDMs 训练 encoder $E$，将原始 $x_0$ 从像素空间映射到 latent space。然后，decoder $D$ 将 $z_0$ 映射回像素空间。考虑到 video outpainting 任务需要大内存，选择 LDMs 框架作为 pipeline。此外，LDMs 的预训练参数可以作为 image prior，加速收敛。在上式中，将 $x$ 重写为 $z$。

### Masked 3D Diffusion Model

使用 LDMs 的一种简单方法是将原始视频 clip 的 noise latent 与视频 clip 的 context 进行拼接作为条件输入，并训练模型预测噪声。然后从随机采样的高斯噪声分布中恢复原始视频 clip（原始视频）。

但是由于视频通常包含数百帧，模型需要分别对同一视频的不同 clip 进行推理，然后将生成的 clip 拼接在一起形成完整视频的最终 outpainting 结果。在这种情况下，上述简单方法无法保证预测的视频 clip 的时间一致性。

于是提出了 masked 3D diffusion model，其概述如图：
![](image/Pasted%20image%2020240403111813.png)

模型可以一次生成 $F$ 帧。采样不同帧率（fps）的视频帧，并将帧率输入 3D UNet。从而可以用一个统一的模型适应不同帧率的视频。框架和 LDMs 一样，先通过预训练的 encoder $E$ 将视频帧从像素空间映射到 latent space。训练时，每个 context frame 在输入 encoder $E$ 之前以概率 $p_{frame}$ 被 raw video frames 替换。模型在推理阶段可以使用两个以上的 frames 来生成其他 frames。好处：
+ 确保 coarse-to-fine 推理 pipeline 在多次 pass 之间的推理时间一致；
+ 与仅使用第一个或最后一个 raw frames 作为输入条件相比，双向学习可以帮助模型更好地感知上下文信息，从而提高生成质量


Mask 策略：为了构建 video outpainting 的训练样本，随机 mask 每个 frame 的边缘。采用不同的方向策略：四个方向、单方向、双方向（左右或上下）、任意四个方向中的随机方向、mask all。考虑到实际应用场景，采用这五种策略的比例分别为 0.2、0.1、0.35、0.1 和 0.25。"mask all" 策略使模型可以进行无条件生成，从而可以在推理时用 classifier-free guidance。且 mask ratio 从 [0.15, 0.75] 中均匀采样。

为了生成 masked guide frames，有三种情况下替换 contextual frame 为 raw frame：
1. 所有 $F$ frames 只给出上下文信息，每个 frame 都使用上述 masking 策略；
2. 第一个 frame 或第一个和最后一个 frame 被替换为未 mask 的 raw frame，其余 frame 只给出上下文信息；
3. 任何 frame 以概率 $p_{frame}=0.5$ 被替换为未 mask 的 raw frame。

guide frames 可以帮助模型预测边缘区域，不仅基于上下文信息，还基于相邻 guide frames。相邻 guide frames 可以帮助生成更一致、更少 jitter 的结果。三种情况的训练比例均匀分布，分别为 0.3、0.35 和 0.35。


全局视频 clip 作为 prompt：为了使模型感知超出当前 clip 的全局视频信息，从视频中均匀采样 $g$ 帧。这些 global frames 通过可学习的轻量级 encoder 得到 feature map，然后通过 cross-attention 输入 3D-UNet。
> 不在 3D-UNet 的输入层中输入 global frames，因为 cross-attention 可以帮助 masked frames 与 global frames 更好地交互。global frames 与当前 video clip 的上下文对齐，并与其他 frames 一样被 mask 以避免信息泄漏。

Classifier-free Guidance：Classifier-free Guidance 可以改善条件生成，其中隐式分类器 $p_\theta(c|z_t)$ 为条件 $c$ 分配高概率。这里有两个条件输入。一个是视频的 context $c_1$，另一个是 global video clip $c_2$。通过随机设 $c_1$ 和 $c_2$ 为一个固定的空 $\emptyset$ 的概率 $p_1$ 和 $p_2$ 来同时训练无条件和条件模型。推理时，遵循 Brooks 的方法，使用条件和无条件分数估计的线性组合：
$$\begin{aligned}\hat{\epsilon}(z_t,c_1,c_2)=\epsilon(z_t,\emptyset,\emptyset)+s_1(\epsilon(z_t,c_1,\emptyset)-\epsilon(z_t,\emptyset,\emptyset))\\+s_2(\epsilon(z_t,c_1,c_2)-\epsilon(z_t,c_1,\emptyset)),\end{aligned}$$

其中 $s_1$ 和 $s_2$ 是 guidance scale，用于控制生成的视频更多地依赖于视频的 context 还是全局 frames。


### Hybrid Coarse-to-Fine Pipeline for Video Outpainting

在 video generation 任务中，生成长视频往往会导致 artifact 积累，从而导致性能下降。最近的研究使用 hierarchical 结构首先生成视频的稀疏关键帧，然后使用 infilling 策略填充密集视频帧。infilling 策略需要第一个和最后一个 frame 作为 guide frames 来指导下一级的生成。然而，仅使用 infilling 可能导致 coarse 阶段帧之间的时间间隔较大。例如，如图所示，如果仅使用 infilling 策略，模型在 coarsest level 需要 225 的帧间隔而不是 30。由于问题的难度和训练集中缺乏长视频数据，这可能导致结果很差。

由于双向学习，我们的 3D UNet 可以通过结合 infilling 和 interpolation 来进行 video outpainting。这避免了 coarse 生成阶段帧间隔较大的问题。我们的 coarse-to-fine 过程如下图所示。pipeline 分为三个层次：
1. 在第一级（coarse）中，无条件生成第一个视频 clip，然后基于上一次迭代的最后一帧结果迭代生成所有关键帧。
2. 在第二级（coarse）中，使用第一级生成的关键帧作为条件输入，通过插值生成更多关键帧。
3. 在第三级（fine）中，使用第一个和最后一个 frame 作为 guide frames，生成最终的 video outpainting 结果，帧间隔为 1。

## 实验（略）
