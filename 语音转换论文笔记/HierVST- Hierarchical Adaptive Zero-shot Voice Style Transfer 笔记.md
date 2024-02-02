> InterSpeech 2023，高丽大学

1. 最近的 zero-shot VST 系统仍然缺乏将新说话人的语音风格迁移的能力
2. 提出 HierVST，一个 hierarchical adaptive end-to-end zero-shot VST 模型，采用 hierarchical variational inference 和 self-supervised representation，在不需要文本的情况下，只使用语音数据集训练模型
3. 采用 hierarchical adaptive generator，逐步生成 pitch representation 和音频
4. 采用无条件生成，提高 acoustic representation 下的说话人相关的 acoustic capacity

## Introduction

1. 现有的 VC 模型仍然在 speaker adaptation 上性能不好，且要文本才能从 speech 中解耦 linguistic representations
2. [HierSpeech- Bridging the Gap between Text and Speech by Hierarchical Variational Inference using Self-supervised Representations for Speech Synthesis 笔记](../语音合成论文笔记/HierSpeech-%20Bridging%20the%20Gap%20between%20Text%20and%20Speech%20by%20Hierarchical%20Variational%20Inference%20using%20Self-supervised%20Representations%20for%20Speech%20Synthesis%20笔记.md) 采用语音自监督表征从语音中提取 linguistic representation，但是还是需要文本来 regularize
3. 提出 HierVST，一个 hierarchical adaptive end-to-end VST 系统：
    1. 采用 multi-path 自监督语音表征，从单段语音通过 perturbed speech 来恢复 speaker-agnostic linguistic representation，从原始语音中提取 speaker-related linguistic representation
    2. 引入 hierarchical adaptive generator (HAG) with source modeling，通过 hierarchical variational inference 连接多个表征
    3. 提出 prosody distillation 来增强 linguistic representation，提出无条件生成来提高 acoustic capacity
4. HierVST 在 zero-shot VST 上，音频质量和说话人相似度都优于其他模型

## HierVST

结构如图：
![](image/Pasted%20image%2020240201105408.png)

### Speech representation

先将语音分解为 perturbed linguistic representation, linguistic representation 和 acoustic representation，然后从解耦后的表征中重构语音。采用 high-resolution linear spectrogram 提取 acoustic representation。从 Mel-spectrogram 提取 style representation。

具体而言：

1. 从 XLS-R 的中间层提取特征 $x_{w2v}$。为了做解耦，提出 multi-path 自监督语音表征，采用 data perturbation 来减少来自同一个自监督语音模型的 content-irrelevant features。perturbed speech 中得到的 $x_{w2v,pert}$ 送到 linguistic restorer 来恢复 linguistic representation。原始语音中得到的 $x_{w2v}$ 送到 linguistic encoder 来提取 enhanced linguistic representation。

2. 对于全局的 voice style representation，从 mel 谱提取 style representation。采用 style encoder 提取 style representation。对于 hierarchical style adaptation，style representation 送到所有网络中，包括 linguistic restorer, linguistic encoder, acoustic encoder, normalizing flow modules 和 HAG。对于 zero-shot VST，不使用 speaker ID 信息，只从 speech 中提取 style representation。

### Hierarchical variational autoencoder

采用 [HierSpeech- Bridging the Gap between Text and Speech by Hierarchical Variational Inference using Self-supervised Representations for Speech Synthesis 笔记](../语音合成论文笔记/HierSpeech-%20Bridging%20the%20Gap%20between%20Text%20and%20Speech%20by%20Hierarchical%20Variational%20Inference%20using%20Self-supervised%20Representations%20for%20Speech%20Synthesis%20笔记.md) 的结构，用 linguistic restorer 替换 text encoder。采用 perturbed linguistic representation $x_{w2v,pert}$ 作为 conditional information $c$ 来 hierarchically 生成音频。还用了原始波形得到的自监督表征中的 enhanced linguistic representation。采用 linear spectrogram 提取 acoustic representation。为了连接 acoustic 和 multi-path linguistic representations，采用 hierarchical variation inference，HierVST 的优化目标如下：
$$\begin{aligned}
{\log p_\theta(x|c)}& \large\geq\mathbb{E}_{q_\phi(z|x)}\left[\log p_{\theta_d}(x|z_a)\right.  \\
&-\log\frac{q_{\phi_a}(z_a|x_{spec})}{p_{\theta_a}(z_a|z_l)}-\log\frac{q_{\phi_l}(z_l|x_{w2v})}{p_{\theta_l}(z_l|c)}]
\end{aligned}$$
其中 $q_{\phi_a}(z_a|x_{spec})$ 和 $q_{\phi_l}(z_l|x_{w2v})$ 分别是 acoustic 和 linguistic representations 的近似后验。$p_{\theta_l}(z_l|c)$ 表示 linguistic latent variables $z_l$ 的先验分布，$p_{\theta_a}(z_a|z_l)$ 表示 acoustic latent variables $z_a$ 的先验分布，$p_{\theta_d}(x|z_a)$ 是 HAG 产生数据 $x$ 的似然函数。此外，采用 normalizing flow 来提高每个 linguistic representation 的表达能力。对于重构损失，采用 HAG 中的多个重构项。

### Hierarchical adaptive generator

结构如图：
![](image/Pasted%20image%2020240201105319.png)

对于端到端 VC，引入 HAG $G$，包括 source generator $G_s$ 和 waveform generator $G_w$。生成的 acoustic representation $z_a$ 和 style representation $s$ 送到 $G_s$ 来生成 refined pitch representation $p_h$，用辅助的 F0 predictor 用来 enforce $p_h$ 上的 F0 信息：
$$\begin{aligned}L_{pitch}&=\|p_x-G_s(z_a,s)\|_1,\end{aligned}$$
其中 $p_x$ 是 ground-truth (GT) log-scale F0。然后，$G_w$ 从 $z_a,p_h,s$ hierarchically 合成音频，采用 STFT 和 Mel-filter $\psi$ 将波形转换为 Mel-spectrogram，用 GT 和生成的 Mel-spectrogram 之间的重构损失：
$$\begin{aligned}L_{STFT}=\|\psi(x)-\psi(G_w(z_a,p_h,s))\|_1.\end{aligned}$$

此外，采用 adversarial training 来提高音频质量。采用 multi-period discriminator (MPD) 和 multi-scale STFT discriminator (MS-STFTD)：
$$\begin{aligned}\mathcal{L}_{adv}(D)&=\mathbb{E}_{(x,z_a)}\bigg[(D(x)-1)^2+D(G(z_a,s))^2\bigg],\\\\\mathcal{L}_{adv}(\phi_a,\theta_d)&=\mathbb{E}_{(z_a)}\bigg[(D(G(z_a,s))-1)^2\bigg]\end{aligned}$$

### 韵律蒸馏

引入韵律蒸馏来从 linguistic encoder 中提取 enhanced linguistic representation $z_l$。$z_l$ 送到 prosody decoder 来生成包含 prosody representation 的前 20 个 mel bin。让 $z_l$ 获取 speaker-related prosody information 来增强 linguistic information。采用 prosody loss $L_{prosody}$，最小化 GT 和重构的 Mel-spectrogram 之间的 l1 距离。

### 无条件生成

对于 speaker adaptation，采用 style representation 作为 condition。因此，引入 unconditional generation 来增加 acoustic representation 上的 speaker characteristic，以便进行 progressive speaker adaptation。其实就是将 style representation $s$ 以 10% 的概率替换为 null speaker embedding $\emptyset$，这样我们可以将单个模型视为的 conditional 和 unconditional。

## 实验（略）
