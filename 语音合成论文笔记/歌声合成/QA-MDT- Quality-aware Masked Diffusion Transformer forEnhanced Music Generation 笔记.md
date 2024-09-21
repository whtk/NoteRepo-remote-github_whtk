> Preprint 2024，中科大、讯飞
<!-- 翻译 & 理解 -->
<!-- In recent years, diffusion-based text-to-music (TTM) generation has gained prominence, offering an innovative approach to synthesizing musical content from textual descriptions. Achieving high accuracy and diversity in this generation process requires extensive, high-quality data, including
both high-fidelity audio waveforms and detailed text descriptions, which often constitute only a small portion of available
datasets. In open-source datasets, issues such as low-quality
music waveforms, mislabeling, weak labeling, and unlabeled
data significantly hinder the development of music generation models. To address these challenges, we propose a novel
paradigm for high-quality music generation that incorporates
a quality-aware training strategy, enabling generative models to discern the quality of input music waveforms during
training. Leveraging the unique properties of musical signals,
we first adapted and implemented a masked diffusion transformer (MDT) model for the TTM task, demonstrating its
distinct capacity for quality control and enhanced musicality. Additionally, we address the issue of low-quality captions
in TTM with a caption refinement data processing approach.
Experiments demonstrate our state-of-the-art (SOTA) performance on MusicCaps and the Song-Describer Dataset. Our
demo page can be accessed at https://qa-mdt.github.io/. -->
1. 现有的 text-to-music (TTM) 需要高质量的数据，但是开源的数据集中存在低质量音频、错误标注、弱标注和无标注数据等问题
2. 本文提出 quality-aware training strategy，让生成模型在训练时识别输入音频的质量
    1. 先使用 masked diffusion transformer (MDT) 模型实现 TTM
    2. 使用 caption refinement data processing 解决 TTM 中低质量 caption 的问题
3. 在 MusicCaps 和 Song-Describer Dataset 上实现了 SOTA 

## Introduction
<!-- Text-to-music (TTM) generation aims to transform textual
descriptions of emotions, style, instruments, rhythm, and
other aspects into corresponding music segments, providing new expressive forms and innovative tools for multimedia creation. According to scaling law principles (Peebles
and Xie 2023; Li et al. 2024a), effective generative models require a large volume of training data. However, unlike image generation tasks (Chen et al. 2024a; Rombach
et al. 2021), acquiring high-quality music data often presents
greater challenges, primarily due to copyright issues and the
need for professional hardware to capture high-quality music. These factors make building a high-performance TTM
model particularly difficult. -->
1. TTM 将文字描述（情感、风格、乐器、节奏等）转换为音乐片段
2. 获取高质量音乐数据困难（版权）
<!-- In the TTM field, high-quality paired data of text and
music signals is scarce. This prevalent issue of low-quality
data, highlighted in Figure 1, manifests in two primary challenges. Firstly, most available music signals often suffer from distortion due to noise, low recording quality, or outdated recordings, resulting in diminished generated quality,
as measured by pseudo-MOS scores from quality assessment models (Ragano, Benetos, and Hines 2023). Secondly,
there is a weak correlation between music signals and captions, characterized by missing, weak, or incorrect captions,
leading to low text-audio similarity, which can be indicated
by CLAP scores (Wu* et al. 2023). These challenges significantly hinder the training of high-performance music generation models, resulting in poor rhythm, noise, and inconsistencies with textual control conditions in the generated
audio. Therefore, effectively training on large-scale datasets
with label mismatches, missing labels, or low-quality waveforms has become an urgent issue to address. -->
1. TTM 领域缺乏高质量的 text-music pair 数据，且：
    1. 音乐信号质量低
    2. 音乐信号和 caption 之间的相关性弱
<!-- In this study, we introduce a novel quality-aware masked
diffusion transformer (QA-MDT) to enhance music generation. This model effectively leverages extensive, opensource music databases, often containing data of varying
quality, to produce high-quality and diverse music. During training, we inject quantified music pseudo-MOS (pMOS) scores into the denoising stage at multiple granular ities to foster quality awareness, with coarse-level quality
information seamlessly integrated into the text encoder and
fine-level details embedded into the transformer-based diffusion architecture. A masking strategy is also employed to
enhance the spatial correlation of the music spectrum and
further accelerate convergence. This innovative approach
guides the model during the generation phase to produce
high-quality music by leveraging information associated
with elevated p-MOS scores. Additionally, we utilize large
language models (LLMs) and CLAP model to synchronize
music signals with captions, thereby enhancing text-audio
correlation in extensive music datasets. Our ablation studies
on public datasets confirm the effectiveness of our methodology, with the final model surpassing previous works in
both objective and subjective measures. The main contributions of this study are as follows: -->
1. 提出 quality-aware masked diffusion transformer (QA-MDT)：
    1. 在训练时，将 quantified music pseudo-MOS (pMOS) scores 注入到 denoising stage 来提高质量
    2. 使用 masking strategy 增强音乐频谱的空间相关性，加速收敛
    3. 使用 LLMs 和 CLAP model 同步音乐信号和 caption，提高 text-audio 相关性