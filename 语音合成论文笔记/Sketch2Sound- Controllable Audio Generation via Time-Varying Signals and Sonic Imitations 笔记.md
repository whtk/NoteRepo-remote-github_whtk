> Adobe、西北大学
<!-- 翻译 & 理解 -->
<!-- Abstract—We present Sketch2Sound, a generative audio model capable
of creating high-quality sounds from a set of interpretable time-varying
control signals: loudness, brightness, and pitch, as well as text prompts.
Sketch2Sound can synthesize arbitrary sounds from sonic imitations
(i.e., a vocal imitation or a reference sound-shape). Sketch2Sound can
be implemented on top of any text-to-audio latent diffusion transformer
(DiT), and requires only 40k steps of fine-tuning and a single linear
layer per control, making it more lightweight than existing methods like
ControlNet. To synthesize from sketchlike sonic imitations, we propose
applying random median filters to the control signals during training,
allowing Sketch2Sound to be prompted using controls with flexible levels
of temporal specificity. We show that Sketch2Sound can synthesize sounds
that follow the gist of input controls from a vocal imitation while retaining
the adherence to an input text prompt and audio quality compared to
a text-only baseline. Sketch2Sound allows sound artists to create sounds
with the semantic flexibility of text prompts and the expressivity and
precision of a sonic gesture or vocal imitation. Sound examples are
available at https://hugofloresgarcia.art/sketch2sound/. -->
1. 提出 Sketch2Sound，可以从时变控制信号（loudness、brightness、pitch 和文本提示）生成高质量音频，可以从 sonic imitations（声音模仿或参考声音形状）合成任意声音
2. Sketch2Sound 可以在任何 T2A latent DiT 中实现，只需 40k 步微调，比 ControlNet 更轻量
3. 为了从 sketchlike sonic imitations 合成音频，提出在训练时对控制信号进行随机中值滤波

## Introduction
<!-- Sound design is the craft of storytelling through sonic composition.
Within sound design, Foley sound is a technique where special sound
effects are designed and performed in sync to a film during post-
production [1]. These sound scenes are typically performed by a
Foley artist on a stage equipped with abundant sound instruments
and other soundmaking materials1. Foley sound is a skilled and
gestural performance art: performing a sound scene with sound-
making objects and instruments (instead of arranging pre-recorded
samples post hoc) allows sound artists to create fluent and temporally
aligned sounds with a “human” (i.e., gestural) touch. Adding this
gestural touch to the resulting sound composition often results in a
sonic product of great aesthetic and production value. -->
