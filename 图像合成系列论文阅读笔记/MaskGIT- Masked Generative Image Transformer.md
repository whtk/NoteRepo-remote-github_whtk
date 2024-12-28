> ICCV 2023，Google
<!-- Generative transformers have experienced rapid popu-
larity growth in the computer vision community in synthe-
sizing high-fidelity and high-resolution images. The best
generative transformer models so far, however, still treat an
image naively as a sequence of tokens, and decode an image
sequentially following the raster scan ordering (i.e. line-
by-line). We find this strategy neither optimal nor efficient.
This paper proposes a novel image synthesis paradigm us-
ing a bidirectional transformer decoder, which we term
MaskGIT. During training, MaskGIT learns to predict ran-
domly masked tokens by attending to tokens in all direc-
tions. At inference time, the model begins with generating
all tokens of an image simultaneously, and then refines the
image iteratively conditioned on the previous generation.
Our experiments demonstrate that MaskGIT significantly
outperforms the state-of-the-art transformer model on the
ImageNet dataset, and accelerates autoregressive decoding
by up to 64x. Besides, we illustrate that MaskGIT can be
easily extended to various image editing tasks, such as in-
painting, extrapolation, and image manipulation. -->
1. 现有的 generative transformer 将图像视为 token 序列，按照 raster scan 顺序（逐行）解码，不高效
