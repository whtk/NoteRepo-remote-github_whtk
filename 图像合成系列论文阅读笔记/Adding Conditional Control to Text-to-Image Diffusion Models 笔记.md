> ICCV 2023，Stanford
<!-- We present ControlNet, a neural network architecture to
add spatial conditioning controls to large, pretrained text-
to-image diffusion models. ControlNet locks the production-
ready large diffusion models, and reuses their deep and ro-
bust encoding layers pretrained with billions of images as a
strong backbone to learn a diverse set of conditional controls.
The neural architecture is connected with “zero convolutions”
(zero-initialized convolution layers) that progressively grow
the parameters from zero and ensure that no harmful noise
could affect the finetuning. We test various conditioning con-
trols, e.g., edges, depth, segmentation, human pose, etc., with
Stable Diffusion, using single or multiple conditions, with
or without prompts. We show that the training of Control-
Nets is robust with small (<50k) and large (>1m) datasets.
Extensive results show that ControlNet may facilitate wider
applications to control image diffusion models. -->
1. 提出 ControlNet，在大型预训练的 T2I diffusion 模型中添加条件控制
    1. ControlNet 锁住训练好的 diffusion 模型，利用训练好的模型作为 backbone，学习多种条件控制
    2. 使用 zero convolutions 逐渐增加参数，确保微调过程中不会受噪声的影响
2. 测试多种条件控制，如 edges, depth, segmentation, human pose 等、单个或多个条件、有无 prompt 
3. ControlNet 的训练在小（<50k）和大（>1m）数据集上都很稳健

## Introduction
<!-- Many of us have experienced flashes of visual inspiration
that we wish to capture in a unique image. With the advent
of text-to-image diffusion models [54, 62, 72], we can now
create visually stunning images by typing in a text prompt.
Yet, text-to-image models are limited in the control they
provide over the spatial composition of the image; precisely
expressing complex layouts, poses, shapes and forms can be
difficult via text prompts alone. Generating an image that
accurately matches our mental imagery often requires nu-
merous trial-and-error cycles of editing a prompt, inspecting
the resulting images and then re-editing the prompt. -->
1. 