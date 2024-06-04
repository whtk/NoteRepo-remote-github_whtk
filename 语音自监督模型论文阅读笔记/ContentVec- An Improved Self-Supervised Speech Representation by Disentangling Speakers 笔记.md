> ICML 2022，MIT-IBM Watson AI Lab
<!-- 翻译 & 理解 -->
<!-- Self-supervised learning (SSL) in speech involves training a speech representation network on a large-scale unannotated speech corpus, and then applying the learned representations to down- stream tasks. Since the majority of the down- stream tasks of SSL learning in speech largely fo- cus on the content information in speech, the most desirable speech representations should be able to disentangle unwanted variations, such as speaker variations, from the content. However, disentan- gling speakers is very challenging, because remov- ing the speaker information could easily result in a loss of content as well, and the damage of the latter usually far outweighs the benefit of the for- mer. In this paper, we propose a new SSL method that can achieve speaker disentanglement without severe loss of content. Our approach is adapted from the HuBERT framework, and incorporates disentangling mechanisms to regularize both the teachers (masked prediction labels) and the stu- dents (learned representations). We evaluate the benefit of speaker disentanglement on a set of content-related downstream tasks, and observe a consistent and notable performance advantage of our speaker-disentangled representations.1 -->
1. 理想的 SSL 需要从 content 中解耦 unwanted variations 如 speaker variations，但是很困难，因为去除 speaker 信息可能会导致内容丢失
2. 提出一种新的 SSL 方法，可以实现 speaker disentanglement 而不严重丢失 content：
    1. 基于 HuBERT 框架，加入 disentangling 机制来规范 teachers 和 students
3. 在一系列 content-related downstream tasks 上评估 speaker disentanglement 的效果，效果很好

## Introduction
<!--  -->