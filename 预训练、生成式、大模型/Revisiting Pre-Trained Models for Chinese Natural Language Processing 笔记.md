> EMNLP 2020，哈工大、讯飞科技
<!-- 翻译 & 理解 -->
<!-- Bidirectional Encoder Representations from Transformers (BERT) has shown marvelous improvements across various NLP tasks, and consecutive variants have been proposed to further improve the performance of the pre- trained language models. In this paper, we target on revisiting Chinese pre-trained lan- guage models to examine their effectiveness in a non-English language and release the Chi- nese pre-trained language model series to the community. We also propose a simple but effective model called MacBERT, which im- proves upon RoBERTa in several ways, espe- cially the masking strategy that adopts MLM as correction (Mac). We carried out extensive experiments on eight Chinese NLP tasks to revisit the existing pre-trained language mod- els as well as the proposed MacBERT. Ex- perimental results show that MacBERT could achieve state-of-the-art performances on many NLP tasks, and we also ablate details with sev- eral findings that may help future research.1 -->
1. 本文关注中文的预训练语言模型
2. 提出 MacBERT，在 RoBERTa 的基础上改进，采用 MLM as correction (Mac) 的 masking 策略
3. 在八个中文 NLP 任务上进行实验，可以达到 SOTA

## Introduction