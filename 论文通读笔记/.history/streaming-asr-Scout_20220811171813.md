<!--
 * @Description: Low Latency End-to-End Streaming Speech Recognition with a Scout Network 笔记
 * @Autor: 郭印林
 * @Date: 2022-08-11 17:14:45
 * @LastEditors: 郭印林
 * @LastEditTime: 2022-08-11 17:17:13
-->

## Low Latency End-to-End Streaming Speech Recognition with a Scout Network 笔记
1. 基于Transformer，提出了一种新的流式语音识别模型
2. 包含一个 scout network 和 一个 recognition network
3. scout network 不看未来的帧来检测整个 word boundary
4. 