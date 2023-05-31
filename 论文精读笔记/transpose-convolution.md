<!--
 * @Author: error: git config user.name && git config user.email & please set dead value or install git
 * @Date: 2022-07-25 09:50:54
 * @LastEditors: error: git config user.name && git config user.email & please set dead value or install git
 * @LastEditTime: 2022-07-25 09:57:18
 * @FilePath: \论文笔记\transpose-convolution.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
## Transpose Convolution 转置卷积（反卷积）
卷积通过下采样来减少输出维度的大小，而转置卷积（transposed convolution）⽤于扭转下采样导致的空间尺⼨减小。对于 stride 为 $1$，kerrnel size为 $2 \times 2$ 的转置卷积，其运算过程如下（设输入的图像尺寸为 $2 \times 2$ ）：
![1658714237606](image/transpose-convolution/1658714237606.png)