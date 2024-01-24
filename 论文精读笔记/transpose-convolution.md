
## Transpose Convolution 转置卷积（反卷积）
卷积通过下采样来减少输出维度的大小，而转置卷积（transposed convolution）⽤于扭转下采样导致的空间尺⼨减小。对于 stride 为 $1$，kerrnel size为 $2 \times 2$ 的转置卷积，其运算过程如下（设输入的图像尺寸为 $2 \times 2$ ）：
![1658714237606](image/transpose-convolution/1658714237606.png)