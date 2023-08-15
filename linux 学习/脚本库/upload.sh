#!/bin/bash
###
 # @Author: 郭印林 1264514936@qq.com
 # @Date: 2023-06-11 15:59:17
 # @LastEditors: 郭印林 1264514936@qq.com
 # @LastEditTime: 2023-06-11 15:59:24
 # @FilePath: \undefinede:\gyl\gyl的论文和笔记库\笔记库\linux 学习\脚本库\upload.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

# 设置工作目录
cd /home3/guoyinlin/code/lightning-v2/adapter-asv-spoof-lin

# 添加所有修改的文件
git add .

# 提交更改，并以当天日期作为提交消息
git commit -m "$(date +%Y-%m-%d)"

# 推送更改到远程仓库
git push origin master