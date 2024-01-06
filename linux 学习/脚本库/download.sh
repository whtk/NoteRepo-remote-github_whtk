#!/bin/bash

# 获取参数
url=$1

# 下载文件
while true; do
  wget -c "$url"
  if [ $? -eq 0 ]; then
    echo "文件下载成功"
    break
  else
    echo "文件下载失败，正在重试..."
    sleep 2
  fi
done