#!/bin/bash
###
 # @Author: 郭印林 1264514936@qq.com
 # @Date: 2023-06-10 17:19:07
 # @LastEditors: GuoYinlin-Mac 1264514936@qq.com
 # @LastEditTime: 2023-11-11 22:23:07
 # @FilePath: \undefinede:\gyl\gyl的论文和笔记库\笔记库\linux 学习\bash 脚本\create_user.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

# 获取参数
username=$1
directory=$2
password=$3
is_root=$4

# 判断目录是否存在，如果不存在则创建此目录
if [ ! -d "$directory" ]; then
  mkdir -p "$directory"
fi

# 判断用户名是否存在
if id "$username" >/dev/null 2>&1; then
  echo "用户名已存在"
  exit 1
fi

# 创建用户
useradd -d "$directory" -m -s /bin/bash "$username"

# 设置密码
echo "$username:$password" | chpasswd

# 是否赋予 root 权限
if [ "$is_root" = "true" ]; then
  usermod -aG sudo "$username"
fi

# 赋予目录权限
chown -R "$username":"$username" "$directory"

# 输出结果
echo "用户 $username 创建成功"
