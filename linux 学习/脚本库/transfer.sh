#!/bin/bash
###
 # @Description: scp 跨服务器传输文件及文件夹
 # @Autor: 郭印林
 # @Date: 2023-06-11 15:06:44
 # @LastEditors: 郭印林
 # @LastEditTime: 2024-02-14 14:56:49
 # @Email: 22231138@zju.edu.cn
### 

# 更新说明：
# 1. 避免了对于未连接过的主机，需要输入yes进行确认
# 2. 添加了传输文件细节和进度条显示

# 使用示例：
# bash transfer.sh 源文件 目标路径 目标服务器 用户名 密码
# e.g.
# bash transfer.sh scr_file tgt_path s_218 guoyinlin gyl123


# 定义服务器信息
s1_addr="10.13.71.37"
s1_port="4001"

s2_addr="10.13.71.37"
s2_port="3001"

s3_addr="10.13.71.37"
s3_port="3002"

s4_addr="10.115.107.217"
s4_port="10022"

s5_addr="10.115.107.218"
s5_port="21822"

# 获取参数
src_file="$1"
dst_path="$2"
dst_server="$3"
username="$4"
password="$5"

# 判断源文件是否存在
if [ -e "$src_file" ]; then
    if [ -f "$src_file" ]; then
        echo "检测到文件"
        scp_command="scp"
    elif [ -d "$src_file" ]; then
        echo "检测到目录"
        scp_command="scp -r"
    else
        echo "Error: Invalid source file or directory."
        exit 1
    fi
fi


# 判断服务器名是否合法
case "$dst_server" in
  s_4001) dst_addr="$s1_addr"; dst_port="$s1_port";;
  s_3001) dst_addr="$s2_addr"; dst_port="$s2_port";;
  s_3002) dst_addr="$s3_addr"; dst_port="$s3_port";;
  s_217) dst_addr="$s4_addr"; dst_port="$s4_port";;
  s_218) dst_addr="$s5_addr"; dst_port="$s5_port";;
  *) echo "Error: Invalid destination server name."; exit 1;;
esac



ssh_options="-o StrictHostKeyChecking=no"

sshpass -p "$password" ssh -p "$dst_port" $ssh_options "$username@$dst_addr" "[ -d $dst_path ]"
echo "开始传输"

scp_command="scp -r"
# sshpass -p "$password" $scp_command -P "$dst_port" "$src_file" "$username@$dst_addr:$dst_path" 
rsync -avz --progress -e "sshpass -p '$password' ssh -p $dst_port $ssh_options" "$src_file" "$username@$dst_addr:$dst_path"

echo "Successfully transferred file(s) to $dst_path."

