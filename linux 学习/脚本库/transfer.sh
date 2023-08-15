#!/bin/bash
###
 # @Author: 郭印林 1264514936@qq.com
 # @Date: 2023-06-10 17:45:09
 # @LastEditors: 郭印林 1264514936@qq.com
 # @LastEditTime: 2023-06-11 15:05:06
 # @FilePath: \undefinede:\gyl\gyl的论文和笔记库\笔记库\linux 学习\脚本库\transfer.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

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
s5_port="22"

# 获取参数
src_file="$1"
dst_path="$2"
dst_server="$3"

# 判断源文件是否存在
if [ -e "$src_file" ]; then
    if [ -f "$src_file" ]; then
        echo "File detected."
        scp_command="scp"
    elif [ -d "$src_file" ]; then
        echo "Directory detected."
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

# 使用 sshpass 判断目标路径是否合法
sshpass -p "gyl123" ssh -P "$dst_port" guoyinlin@"$dst_addr" "[ -d $dst_path ]"

# 执行传输命令
sshpass -p "gyl123" $scp_command -P "$dst_port" "$src_file" guoyinlin@"$dst_addr":"$dst_path"

echo "Successully transfered file(s) to $dst_path."