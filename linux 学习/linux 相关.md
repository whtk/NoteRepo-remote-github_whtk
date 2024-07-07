
https://pypi.tuna.tsinghua.edu.cn/simple

### linux 常用命令

1.  查看文件内容
    1.  cat {file} — 查看全部
    2.  cat -n {file} — 输出行号查看全部
    3.  more {file} — 只可向下翻页查看全部，q退出查看后终端还是有文件内容
    4.  less {file} — 可上下翻页查看全部，q退出查看后终端不会显示文件内容
2.  解压 tar
    1.  tar -xvf {file} — 解压文件到当前目录
    2.  tar -xvf {file} -C {dir} — 解压文件到{dir}目录
3. zip 压缩
	1. `zip -r file.zip dir/`
	2. 跳过某个目录 -x '' （引号必须有，引号内部写相对目录位置）
4.  统计文件数量
    1.  wc -l — 列出行数
        1.  统计文件夹中**文件**的个数：`ls -l | grep "^-" | wc -l`
        2.  统计文件夹中**文件夹**的个数：`ls -l | grep "^d" | wc -l`
        3. 统计特定后缀的文件个数：`find . -type f -name "星.txt" | wc -l ``
            1. 实际使用发现，对于大量文件的统计（大于 10w 个），`find` 命令的速度比 `ls` 快得多
    2.  wc -w — 列出英文单词数
    3.  wc -m — 列出字符数
5.  磁盘相关
    1.  查看当前文件夹下的所有文件大小：du -sh *
    2. 查看磁盘占用：df -h
6.  查看cpu、gpu占用率等情况：
    1.  nvidia-smi查看 gpu占用率
    2.  top 查看 cpu 使用情况
        -   %us：表示用户空间程序的cpu使用率（没有通过nice调度）
        -   %sy：表示系统空间的cpu使用率，主要是内核程序。
        -   %ni：表示用户空间且通过nice调度过的程序的cpu使用率。
        -   %id：空闲cpu
        -   %wa：cpu运行时在等待io的时间
        -   %hi：cpu处理硬中断的数量
        -   %si：cpu处理软中断的数量
        -   %st：被虚拟机偷走的cpu
7.  文件传输
    1.  Linux2Win：scp -P 10022 [guoyinlin@10.115.110.54](mailto:guoyinlin@10.115.110.54):/home2/guoyinlin/path/filename path/filename
    2.  Win2Linux：scp -P 4001 /path/filename guoyinlin@10.13.71.37:/home3/guoyinlin/
    3. linux 互传：scp -r -P 10022  /home/guoyinlin/test.txt guoyinlin@10.13.71.37:/home/destination（-r 表示传输文件夹）
8.  ls 命令：
    1.  ls -a 查看隐藏文件
    2. ls -l 查看详细信息
    3. ls -h 显示人类可读的大小
9. 文件复制：
	1. cp -i 用于防止覆盖（不加的话会直接覆盖已有文件）
	2. cp -r 用于复制目录
10. 文件（夹）移动或重命名
	1. mv -i 用于防止覆盖
11. 文件删除：
	1. rm -i 用于再次确认
	2. rm -r 用于删除目录，推荐使用 rm -i -r 每次都检查目录文件
12. 查看文件的前/后 k 行：
	1. 前：`head -n <k> file.txt`
    2. 后：`tail -n <k> file.txt`
13. ps 进程管理
	1. `ps -ef` 详细列出所有的进程
    2. `ps aux | grep <name>` 查找名字为 `<name>` 的进程
14. 数据搜索
	1. 在文件中查找关键词（输出关键词所在的行）：`grep <key_word> <file>`，同时 `key_word` 可以是正则表达式
15. 修改 dns：`sudo vim /etc/hosts`；刷新 `sudo /etc/init.d/network-manager restart`
16. 重启网络：`service networking restart`
17. 解决 gitee 无法使用 ssh 连接：
	1. 设置 网卡的 mtu 为 1200（一般默认都是 1500）：`sudo ifconfig enp5s0 mtu 1200`
	2. 重启网络：`service networking restart`
18. linux 审计系统查看：`sudo tail -f /var/log/audit/audit.log`
	1. 高级审计：ausearch
19. 挂载和卸载：
	1. 挂载：`mount <分区> <路径>`
	2. 卸载：`unmount <分区或路径>`
20. nvidia 驱动 runfile 卸载和安装：
	1. 卸载：`cd /usr/local/cuda-xx.x/bin`，然后 `sudo ./cuda-uninstaller`
	2. 安装：下载对应的 runfile 文件（https://developer.nvidia.com/cuda-12-2-0-download-archive），`sudo sh cuda_xxxx_linux.run`
21. vscode 代理下，服务器登录和文件传输（以 4001 为例）：
	1. ssh 登录：`ssh -o "ProxyCommand=nc -X connect -x 127.0.0.1:1081 %h %p" guoyinlin@10.13.71.37 -p 4001`
	2. scp 传输：`scp -o "ProxyCommand=nc -X connect -x 127.0.0.1:1081 %h %p" -P 4001 /path/to/local/file guoyinlin@10.13.71.37:/path/to/remote/directory`
22. 深度查看 cpu 占用情况：`sudo sysdig -c topprocs_cpu`
23. gpu 驱动、cuda 版本相关问题：
	1. 关于 gpu driver 和 cuda 版本，可以查看 nvidia 官方手册的表 3：https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
	2. 实际上，nvidia-smi 中显示的 12.4 之类的是所谓的 driver API，而我们平时跑代码用的是所谓的 runtime API（也包括 nvcc 这个命令得到的 cuda 驱动版本），且一般来说，只要 runtime API 的版本小于 driver API 就是正常的（因为 driver API 是向下兼容）
	3. 对于 pytorch，安装 pytorch 某个版本时，python 会自动安装对应的 runtime API 的 cuda 版本，所以即使 nvcc 这个命令显示没有也不影响跑 cuda + pytorch 的代码；但是需要注意的是，pytorch 的版本一定要和 cuda 版本对应，可以在这里查看：https://pytorch.org/get-started/previous-versions/
	4. 存在一种比较极端的情况，如果 runtime API 的版本太小了（远远小于 driver API，比如一个 10.2 一个 12.4），这时跑 python 代码的时候可能报错，可以通过安装更高版本的 cuda 和对应的 pytorch 来解决（一般一个 pytorch 也会兼容不同的 cuda 版本）
24. 清理黄色缓存：`sync; sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"`
25. 批量查找相关进程并 kill 掉：`pgrep -f 'a.py' | xargs kill`
26. 查看端口占用情况：`sof -i:<端口号>`
27. MD5 校验：`md5sum <文件名>`
28. linux 中的文件下载：
	1. 使用 wget，一些常用的配置为：
        1. 重试次数 `-t n`， `n=0` 表示无限次重试
        2. 指定文件名 `-O name`，`name` 表示文件名
        3. 断点续传 `-c`
        4. 多文件下载 `-i file`，`file` 表示包含下载链接的文件
    2. 使用 curl，一些常用的配置为：
        1. 指定文件名 `-o name`，`name` 表示文件名
        2. 断点续传 `-C -`，`-` 表示从上次下载的地方继续下载
    3. curl 和 wget 的区别：
        1. wget 是一个独立的下载程序，无需额外的资源库
        2. curl是一个多功能工具，是libcurl这个库支持的。它可以下载网络内容，但同时它也能做更多别的事情。


### bash 中的测试命令

```bash
test condition
```
因为 bash 中的 if 不会判断条件是否成立，而是执行条件对应的命令，如果条件执行成功，则 if 成立。

而 test 命令可以将条件转成真假，此时就和常见的编程语言中的 if 差不多了。

test 中的
数值比较：![[Pasted image 20230417114957.png]]
字符串比较：![[Pasted image 20230417115046.png]]
文件比较：![[Pasted image 20230417115755.png]]
> 注意，判读字符串变量的时候，要加 "${val}" 双引号！！ 

此外，还可以使用 ```[ condition ]``` 来实现（注意方括号之间一定要有空格）。

### linux 种 # profile、bashrc、bash_profile 的区别

+ profile：位于 **/etc/profile**，用于设置系统级的环境变量和启动程序，在这个文件下配置会对**所有用户**生效，当用户登录（login）时，文件会被执行，并从 `/etc/profile.d` 目录的配置文件中查找shell设置。
+ bashrc：用于配置函数或别名：
	+ 系统级的位于 `/etc/bashrc`，对所有用户生效。
	+ 用户级的位于 `~/.bashrc`，仅对当前用户生效
	+ bashrc 文件只会对指定的 shell 类型起作用，bashrc 只会被 bash shell 调用
+ bash_profile：只对单一用户有效，文件位于`~/.bash_profile`，该文件是一个用户级的设置，可以理解为某一个用户的 profile 目录下，这个文件同样也可以用于配置环境变量和启动程序，但只针对单个用户有效。和 profile 文件类似，bash_profile 也会在用户登录（login）时生效，也可以用于设置环境变理。但与 profile 不同，bash_profile 只会对当前用户生效。

这三种文件类型的差异用一句话表述就是：

`/etc/profile`，`/etc/bashrc` 是系统全局环境变量设定；

`~/.profile`，`~/.bashrc` 用户目录下的私有环境变量设定。
三个文件的执行情况如下：
![](v2-ea0eb026fe5e9c7a9520a930f34e5125_720w.webp)

> 在 vscode 中，➕ 号创建的 shell 为交互式非登录 shell，此时只会执行 ~/.bashrc 中的代码。


### 服务器基本信息

| 校区  |       ip       |  端口   |                    显卡                     |
| :-: | :------------: | :---: | :---------------------------------------: |
| 玉泉  |  10.13.71,37   | 4001  |            GeForce RTX 3090*4             |
| 玉泉  |  10.13.71,37   | 3001  |            GeForce GTX 1080*3             |
| 玉泉  |  10.13.71,37   | 3002  | GeForce GTX 1080*1, GeForce GTX 1080 Ti*2 |
| 玉泉  |  10.13.71,37   | 2001  |            GeForce GTX 1080*2             |
| 工院  | 10.115.107.217 |  22   |               Tesla P100*4                |
| 工院  | 10.115.107.218 | 21822 |               Tesla P100*8                |
| 工院  | 10.115.107.218 | 4002  |           GeForce RTX 4090 D*4            |
| 工院  | 10.115.107.218 | 4003  |           GeForce RTX 4090 D*4            |

服务器硬盘和目录映射：
217 服务器：
home5 : /dev/sdb1
home2 : /dev/sdc1
218 服务器：
/dev/sdg1 : /home6
/dev/sdc1 : /home3

### 其他杂项

1. 服务器中病毒（挖矿）参考解决方案：https://www.cc98.org/topic/5812515

2. 关于 linux 源：
	+ 系统的软件源位于 /etc/apt/sources.list 的文件中，所以如果需要新加系统软件源可以直接 vim 编辑这个文件
	+ 其他自己加的软件源位于 /etc/apt/sources.list.d/ 的文件夹下

3. 如何在 Linux 终端中使用临时的系统代理：
	- export http_proxy=http://127.0.0.1:20171
	- export https_proxy=http://127.0.0.1:7890

4. 安装 python 包 的几种方法及其区别：
	1. 从 github 安装：
		1. 先下载源文件，git clone 地址，然后cd进入到目录下：
			1. python setup.py install 使用 Python 的 `setuptools` 模块来安装包，执行该目录下的 `setup.py` 脚本，根据 `setup.py` 中的配置信息，将包安装到 Python 解释器的默认路径中，使得你可以在任何地方导入该包的模块
			2. pip install .：安装后的模块freeze在pip/conda依赖下，换句话说，再修改本地的原项目文件，不会导致对应模块发生变化
			3. pip install -e .：-e 理解为 editable，修改本地文件，调用的模块以最新文件为准
		2. 不下载源文件，pip install git+地址
	2. 安装 wheel 文件：pip install xxx.whl
	3. 直接 pip install + 包名字：pip install xxx

### 4002 & 4003 服务器配置记录

一些常用的命令：
1. 删除用户所有的信息（包括文件、记录等）： userdel -r user
2. 查看所有用户：cat /etc/passwd
3. 查看所有用户组：cat /ect/group
4. 修改 root 权限：建议添加至 sudo 用户组（不是 root 用户组），避免直接修改 sudoers 文件（如果要修改，用 visudo）
5. 查看当前用户所在用户组（一个用户可以在多个用户组中）：groups
6. 查看 gpu 型号：lspci | grep -i nvidia，然后在 https://admin.pci-ids.ucw.cz/read/PC/10de 这里搜索（或者 nvidia-smi -L）
7. 查看 cpu 情况：lscpu

已做的修改：
1. 更改 sudoers 文件中的 %sudo 用户组的 NOPASSWD 
2. 设置了 root 的密码：**wyh210518** 
3. 关闭自动更新（用于解决 Failed to initialize NVML: Driver/library version mismatch 问题，直接重启其实就可以解决这个问题，但是这个还是要关闭）：
	1. sudo vim /etc/apt/apt.conf.d/10periodic
	2. sudo vim /etc/apt/apt.conf.d/20auto-upgrades
4. 关闭自动休眠：sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
5. 开启持久模式：sudo nvidia-smi  -pm 1
6. 映射机械硬盘到 /mnt/sda1 目录

4002 & 4003 参数配置：
+ CPU：Intel(R) Xeon(R) Platinum 8352V CPU @ 2.10GHz，双 CPU、每个 CPU 36 核心，每个核 双线程
+ GPU：GeForce RTX 4090 D
+ 硬盘：三星 SSD 990 PRO 4TB + 西数 WUS721010ALE6L4 10TB
+ 内存：16 卡槽，但是只用了 4 x Samsung 3200MT/s 64G DDR4 

