
官方资源：https://git-scm.com/book/zh/v2

查看远程仓库地址：
```bash
git remote -v
```

查看配置
```bash
git config --list
```

配置用户名和邮箱：
```bash
git config --global user.name "Name"
git config --global user.email "Email"
```

> 这里的 "" 号内填写用户名和邮箱，可以是任意的，不一定是真实信息。


一个本地仓库可以对应多个远程，添加新的远程仓库：
```bash
git remote add <name> <url>
```
> name 是远程仓库的名字（可以随便指定，不一定是 orig 或者 master），url 是远程仓库的地址。

向指定的远程仓库推送：
```bash
git push <name>
```
> name 是远程仓库的名字，和上面的 name 对应。


拉取远程仓库的代码：
```bash
git fetch <远程主机名> <分支名>
```
> 这个命令只是将远程仓库的代码拉取到本地，不会合并到本地仓库，需要手动合并。

合并远程仓库的代码：
```bash
git merge <远程主机名>/<分支名>
```
> 这个命令会将远程仓库的代码合并到本地仓库。
> 注意：对于非默认远程分支，需要手动指定分支名。

使用 `git pull` 命令可以一次性拉取远程仓库的代码并合并到本地仓库：
```bash
git pull <远程主机名> <分支名>
```
> 注意：对于非默认远程分支，需要手动指定分支名。

git pull = git fetch + git merge

善用 stash。

少用 rebase。
