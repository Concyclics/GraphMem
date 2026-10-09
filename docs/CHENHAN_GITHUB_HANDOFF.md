# chenhan 提交与推送指引

## 当前状态（2026-10-09）

- 仓库：`/ssd3/chenhan/Spark_MemGraph_Dev/GraphMem`
- GitHub：`https://github.com/Concyclics/GraphMem.git`
- 当前本地分支：`v5-8-cleanup`；本次目标远端分支：`main`。
- 实现已经提交：`44cffa2`，包含截至 V5.81 的代码、配置、测试、实验脚本和 README。不要重复提交这些实现。
- 本次完整单元测试：626 项通过；没有重新运行全量 benchmark。
- 尚未完成 GitHub 推送：`sen` 没有 GitHub 登录凭据，由 `chenhan` 使用自己的 GitHub 凭据推送。
- 本指引作为新文件留在工作区，由 `chenhan` 检查、提交后一起推送。

## 本地权限处理

已检查 1,298 个受版本控制文件、相关目录及 Git 元数据路径，补齐 122 个 `sen` 所有路径的 `chenhan` ACL 读写权限；检查时没有剩余访问或默认 ACL 缺项。Git 目录及代码目录已有 `sen` / `chenhan` 默认 ACL，供新文件继承。

仓库所有权不变，没有使用 `chmod 777`、没有开放其他用户权限，也没有修改任何 GitHub 凭据。仓库本地配置设为 `core.filemode=false`，忽略共享目录产生的执行位噪声，不改变文件内容或已提交的执行位。

以上是按 ACL 做的权限检查；当前会话不能切换为 `chenhan`，最终以该账户运行下面的命令为准。文件系统权限与 GitHub 写权限是两件事。

## 1. 使用 chenhan 登录服务器并检查

```bash
whoami
cd /ssd3/chenhan/Spark_MemGraph_Dev/GraphMem
git status --short
git log -1 --oneline
test -r .git/index && test -w .git/index && test -w .git && echo 'Git index permissions OK'
```

`whoami` 应为 `chenhan`；实现提交应包含 `44cffa2`。交接时只有本指引尚未提交。如果看到其他内容修改，先检查，不要执行 `git add .`。

## 2. 提交本指引

确认暂存区没有其他人的修改后执行：

```bash
git diff --cached --stat
git add docs/CHENHAN_GITHUB_HANDOFF.md
git diff --cached --stat
git -c user.name='Chen Han' -c user.email='chenhan@u.nus.edu' commit -m 'Document chenhan GitHub handoff and shared permissions'
```

如果本指引已提交，跳过此步。上述身份仅用于这次文档提交，不覆盖账户全局配置，不重写已有实现提交。

## 3. 检查 GitHub 登录

如果账户已有可用的 Git 凭据，可以直接执行下一步。若推送要求认证，使用当前 `chenhan` 账户登录：

```bash
gh auth status
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
```

GitHub 账户必须有 `Concyclics/GraphMem` 的写权限。不要把 Token 写入仓库、文档、远端 URL 或聊天消息。

## 4. 安全推送到 main

```bash
git fetch origin main
if git merge-base --is-ancestor FETCH_HEAD HEAD; then
  git log --oneline FETCH_HEAD..HEAD
  git push origin HEAD:main
else
  echo '远端 main 有本地未包含的提交：停止推送，先合并并验证。不要强制推送。'
fi
```

当前本地分支名称不影响 `HEAD:main` 的目标。不要使用裸 `git push`：当前分支的旧 upstream 是 `origin/v5-8-cleanup`，不是 `main`。也不需要切换到可能落后的本地 `main`。

## 5. 核验

```bash
git rev-parse HEAD
git ls-remote --heads origin main
git status --short
```

前两条命令的提交哈希应一致；没有新修改时，第三条应无输出。

如果仍出现 `Permission denied`，记录报错的具体路径并执行 `getfacl -p <路径>`，不要清空 `.git`、强制重置工作区或删除不确定用途的锁文件。如果报 `403` 或认证失败，需要处理 GitHub 账户的仓库写权限，本地 ACL 无法解决。
