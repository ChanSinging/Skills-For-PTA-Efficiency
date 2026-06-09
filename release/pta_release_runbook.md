# PTA 版本发布 Runbook

Claude Code 可执行指令文档，用于 PTA（PyTorch Ascend Adapter）版本发布。

---

## CC 角色与执行规则

你正在为 PTA 仓库执行版本发布。先通读本文档全文，然后严格按照步骤顺序执行。

**执行规则：**

- **版本号优先。** 每步首先修改 `requirements.txt` 和 `test/requirements.txt` 的版本号和下载链接，再做其他变更。
- **检查上游分布式测试。** 版本号修改后，对比社区 PyTorch release tag 的 `test/distributed/` 目录与 PTA 仓库，报告缺失的测试文件。
- **CI 驱动修复循环。** 代码推送到远程后，用户触发 CI 流水线。CC 读取用户粘贴的 CI 报错日志，分析根因，本地修改代码，squash 到同一 commit，force-push 覆盖远程。重复此循环直至 CI 通过。
- **Squash 提交策略。** CC 只做 `git add -A` 并生成 commit message，**不执行 git commit 和 git push**——由用户手动完成。远程分支只保留一次提交记录（初始 `git commit`，后续修复 `git commit --amend` + `git push --force-with-lease`）。
- 每步开始时使用 AskUserQuestion 收集版本参数。
- 修改文件前必须先 Read 该文件以探测当前状态。
- 每步完成后输出手动 checklist。
- **版本号存在性验证。** 每次修改版本号或 extra-index-url 后，使用 curl 探测对应 whl 文件是否存在（HTTP 200/404）。版本不存在时主动询问用户：版本号是否正确、是否等待构建、是否跳过。不要只看 index 页面可访问——404 表示版本还未构建。
- 本地 test-and-fix loop 作为 CI 不可用时的 fallback（见附录 E）。

---

## Step 0：环境预检

### 0.1 Git 状态

执行以下命令。如存在未提交变更且用户未确认，不得继续。

```bash
git status
git branch --show-current
```

**预期结果：** 工作区干净，分支 = `master`。如不符，询问用户确认后再继续。

### 0.2 NPU 设备检查

```bash
python -c "import torch; import torch_npu; print('torch:', torch.__version__); print('NPU count:', torch_npu.npu.device_count())"
```

若失败，报告错误并暂停。

### 0.3 上游远程连通性检查

验证能访问社区 PyTorch 仓库（用于后续对比上游测试用例）：

```bash
git ls-remote --tags https://github.com/pytorch/pytorch.git 2>&1 | head -5
```

若失败，报告错误（后续上游测试对比将跳过），但不阻塞流程。

### 0.4 当前版本探测

读取以下文件并提取当前版本号：

- `Read requirements.txt` → 提取以 `torch==` 开头的行，记为 `$CURRENT_TORCH`
- `Read test/requirements.txt` → 提取以 `torchvision==` 开头的行，记为 `$CURRENT_TORCHVISION`
- `Read version.txt` → 记为 `$CURRENT_VERSION_TXT`
- `Read ci/build.sh` → 提取 `SUPPORTED_TORCH_VERSION` 数组的值

向用户输出当前版本汇总表后再继续。

### 0.5 目标文件存在性检查

验证以下全部文件存在且可读（使用 Bash `ls`）：

```
requirements.txt
test/requirements.txt
version.txt
ci/build.sh
ci/docker/X86/Dockerfile
ci/docker/ARM/Dockerfile
ci/docker/README.md
setup.py
.lintrunner.toml
pyproject.toml
README.md
README.zh.md
SECURITYNOTE.md
```

报告缺失文件，如有缺失则暂停。

---

## Step 1：bump-dev — 升级 Dev Nightly 版本号

**时机：** 社区 PyTorch 发布新 Nightly 构建（更新了 dev 日期）。
**目标分支：** master

### 1.1 收集参数

使用 AskUserQuestion 询问：

> **问题：** 新的 torch dev nightly 版本号是多少？
> **示例：** 2.13.0.dev20260601
> **标题：** Torch Dev 版本

保存为 `$NEW_TORCH`。

再次使用 AskUserQuestion 询问：

> **问题：** 本次工作的分支名是什么？
> **示例：** bump-dev-20260601
> **标题：** 分支名

保存为 `$BRANCH`。

### 1.2 版本号修改（版本优先）

**探测当前状态：**

```bash
grep 'torch==' requirements.txt
grep 'torchvision==' test/requirements.txt
```

提取当前版本字符串，分别记为 `$OLD_TORCH` 和 `$OLD_TORCHVISION`。

**幂等性检查：** 如果 `$OLD_TORCH` 已经等于 `$NEW_TORCH`，输出 `⏭️ 跳过：版本已为目标值`，跳至 1.4。

**Edit `requirements.txt`：**

将 `torch==$OLD_TORCH` 替换为 `torch==$NEW_TORCH`。

```
old_string: torch==$OLD_TORCH
new_string: torch==$NEW_TORCH
```

**Edit `test/requirements.txt`：**

从 `$NEW_TORCH` 中提取 dev 日期（`.dev` 后面的部分，如 `20260601`）。
构造新 torchvision 版本：取 `$OLD_TORCHVISION` 的基础版本（`.dev` 之前的部分，如 `0.26.0`），拼接 `.dev<日期>`。

**重要：torchvision 与 torch 版本规则：**
- dev 日期必须与 torch 一致（如 torch 是 `dev20260609`，torchvision 也必须是 `dev20260609`）
- torchvision 的主版本号（如 `0.26.0`、`0.27.0`）保持不变，**只更新日期后缀**
- 除非 Step 2（switch-rc）社区要求升级 torchvision 大版本，否则永远不动主版本号

```
old_string: torchvision==$OLD_TORCHVISION
new_string: torchvision==<基础版本>.dev<日期>
```

### 1.3 验证版本变更

```bash
grep 'torch==' requirements.txt
grep 'torchvision==' test/requirements.txt
```

确认全部输出新版本号。

### 1.4 验证版本号存在性

**必须确认新版本号在 extra-index-url 中真实存在，否则 CI 会失败。**

```bash
# 验证 torch 版本是否存在于 nightly 频道
curl -I -L "https://download.pytorch.org/whl/nightly/cpu/torch-${NEW_TORCH//.dev/-dev}%2Bcpu-cp311-cp311-linux_x86_64.whl" 2>&1 | grep -E "HTTP/[0-9.]|404|Location"

# 验证 torchvision 版本是否存在于 nightly 频道
TORCHVISION_DEV=$(grep 'torchvision==' test/requirements.txt | grep -oP '==\K.*')
curl -I -L "https://download.pytorch.org/whl/nightly/cpu/torchvision-${TORCHVISION_DEV//.dev/-dev}%2Bcpu-cp311-cp311-linux_x86_64.whl" 2>&1 | grep -E "HTTP/[0-9.]|404|Location"

# 也可以直接用 WebFetch 访问目录页，搜索版本号是否存在
# WebFetch https://download.pytorch.org/whl/nightly/cpu/torch/  搜索 "$NEW_TORCH"
# WebFetch https://download.pytorch.org/whl/nightly/cpu/torchvision/  搜索 "torchvision 版本号"
```

若返回 HTTP 200/302 → ✅ 版本存在，继续。

**若返回 HTTP 404 或连接失败：**
- 输出 `⚠️ 版本 <版本号> 在 nightly 频道中不存在（HTTP 404）`
- 使用 AskUserQuestion 询问：
  > **问题：** `<版本号>` 在 `nightly` 频道中未找到。可能还未构建完成（UTC 12:00 左右完成）。如何处理？
  > **标题：** 版本不存在
  > **选项：** "版本号写错了，我要改" / "等待镜像构建，稍后重试" / "跳过验证，继续执行"

### 1.5 检查上游新增分布式测试

对比社区 PyTorch 对应 tag 和 PTA 仓库的 `test/distributed/` 目录，列出 PTA 缺失的测试文件。

```bash
# 获取社区最新 release tag
git ls-remote --tags https://github.com/pytorch/pytorch.git | grep -E 'v[0-9]+\.[0-9]+\.[0-9]+$' | tail -5

# 列出社区 tag 的 test/distributed 文件（选择与新 torc 版本匹配的 tag）
# 记社区 tag 为 $UPSTREAM_TAG
git ls-tree --name-only -r $UPSTREAM_TAG:test/distributed/ 2>/dev/null | grep '\.py$' | sort > /tmp/upstream_dist_tests.txt

# 列出 PTA 当前 test/distributed 文件
find test/distributed/ -name '*.py' | sort > /tmp/pta_dist_tests.txt

# 对比差异
comm -23 /tmp/upstream_dist_tests.txt /tmp/pta_dist_tests.txt
```

**如果 comm 有输出：** 列出缺失文件，使用 AskUserQuestion 询问用户是否需要移植：
> **问题：** 社区上游新增了以下分布式测试文件（PTA 缺失），是否需要移植？
> **标题：** 新增测试
> **选项：** "全部移植" / "跳过" / "手动选择"

**如果无输出：** `✅ 无新增上游分布式测试，继续。`

### 1.6 创建分支、提交、推送触发 CI

```bash
git checkout -b $BRANCH master
git add -A
```

输出：

```
✅ 文件已暂存到 $BRANCH 分支。请手动提交并推送：

   git commit -m "bump-dev: torch $OLD_TORCH → $NEW_TORCH"
   git push -u origin $BRANCH

   推送后在 CI 系统触发流水线。CI 完成后将报错日志粘贴至此处，如直接通过回复"通过"。
```

### 1.7 CI 驱动修复循环

**当用户粘贴 CI 报错日志时：**

1. **分析报错。** 读取用户粘贴的日志，定位失败文件和错误类型。
2. **读取相关源文件。** 根据附录 B 的测试-源文件映射找到对应代码。
3. **修改代码。** 分析根因并 Edit 源文件（常见 bug 模式见附录 C）。
4. **暂存修复并生成 squash 命令：**

   ```bash
   git add -A
   ```

   输出给用户：

   ```
   🔄 已修复，文件已暂存。请手动 squash 并推送：
      git commit --amend --no-edit
      git push --force-with-lease origin $BRANCH --no-verify
      推送后请重跑 CI 流水线，将结果粘贴到此处。
   ```

5. **重复 1-4** 直到用户回复 "通过" 或 CI 日志显示全绿。

**重要规则：**
- CC 只做 `git add -A`，不执行 commit/push。
- 远程始终只保留一条 commit（用户用 `commit --amend` squash）。
- 如果用户反馈 "CI 不可用"，回退到本地 test-and-fix（附录 E）。

### 1.8 输出 Checklist

CI 通过后输出：

```
✅ Step 1 完成：bump-dev

手动 Checklist：
- [ ] 验证新 Nightly whl 可从 https://download.pytorch.org/whl/nightly/cpu 下载
- [ ] 确认同日期的 torchvision Nightly 存在
- [ ] 远程分支 $BRANCH 已 squash 为单 commit
- [ ] CI 流水线全部通过

下一步：社区发布 RC 后执行 Step 2。

历史参考 PR：Ascend/pytorch!34141
  requirements.txt: torch==2.12.0.dev20260217 → 2.12.0.dev20260317
```

---

## Step 2：switch-rc — 切换 RC 正式版本 + NPU 适配修复

**时机：** 社区 PyTorch 发布 RC（Release Candidate）。
**目标分支：** master
**⚠️ 这是最复杂的一步。预期会有测试失败，需要多轮 CI 修复。**

### 2.1 收集参数

使用 AskUserQuestion 询问：

> **问题：** RC 各组件版本是多少？请提供所有适用项。
> **标题：** RC 版本（示例：2.13.0）
> 选项：
> - torch: 2.13.0
> - torchvision: 0.28.0
> - onnx（如需变更）: 1.22.0
> - pyg torch 版本（-f URL 用）: 2.13.0
> - torch-scatter（如需变更）: 2.1.2

保存为 `$RC_TORCH`、`$RC_TORCHVISION`、`$RC_ONNX`（可选）、`$RC_PYG_TORCH`（可选）、`$RC_TORCH_SCATTER`（可选）。

再次使用 AskUserQuestion 询问：

> **问题：** 本次工作的分支名是什么？
> **标题：** 分支名（示例：switch-rc-2.13.0）

保存为 `$BRANCH`。

### 2.2 版本号修改（版本优先）

**探测当前状态：**

```bash
grep -n 'extra-index-url\|torch==\|torchvision==\|onnx==\|torch-scatter==\|data.pyg.org' requirements.txt test/requirements.txt
```

保存全部旧值以备参考。

**Edit `requirements.txt`：**

1. 替换 extra-index-url：
   ```
   old_string: https://download.pytorch.org/whl/nightly/cpu
   new_string: https://download.pytorch.org/whl/test/cpu
   ```
2. 替换 torch 版本：
   ```
   old_string: torch==$OLD_TORCH
   new_string: torch==$RC_TORCH
   ```

**Edit `test/requirements.txt`：**

1. 替换 extra-index-url（同上：nightly → test）
2. 替换 torchvision 版本：
   ```
   old_string: torchvision==$OLD_TORCHVISION
   new_string: torchvision==$RC_TORCHVISION
   ```
3. 如果提供了 `$RC_ONNX`，替换 onnx 版本。
4. 如果提供了 `$RC_PYG_TORCH`，更新 `-f` URL：
   ```
   old_string: https://data.pyg.org/whl/torch-<当前值>+cpu.html
   new_string: https://data.pyg.org/whl/torch-$RC_PYG_TORCH+cpu.html
   ```
5. 如果提供了 `$RC_TORCH_SCATTER`，替换 torch-scatter 版本。

### 2.3 验证版本变更

```bash
grep -n 'extra-index-url\|torch==' requirements.txt
grep -n 'extra-index-url\|torchvision==\|onnx==\|data.pyg.org' test/requirements.txt
```

确认所有值与 RC 参数匹配。

### 2.4 验证版本号存在性

Step 2 修改了 URL（nightly→test）和版本号（dev→RC），**必须逐一确认 RC 版本在 test 频道中存在**：

```bash
# 验证 test 频道可访问 + torch RC 版本存在
curl -I -L "https://download.pytorch.org/whl/test/cpu/torch-${RC_TORCH}%2Bcpu-cp311-cp311-linux_x86_64.whl" 2>&1 | grep -E "HTTP|404|Location"

# 验证 torchvision RC 版本存在
curl -I -L "https://download.pytorch.org/whl/test/cpu/torchvision-${RC_TORCHVISION}%2Bcpu-cp311-cp311-linux_x86_64.whl" 2>&1 | grep -E "HTTP|404|Location"

# 如果修改了 pyg URL，验证之
curl -I -L "https://data.pyg.org/whl/torch-$RC_PYG_TORCH+cpu.html" 2>&1 | head -10
```

若全部返回 HTTP 200/302 → ✅ 版本存在，继续。

**若任一返回 HTTP 404：**
- 输出 `⚠️ <包名>==<版本> 在对应频道中不存在`
- 使用 AskUserQuestion 询问：
  > **问题：** `<版本号>` 未找到。可能 RC 还未发布或版本号写错。如何处理？
  > **标题：** 版本不存在
  > **选项：** "版本号写错了，我要改" / "等待发布，稍后重试" / "跳过验证，继续执行"

### 2.5 检查上游新增分布式测试

执行与 Step 1.5 相同的上游对比流程：
- 获取社区 RC tag（如 `v$RC_TORCH`）的 `test/distributed/` 文件列表
- 与 PTA 本地 `test/distributed/` 对比
- 报告缺失文件并询问用户是否移植

### 2.6 创建分支、提交、推送触发 CI

```bash
git checkout -b $BRANCH master
git add -A
```

输出：

```
✅ 文件已暂存到 $BRANCH 分支。请手动提交并推送：

   git commit -m "switch-rc: torch $OLD_TORCH → $RC_TORCH"
   git push -u origin $BRANCH

   推送后在 CI 系统触发流水线。CI 完成后将报错日志粘贴至此处，如直接通过回复"通过"。
```

### 2.7 CI 驱动修复循环

**当用户粘贴 CI 报错日志时：**

1. **分析报错。** 定位失败文件和错误类型。参考附录 B 找到对应源文件。
2. **读取源文件。** Read 相关源文件。
3. **分析根因。** 常见模式：
   - **TypeError: __init__() got unexpected keyword argument** → PyTorch API 参数名变更
   - **ImportError: cannot import name 'X'** → PyTorch 移动/重命名了模块
   - **AttributeError: module has no attribute 'X'** → PyTorch 重命名或移除了属性
   - **public_bindings AssertionError** → 更新 `test/allowlist_for_publicAPI.json`
4. **修改代码。** Edit 源文件进行修复。
5. **暂存修复并生成 squash 命令：**

   ```bash
   git add -A
   ```

   输出给用户：

   ```
   🔄 已修复，文件已暂存。请手动 squash 并推送：
      git commit --amend --no-edit
      git push --force-with-lease origin $BRANCH --no-verify
      推送后请重跑 CI 流水线，将结果粘贴到此处。
   ```

6. **重复 1-5** 直到用户回复 "通过" 或 CI 日志显示全绿。

**如果 CI 不可用：** 回退到本地 test-and-fix（附录 E），逐个运行以下测试：
- `test/npu/__dynamo.py` → 源文件 `torch_npu/utils/_dynamo.py`
- `test/npu/test_public_bindings.py` → 源文件 `torch_npu/*`, `test/allowlist_for_publicAPI.json`
- `test/npu/test_compatibility.py` → 源文件 `torch_npu/distributed/*`
- `test/distributed/test_fault_mode.py` → 源文件 `torch_npu/distributed/fsdp/*`

### 2.8 输出 Checklist

CI 通过后输出：

```
✅ Step 2 完成：switch-rc

手动 Checklist：
- [ ] 远程分支 $BRANCH 已 squash 为单 commit
- [ ] CI 流水线全部通过
- [ ] 如新增了测试文件，.lintrunner.toml 和 pyproject.toml 排除规则已更新
- [ ] 验证 torchvision 版本兼容性

历史参考 PR：Ascend/pytorch!35103
  requirements: nightly→test, torch 2.12.0.dev→2.12.0
  _dynamo.py 修复, public_bindings 白名单, lintrunner 排除规则
```

---

## Step 3：bump-next — 开启下一个 Dev 周期

**时机：** Step 2 RC 合入 master 后，社区开始下一版本 Nightly。
**目标分支：** master

### 3.1 收集参数

使用 AskUserQuestion 询问：

> **问题：** 下一版本号和 dev 日期是什么？
> **标题：** 下一 Dev 版本（示例：2.14.0 / 20260701）

保存为 `$NEXT_VERSION` 和 `$DEV_DATE`。

再次使用 AskUserQuestion 询问分支名，保存为 `$BRANCH`。

### 3.2 版本号修改（版本优先）

**探测当前状态：**

```bash
cat version.txt
grep 'torch==' requirements.txt
grep 'extra-index-url' requirements.txt
grep 'torchvision==' test/requirements.txt
```

保存旧值 `$CURRENT_VERSION_TXT`、`$OLD_TORCH`、`$OLD_TORCHVISION`。

**幂等性检查：** 如果 `version.txt` 已等于 `$NEXT_VERSION`，跳过。

**Edit `version.txt`：**

```
old_string: $CURRENT_VERSION_TXT
new_string: $NEXT_VERSION
```

**Edit `requirements.txt`：**

1. 切换 extra-index-url 回 nightly：
   ```
   old_string: https://download.pytorch.org/whl/test/cpu
   new_string: https://download.pytorch.org/whl/nightly/cpu
   ```
2. 切换 torch 为 dev 版本：
   ```
   old_string: torch==$OLD_TORCH
   new_string: torch==$NEXT_VERSION.dev$DEV_DATE
   ```

**Edit `test/requirements.txt`：**

1. 切换 extra-index-url 回 nightly（test → nightly）
2. 切换 torchvision 为 dev 版本：
   ```
   old_string: torchvision==$OLD_TORCHVISION
   new_string: torchvision==<基础版本>.dev$DEV_DATE
   ```
   （从 `$OLD_TORCHVISION` 提取基础版本号，如 `0.28.0`）
   **规则同 Step 1.2：日期与 torch 一致（`dev$DEV_DATE`），主版本号不变。**

### 3.3 验证版本变更

```bash
cat version.txt
grep 'torch==' requirements.txt
grep 'extra-index-url' requirements.txt
grep 'torchvision==' test/requirements.txt
```

### 3.4 验证版本号存在性

Step 3 将 URL 切回 nightly，且使用了新的 dev 版本号，**必须确认新版本存在于 nightly 频道**：

```bash
# 验证 torch dev 版本存在于 nightly 频道
NEXT_TORCH_DEV="$NEXT_VERSION.dev$DEV_DATE"
curl -I -L "https://download.pytorch.org/whl/nightly/cpu/torch-${NEXT_TORCH_DEV//.dev/-dev}%2Bcpu-cp311-cp311-linux_x86_64.whl" 2>&1 | grep -E "HTTP|404|Location"

# 验证 torchvision dev 版本存在
NEXT_TV_DEV="<基础版本>.dev$DEV_DATE"
curl -I -L "https://download.pytorch.org/whl/nightly/cpu/torchvision-${NEXT_TV_DEV//.dev/-dev}%2Bcpu-cp311-cp311-linux_x86_64.whl" 2>&1 | grep -E "HTTP|404|Location"
```

**若返回 HTTP 404：** 与 Step 1.4 相同处理——询问用户版本号是否正确、等待构建还是跳过。

### 3.5 检查上游新增分布式测试

执行与 Step 1.5 相同的上游对比流程（使用最新社区 dev tag）。

### 3.6 创建分支、提交、推送触发 CI

```bash
git checkout -b $BRANCH master
git add -A
```

输出：

```
✅ 文件已暂存到 $BRANCH 分支。请手动提交并推送：

   git commit -m "bump-next: version $CURRENT_VERSION_TXT → $NEXT_VERSION.dev$DEV_DATE"
   git push -u origin $BRANCH

   推送后在 CI 系统触发流水线。CI 完成后将报错日志粘贴至此处，如直接通过回复"通过"。
```

### 3.7 CI 驱动修复循环

与 Step 1.7 相同的流程：分析 CI 报错 → 修改源文件 → `git add -A` → 生成 `commit --amend` + `push --force-with-lease` 命令给用户，循环至通过。

### 3.8 输出 Checklist

```
✅ Step 3 完成：bump-next

手动 Checklist：
- [ ] 远程分支 $BRANCH 已 squash 为单 commit
- [ ] CI 流水线全部通过
- [ ] 为新的 PTA 分支创建 op-plugin 兼容 PR（参考：Ascend/op-plugin!4923）
- [ ] 验证新 Nightly whl 可用

历史参考 PR：
  Ascend/pytorch!35159 — version.txt: 2.12.0→2.13.0, requirements→nightly
  Ascend/op-plugin!4923 — op-plugin 兼容 PTA 2.13 分支
```

---

## Step 4：cut-release — 切正式 Release 分支 & 全量配置更新

**时机：** 社区 PyTorch 发布正式版本。
**⚠️ 本步会创建新 git 分支，修改 10+ 个文件。**

### 4.1 收集参数

使用 AskUserQuestion 逐一询问：

> **Q1：** Release 版本号是多少？
> **标题：** Release 版本（示例：2.13.0）

保存为 `$REL_VERSION`。

> **Q2：** ci/docker/*/Dockerfile 的 Docker FROM 镜像是什么？
> **标题：** Docker 镜像（示例：pytorch/manylinux2_28-builder:cpu-v2.13.0-rc1）

保存为 `$DOCKER_FROM`。

> **Q3：** Python 版本（逗号分隔）？
> **标题：** Python 版本（示例：3.10,3.11,3.12）

保存为 `$PY_VERSIONS`（解析为数组）。

> **Q4：** CANN 版本（用于 SECURITYNOTE URL）？
> **标题：** CANN 版本（示例：8.5.0）

保存为 `$CANN_VERSION`。

> **Q5：** 需要加入 lintrunner 排除的新 NPU 测试文件（逗号分隔）？
> **标题：** 排除测试（示例：test/npu/test_new.py）

保存为 `$EXCLUDE_TESTS`（解析为数组，可为空）。

### 4.2 版本号修改（版本优先）

**探测当前状态：**

```bash
grep 'extra-index-url\|torch==' requirements.txt
grep 'extra-index-url\|torchvision==\|data.pyg.org' test/requirements.txt
```

保存旧值。

**Edit `requirements.txt`：**

```
old_string: https://download.pytorch.org/whl/test/cpu
new_string: https://download.pytorch.org/whl/cpu
```

**Edit `test/requirements.txt`：**

1. extra-index-url：`test/cpu` → `cpu`（与上相同）
2. pyg torch URL 更新：
   ```
   old_string: https://data.pyg.org/whl/torch-<当前值>+cpu.html
   new_string: https://data.pyg.org/whl/torch-$REL_VERSION+cpu.html
   ```

### 4.3 验证版本号存在性

Step 4 将 URL 切到正式 release 频道，**必须确认 release 版本在 release 频道中存在**：

```bash
# 验证 torch release 版本存在于正式频道
curl -I -L "https://download.pytorch.org/whl/cpu/torch-${REL_VERSION}%2Bcpu-cp311-cp311-linux_x86_64.whl" 2>&1 | grep -E "HTTP|404|Location"

# 验证 pyg URL（如果修改了）
curl -I -L "https://data.pyg.org/whl/torch-$REL_VERSION+cpu.html" 2>&1 | head -10
```

**若返回 HTTP 404：** 与 Step 2.4 相同处理——询问用户版本号是否正确、社区是否已正式发布。

### 4.4 创建 Release 分支

```bash
git checkout -b v$REL_VERSION master
```

验证：

```bash
git branch --show-current
# 预期输出：v$REL_VERSION
```

### 4.5 Dockerfile 组

**Edit `ci/docker/X86/Dockerfile`：**

1. 替换 FROM 行：
   ```
   old_string: FROM <当前镜像>
   new_string: FROM $DOCKER_FROM
   ```
2. 更新 python 软链接行，使用 `$PY_VERSIONS` 中的版本（先 Read 文件了解当前结构，然后更新每条 `ln -s` 行的 python 版本）
3. 更新 pip install torch 行——将所有 `pip3.X install torch==<旧值>` 替换为对应的 `pip3.X install torch==$REL_VERSION+cpu --index-url https://download.pytorch.org/whl/cpu`

**Edit `ci/docker/ARM/Dockerfile`：**

与 X86 相同的修改，但使用 `torch==$REL_VERSION`（ARM 不加 +cpu 后缀）。

**幂等性：** 如果 FROM 已经匹配 `$DOCKER_FROM`，跳过 Dockerfile 编辑。

### 4.6 CI Docker README

**Edit `ci/docker/README.md`：**

更新示例中的 Python 版本和构建命令。

### 4.7 构建脚本

**Edit `ci/build.sh`：**

检查 `$REL_VERSION` 是否已在 `SUPPORTED_TORCH_VERSION` 数组中：

```bash
grep "SUPPORTED_TORCH_VERSION" ci/build.sh
```

如果不存在，将 `$REL_VERSION` 追加到 `)` 之前。

**幂等性：** 如果 `$REL_VERSION` 已存在，跳过。

### 4.8 Setup.py Classifier

**Edit `setup.py`：**

对 `$PY_VERSIONS` 中的每个版本，检查 classifier `"Programming Language :: Python :: <version>"` 是否存在。若不存在，追加到 `"Programming Language :: Python :: 3 :: Only",` 之后。

### 4.9 Lintrunner & Pyproject 排除规则

**如果 `$EXCLUDE_TESTS` 非空：**

**Edit `.lintrunner.toml`：** 对每个 `exclude_patterns = [` 块，逐一将新测试文件插入（如尚不存在）。

**Edit `pyproject.toml`：** 在 `extend-exclude = [` 列表中加入新测试文件。

**幂等性：** 已存在的文件跳过。

### 4.10 README.md & README.zh.md

**Edit `README.md`：**

1. Torch 安装命令：`torch==<旧值>` → `torch==$REL_VERSION`
2. 分支引用：`-b v<旧_主_次>` → `-b v<REL_VERSION的主_次>`
3. torch_npu 安装：`torch-npu==<旧值>` → `torch-npu==$REL_VERSION`
4. 维护策略表格：在顶部新增一行
5. CANN 版本表格：更新支持的版本

**Edit `README.zh.md`：** 与 README.md 相同的替换模式。

### 4.11 SECURITYNOTE.md

**Edit `SECURITYNOTE.md`：**

1. 更新 CANN 版本 URL 引用：
   - CANN 8.5.0 → doc 路径 `850`
   - CANN 8.1.RC1 → doc 路径 `81RC1`
2. 更新 PyTorch release 分支引用
3. 更新 Pytorch 文档版本引用

### 4.12 检查上游新增分布式测试

执行与 Step 1.5 相同的上游对比流程（使用社区 release tag `v$REL_VERSION`）。

### 4.13 提交、推送触发 CI

```bash
git add -A
```

输出：

```
✅ 文件已暂存到 v$REL_VERSION 分支。请手动提交并推送：

   git commit -m "cut-release: v$REL_VERSION"
   git push -u origin v$REL_VERSION

   推送后在 CI 系统触发流水线。CI 完成后将报错日志粘贴至此处，如直接通过回复"通过"。
```

### 4.14 CI 驱动修复循环

与 Step 1.7 相同的流程：分析 CI 报错 → 修改源文件 → `git add -A` → 生成 `commit --amend` + `push --force-with-lease` 命令给用户，循环至通过。

**如果 CI 不可用：** 回退到本地 test-and-fix（附录 E），运行 `test/npu/test_resnet.py` 验证核心模型不被破坏。

### 4.15 输出 Checklist

```
✅ Step 4 完成：cut-release

分支：v$REL_VERSION

手动 Checklist：
- [ ] 远程分支 v$REL_VERSION 已 squash 为单 commit
- [ ] CI 流水线全部通过
- [ ] 验证 README 中 whl 下载链接可访问
- [ ] 检查 README whl 链接中的 sha256 hash（手动——无法自动生成）
- [ ] 确认 SECURITYNOTE.md 中所有 URL 指向正确的文档版本
- [ ] 确认 Docker FROM 镜像在 Docker Hub 存在
- [ ] 更新 op-plugin submodule 指向 release 分支
- [ ] 在 gitcode.com 创建目标为 v$REL_VERSION 分支的 PR

历史参考 PR：Ascend/pytorch!35503
  Release 2.12.0rc1：README/Docker/SECURITYNOTE/setup.py/lintrunner 全量更新
```

---

## 附录 A：历史 PR 索引

| Step | PR | 仓库 | 说明 |
|------|-----|------|------|
| 1 | [!34141](https://gitcode.com/Ascend/pytorch/pull/34141) | pytorch | bump-dev: torch 2.12.0.dev20260217→20260317 |
| 2 | [!35103](https://gitcode.com/Ascend/pytorch/pull/35103) | pytorch | switch-rc: torch 2.12.0, _dynamo 修复, lintrunner 排除 |
| 3 | [!35159](https://gitcode.com/Ascend/pytorch/pull/35159) | pytorch | bump-next: 2.12.0→2.13.0.dev20260416 |
| 3 | [!4923](https://gitcode.com/Ascend/op-plugin/pull/4923) | op-plugin | op-plugin 兼容 PTA 2.13 |
| 4 | [!35503](https://gitcode.com/Ascend/pytorch/pull/35503) | pytorch | cut-release: v2.12.0, 全量文档/docker/配置更新 |

### 关键 diff 参考

**PR 34141 — bump-dev：**
```
requirements.txt:      torch==2.12.0.dev20260217 → torch==2.12.0.dev20260317
test/requirements.txt: torchvision==0.26.0.dev20260217 → torchvision==0.26.0.dev20260317
```

**PR 35103 — switch-rc：**
```
requirements.txt:      nightly/cpu → test/cpu, torch==2.12.0.dev20260317 → 2.12.0
test/requirements.txt: nightly/cpu → test/cpu, torchvision 0.26.0.dev→0.27.0, onnx 1.20.0→1.21.0
torch_npu/utils/_dynamo.py: Config("npu_backend", default=..., type=...) — 增加了第一个参数
.lintrunner.toml: 新增 fault_mode, _dynamo, resnet, public_bindings, compatibility 测试排除
pyproject.toml: extend-exclude 新增
```

**PR 35159 — bump-next：**
```
requirements.txt:      test/cpu → nightly/cpu, torch==2.12.0 → 2.13.0.dev20260416
test/requirements.txt: test/cpu → nightly/cpu, torchvision 0.27.0 → 0.27.0.dev20260416
version.txt:           2.12.0 → 2.13.0
```

**PR 35503 — cut-release：**
```
requirements.txt: test/cpu → cpu
Dockerfiles:      FROM 镜像, python 3.10/11/12, torch 2.12.0, numpy 更新
build.sh:         在 SUPPORTED_TORCH_VERSION 中增加 2.12.0
setup.py:         增加 Python 3.12 classifier
.lintrunner.toml: 增加 30 条 exclude_patterns 条目
README.md:        大量版本/URL/链接更新（约 100 行）
README.zh.md:     版本表格更新
SECURITYNOTE.md:  CANN/doc URL 更新, release 分支引用更新
```

---

## 附录 B：测试与源文件映射表

| 测试文件 | 源文件 | 使用步骤 |
|---------|-------|---------|
| `test/npu/__dynamo.py` | `torch_npu/utils/_dynamo.py` | Step 2 |
| `test/npu/test_public_bindings.py` | `torch_npu/*`、`test/allowlist_for_publicAPI.json` | Step 2 |
| `test/npu/test_compatibility.py` | `torch_npu/distributed/*` | Step 2 |
| `test/distributed/test_fault_mode.py` | `torch_npu/distributed/fsdp/*`、`test/distributed/_fault_mode_cases/*` | Step 2 |
| `test/npu/test_resnet.py` | `torch_npu/*` | Step 4 |

---

## 附录 C：NPU 适配常见 Bug 类型

| 症状 | 典型原因 | 修复思路 |
|------|---------|---------|
| `TypeError: __init__() got unexpected keyword argument` | PyTorch API 参数重命名 | 查看社区 diff 获取正确签名，更新调用处 |
| `ImportError: cannot import name 'X'` | PyTorch 模块重组 | 在社区代码中找到新导入路径 |
| `AttributeError: module has no attribute 'X'` | PyTorch 重命名/移除属性 | 查看废弃通知，使用替代 API |
| public_bindings 中 `AssertionError` | PyTorch 新增/移除公开模块 | 更新 allowlist JSON |
| `torchvision` 兼容性错误 | torch 与 torchvision 版本不匹配 | 确认社区 RC 对应的 torchvision 版本 |
| FSDP 测试失败 | PyTorch FSDP API 变更 | 检查社区 FSDP 提交记录，适配 Ascend 补丁 |

---

## 附录 D：幂等性规则

每次 Edit 操作前，检查目标状态是否已达成：

| 编辑目标 | 检查方式 | 跳过条件 |
|---------|---------|---------|
| requirements.txt 中的 `torch==X` | grep 当前版本 | 已经等于目标值 |
| `extra-index-url` | grep 当前 URL | 已经等于目标值 |
| `version.txt` | cat version.txt | 已经等于目标值 |
| SUPPORTED_TORCH_VERSION | grep 该版本是否在数组中 | 已经存在 |
| setup.py classifier | grep classifier 行 | 已经存在 |
| .lintrunner.toml exclude | grep 测试文件是否在该块中 | 已经存在 |
| pyproject.toml exclude | grep 测试文件 | 已经存在 |
| Dockerfile FROM | grep FROM 行 | 已经匹配 |
| git 分支 | git branch --show-current | 已在目标分支 |

如果满足跳过条件，输出 `⏭️ 跳过 <文件名>：已达到目标状态`，继续下一步。

---

## 附录 E：修复循环规则

### E.1 CI 驱动修复循环（主要方案）

1. **推送后等待 CI。** 用户在 CI 系统触发流水线，将报错日志粘贴到对话中。
2. **CC 分析报错。** 从日志提取失败文件、错误类型、traceback。
3. **定位源文件。** 使用附录 B 的映射表找到对应源文件，Read 后分析。
4. **最小化修改。** 仅修改修复错误所必需的代码。
5. **暂存修改。** 每次修复后仅执行 `git add -A`，输出给用户：
   ```
   🔄 已修复。请手动 squash 并推送：
      git commit --amend --no-edit
      git push --force-with-lease origin $BRANCH --no-verify
   ```
6. **循环至通过。** 提示用户推送后重跑 CI，重复直至 CI 全绿。
7. **日志过长时。** 提示用户只粘贴关键错误段（最后 50 行 + traceback）。
8. **CC 不执行 commit/push。** 所有 `git commit` 和 `git push` 命令由用户手动执行。

### E.2 本地 Test-and-Fix Loop（CI 不可用时 fallback）

**触发条件：** 用户明确表示 CI 不可用。

1. **每个文件最多 5 轮。** 如果 5 轮后仍失败，暂停并询问用户。
2. **完整读取错误输出。** 使用 Bash 捕获 stdout 和 stderr。
3. **先定位根因再修改。** 不要猜测——从 traceback 追溯到根源。
4. **修改后立即重跑。** 每次 Edit 后立即重新运行同样的 pytest 命令。
5. **记录迭代轮次。** 每轮之后输出 `[第 N/5 轮] <测试名称>`。
6. **达到 5 轮上限时：**
   - 输出源文件的 `git diff`
   - 输出最后一轮 pytest 错误
   - 询问："已达到 5 轮上限，如何继续？" 选项：
     - "继续尝试"（重置计数器，再试 5 轮）
     - "给我看 diff，我手动修复"
     - "跳过该测试，继续执行"

---

## 附录 F：常见问题排查

| 问题 | 解决方案 |
|------|---------|
| `pip install` 找不到新 Nightly | 等待社区构建完成（约 UTC 12:00）或尝试其他日期 |
| lintrunner.toml 格式不匹配预期 | 跳过自动编辑，输出手动添加的行，请用户手动处理 |
| Dockerfile FROM 镜像不存在 | 检查 https://hub.docker.com/r/pytorch/manylinux2_28-builder/tags |
| 本地 test-and-fix 同一错误反复出现 | 阅读社区 PyTorch RC tag 对应文件的 diff 来理解变更 |
| `git checkout -b` 失败（分支已存在） | 检查 `git branch`，如果已在正确分支上则继续 |
| 版本升级时遇到合并冲突 | 接受新版本号解决冲突，用 grep 重新验证 |
| `--force-with-lease` 推送被拒 | 远程有他人提交，确认后由用户决定是否 `--force` |

---

## 附录 G：Commit Squash 策略

远程分支只保留一次提交记录。所有修复必须 squash 到同一 commit。

**CC 职责：** 只做 `git add -A` 并生成 commit message。**git commit 和 git push 由用户手动执行。**

### 初始提交（每步首次推送，用户手动执行）

```bash
git checkout -b <branch> master
git add -A   # CC 已完成
git commit -m "<描述性提交信息>"
git push -u origin <branch>
```

### 修复提交（CI 报错后的修复迭代，用户手动执行）

**方法一（推荐）：commit --amend**

```bash
git add -A   # CC 已完成
git commit --amend --no-edit
git push --force-with-lease origin <branch> --no-verify
```

**方法二：reset --soft**

如果需要在 squash 的同时修改 commit message：

```bash
git add -A   # CC 已完成
git reset --soft HEAD~1
git commit -m "<新的提交信息>"
git push --force-with-lease origin <branch> --no-verify
```

### 重要说明

- **永远使用 `--force-with-lease`**，不要使用 `--force`。`--force-with-lease` 会检测远程是否有他人提交，有则拒绝推送并告警。
- **`--no-verify` 跳过 pre-push hook**，加快迭代速度。如果项目要求必须跑 hook，去掉此参数。
- **`--no-edit` 复用原 commit message**，减少交互步骤。
- **如果 commit --amend 报错**（如没有历史 commit），说明这是分支的首次提交，改用 `git commit` 而非 `--amend`。
- **禁止 `git rebase -i` 或任何交互式操作**——CC 无法处理交互式编辑器。

---

## 附录 H：版本号存在性验证

每次修改版本号后，**必须构造具体 whl 下载链接验证版本是否真实存在**，而非仅检查 index 页面可访问。

### 核心原则

**index 页面 200 ≠ 版本存在。** 必须用具体 whl 文件路径探测，HTTP 404 意味着该版本还未构建或版本号写错。

### whl URL 构造规则

PyTorch 的 pip 仓库结构为 `<base_url>/<package_name>/<whl_filename>`。版本号中的 `.dev` → URL 中写成 `-dev`：

```
# 正式版本
https://download.pytorch.org/whl/cpu/torch-2.13.0%2Bcpu-cp311-cp311-linux_x86_64.whl

# dev 版本（.dev → -dev）
https://download.pytorch.org/whl/nightly/cpu/torch-2.14.0-dev20260701%2Bcpu-cp311-cp311-linux_x86_64.whl
```

**简化验证（推荐）：** 访问包目录页搜索版本号：

```bash
# 用 WebFetch 访问包目录页，搜索目标版本号
# WebFetch https://download.pytorch.org/whl/nightly/cpu/torch/  搜索目标 dev 版本号
# WebFetch https://download.pytorch.org/whl/test/cpu/torchvision/  搜索目标版本号
```

### 验证方法

**方法一（推荐）：curl 探测具体 whl 文件**

```bash
curl -I -L --max-time 15 "<whl_url>" 2>&1 | grep -E "^HTTP|404|Location"
```

- HTTP 200 / 302 → ✅ 版本存在
- HTTP 404 → ❌ 版本不存在或未构建

**方法二：WebFetch 访问包目录页**

使用 WebFetch 访问 `<base_url>/torch/` 或 `<base_url>/torchvision/` 目录页，搜索目标版本号字符串。目录页列出所有可用版本。

### 常见失败模式

| 失败类型 | 可能原因 | 处理方式 |
|---------|---------|---------|
| HTTP 404（版本不存在） | Nightly 尚未构建完成 | 等待到 UTC 12:00 左右，或检查版本号日期是否正确 |
| HTTP 404（RC 版本） | RC 未发布到此频道 | 确认社区 PyTorch release page 是否有该 RC |
| HTTP 404（正式版本） | 版本号写错 | 检查 PyTorch GitHub release 确认正确版本号 |
| Connection timeout | 网络不通 | 检查外网连通性 |
| index 页面 200 但 whl 404 | 包名拼写错误 | 检查包名和版本号格式 |

### 验证清单

| Step | 需验证的版本 | extra-index-url |
|------|------------|-----------------|
| Step 1 | torch: `$NEW_TORCH`, torchvision: `.dev<日期>` | nightly/cpu |
| Step 2 | torch: `$RC_TORCH`, torchvision: `$RC_TORCHVISION`, pyg URL（如有） | test/cpu |
| Step 3 | torch: `$NEXT_VERSION.dev$DEV_DATE`, torchvision: | nightly/cpu |
| Step 4 | torch: `$REL_VERSION`, pyg URL | cpu |

### 失败时处理流程

1. 输出 `⚠️ <包名>==<版本> 在 <URL> 中不存在（HTTP 404）`
2. 使用 AskUserQuestion 询问：
   > **问题：** `<版本号>` 在对应频道中未找到。可能还未构建完成或版本号写错。如何处理？
   > **标题：** 版本不存在
   > **选项：** "版本号写错了，我要修正" / "等待镜像构建，稍后重试" / "跳过验证，继续执行"
3. 如用户选择跳过，在最终 checklist 中标记 `⚠️ 版本未验证`
