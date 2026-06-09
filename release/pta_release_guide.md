# PTA (PyTorch Ascend Adapter) 版本发布指南

本文档是 PTA 版本跟随和发布的完整操作手册，覆盖从社区 PyTorch Nightly 到正式 Release 的全流程。

## 目录

- [PTA (PyTorch Ascend Adapter) 版本发布指南](#pta-pytorch-ascend-adapter-版本发布指南)
  - [目录](#目录)
  - [概述](#概述)
  - [前置准备](#前置准备)
  - [Step 1: 升级 Dev Nightly 版本号](#step-1-升级-dev-nightly-版本号)
  - [Step 2: 切换到 RC 正式版本](#step-2-切换到-rc-正式版本)
  - [Step 3: 开启下一个 Dev 周期](#step-3-开启下一个-dev-周期)
  - [Step 4: 切正式 Release 分支](#step-4-切正式-release-分支)
  - [NPU 适配指南](#npu-适配指南)
    - [常见 Bug 类型](#常见-bug-类型)
    - [调试流程](#调试流程)
  - [常见问题 FAQ](#常见问题-faq)
    - [Q: 运行 bump-dev 后 CI 构建失败？](#q-运行-bump-dev-后-ci-构建失败)
    - [Q: switch-rc 后 `_dynamo.py` 报错？](#q-switch-rc-后-_dynamopy-报错)
    - [Q: cut-release 的 `--docker-from-image` 参数怎么确定？](#q-cut-release-的---docker-from-image-参数怎么确定)
    - [Q: README 中的 sha256 hash 链接怎么更新？](#q-readme-中的-sha256-hash-链接怎么更新)
    - [Q: op-plugin PR (Step 3) 一般要改什么？](#q-op-plugin-pr-step-3-一般要改什么)
    - [Q: 幂等性如何保证？](#q-幂等性如何保证)
  - [附录: 历史 PR 参考](#附录-历史-pr-参考)
    - [2.12.0 版本周期 PR 清单](#2120-版本周期-pr-清单)
    - [关键 commit diff 速查](#关键-commit-diff-速查)

## 概述

PTA 版本跟随流程以社区 PyTorch 版本为基准，分为 4 个阶段：

```
社区 Nightly         社区 RC             社区 Release          下一版本 Nightly
     │                  │                    │                      │
     ▼                  ▼                    ▼                      ▼
Step 1: bump-dev  Step 2: switch-rc   Step 3: bump-next     Step 4: cut-release
(dev 版本号升级)   (切 RC 正式版本)     (新 dev 周期)          (切 2.x.0 分支)
在 master 分支      在 master 分支       在 master 分支        创建 v2.x.0 分支
```

**自动化工具**: `python tools/pta_version_generator.py <step> [options]`

## 前置准备

- [ ] 确认当前工作目录在 PTA 仓库根目录
- [ ] `git status` 确认工作区干净（无未提交变更）
- [ ] 了解社区 PyTorch 的版本号（Nightly date、RC version、Release date）
- [ ] 知道新版本的 CANN 兼容版本号和文档路径编号

## Step 1: 升级 Dev Nightly 版本号

**时机**: 社区 PyTorch 更新 Nightly 版本号后（通常每 1-2 周）

**自动覆盖文件**:
- `requirements.txt` — torch 版本号
- `test/requirements.txt` — torchvision 版本号

**命令**:
```bash
python tools/pta_version_generator.py bump-dev --torch-version 2.13.0.dev20260601
```

**手动 Checklist**:
- [ ] 验证新 Nightly whl 可从 `https://download.pytorch.org/whl/nightly/cpu` 下载
- [ ] 验证 torchvision Nightly whl 同步可用

**合入分支**: `master`
**参考 PR**: Ascend/pytorch!34141

## Step 2: 切换到 RC 正式版本

**时机**: 社区 PyTorch 发布 RC (Release Candidate) 后

**自动覆盖文件**:
- `requirements.txt` — extra-index-url 切换 + torch 版本
- `test/requirements.txt` — extra-index-url 切换 + torchvision/onnx/pyg 版本

**命令**:
```bash
# 基础用法
python tools/pta_version_generator.py switch-rc --torch-version 2.13.0

# 完整用法（推荐）
python tools/pta_version_generator.py switch-rc \
  --torch-version 2.13.0 \
  --tv-version 0.28.0 \
  --onnx-version 1.22.0 \
  --pyg-torch-version 2.13.0
```

**手动 Checklist**（⚠️ 重要，每版本不同）:
- [ ] 修复 `torch_npu/utils/_dynamo.py` 中因 PyTorch API 变更导致的兼容问题
- [ ] 更新 `test/allowlist_for_publicAPI.json` 白名单
- [ ] 新增 `test/npu/` 下的测试用例文件
- [ ] 更新 `.lintrunner.toml` — 在所有 linter 段添加新测试文件排除规则
- [ ] 更新 `pyproject.toml` — 添加 extend-exclude
- [ ] 运行 CI，根据失败日志修复 NPU 适配问题
- [ ] 验证 `torch_npu/distributed/fsdp/` 相关改动（如有）

**合入分支**: `master`
**参考 PR**: Ascend/pytorch!35103

## Step 3: 开启下一个 Dev 周期

**时机**: RC 合入 master 后，社区开始下一版本 Nightly

**自动覆盖文件**:
- `version.txt` — 版本号
- `requirements.txt` — extra-index 回切 + torch dev 版本
- `test/requirements.txt` — extra-index 回切 + torchvision dev 版本

**命令**:
```bash
python tools/pta_version_generator.py bump-next --next-version 2.14.0 --dev-date 20260701
```

**手动 Checklist**:
- [ ] 在 op-plugin 仓库创建兼容新 PTA 分支的 PR（参考 PR 4923）
- [ ] 更新 torchair commit（如有需要）
- [ ] 验证 Nightly whl 可用

**合入分支**: `master`
**参考 PR**: Ascend/pytorch!35159, Ascend/op-plugin!4923

## Step 4: 切正式 Release 分支

**时机**: 社区 PyTorch 发布正式 Release 后

**自动覆盖文件**（使用 `--dry-run` 预览 → `--no-dry-run` 执行）:
- `requirements.txt` / `test/requirements.txt` — extra-index 切为正式链接
- `ci/docker/X86/Dockerfile` — FROM 镜像、Python、torch、numpy
- `ci/docker/ARM/Dockerfile` — 同上
- `ci/docker/README.md` — Python 示例版本
- `ci/build.sh` — SUPPORTED_TORCH_VERSION 追加
- `setup.py` — Python classifier 追加
- `.lintrunner.toml` — 测试文件排除规则追加
- `pyproject.toml` — extend-exclude 追加
- `README.md` / `README.zh.md` — 版本号、下载链接、文档链接、维护表格
- `SECURITYNOTE.md` — CANN 版本号、文档 URL、torch release 分支引用
- 创建 `v<version>` 分支

**命令**:
```bash
# Step 4a: Dry-run 预览所有变更
python tools/pta_version_generator.py cut-release \
  --release-version 2.13.0 \
  --docker-from-image pytorch/manylinux2_28-builder:cpu-v2.13.0-rc1 \
  --cann-version 8.5.0 \
  --python-versions 3.10,3.11,3.12 \
  --exclude-tests "test/npu/test_new.py" \
  --dry-run

# Step 4b: 确认无误后执行
python tools/pta_version_generator.py cut-release \
  --release-version 2.13.0 \
  --docker-from-image pytorch/manylinux2_28-builder:cpu-v2.13.0-rc1 \
  --cann-version 8.5.0 \
  --python-versions 3.10,3.11,3.12 \
  --no-dry-run
```

**手动 Checklist**:
- [ ] 验证 README 中 whl 下载链接可访问（特别是 sha256 hash）
- [ ] 验证 SECURITYNOTE.md 中所有 CANN 文档链接
- [ ] 检查 setup.py 中 Python 版本 classifiers
- [ ] 确认 Dockerfile 的 FROM 镜像可用
- [ ] 更新 op-plugin submodule 指向到 release 分支
- [ ] Push release 分支: `git push -u origin v2.13.0`
- [ ] 创建 PR 合入 release 分支

**合入分支**: `v<release_version>`（新分支）
**参考 PR**: Ascend/pytorch!35503

---

## NPU 适配指南

每次 PyTorch 版本升级，社区 API 变更可能导致 NPU 适配代码编译失败或测试失败。常见问题类型：

### 常见 Bug 类型

| 类型 | 典型文件 | 现象 | 修复思路 |
|------|---------|------|---------|
| API 签名变更 | `torch_npu/utils/_dynamo.py` | `TypeError: __init__() got unexpected keyword argument` | 对比社区 commit diff，调整参数名/顺序 |
| 内部模块变更 | `test/npu/test_public_bindings.py` | `ImportError` / 新增 module | 更新 allowlist，添加/删除模块 |
| 废弃 API | `torch_npu/distributed/*` | `DeprecationWarning` → 下一版本报错 | 提前替换为新 API |
| FSDP 变更 | `torch_npu/distributed/fsdp/*` | 测试失败 | 追踪社区 FSDP commit，适配 hook/state |
| Lint 规则变化 | `.lintrunner.toml` | CI lint 失败 | 添加排除规则 |

### 调试流程

1. 运行 CI 流水线，收集失败日志
2. 按文件分组失败用例
3. 在社区 PyTorch 对应版本 tag 中搜索相关文件变更
4. 对比 PTA 适配代码与社区代码差异
5. 修改适配代码或更新测试用例
6. 本地验证 `pytest test/npu/<failing_test>.py -x -v`
7. 推送到 PR 分支，重新运行 CI

---

## 常见问题 FAQ

### Q: 运行 bump-dev 后 CI 构建失败？
**A**: 检查 `https://download.pytorch.org/whl/nightly/cpu` 中是否已有对应 dev 日期的 whl。如没有，等待社区构建完成（通常 UTC 12:00 前）。

### Q: switch-rc 后 `_dynamo.py` 报错？
**A**: 社区可能在 RC 阶段改了 torch 内部 API。对比社区 RC tag 和上一个 Nightly 的 diff，找到对应变更并适配。

### Q: cut-release 的 `--docker-from-image` 参数怎么确定？
**A**: 查看社区 PyTorch 对应版本的 CI docker 配置，或从 `https://hub.docker.com/r/pytorch/manylinux2_28-builder` 查找对应 tag。

### Q: README 中的 sha256 hash 链接怎么更新？
**A**: sha256 hash 无法自动生成，需从 `https://download.pytorch.org/whl/cpu/` 页面手动复制对应 whl 的 hash。自动化脚本会替换版本号部分，但 hash 需手动检查。

### Q: op-plugin PR (Step 3) 一般要改什么？
**A**: op-plugin 仓库的 `ci/build.sh` 可能引用了 PTA 仓库地址和分支名，需改为新的 dev 分支。如果没有代码兼容性问题，通常只改 CI 配置。

### Q: 幂等性如何保证？
**A**: 脚本对 Dockerfile、lintrunner.toml、build.sh、setup.py 等文件的修改会先检查是否已有目标内容，避免重复添加。但建议每次操作前 `git stash` 备份。

---

## 附录: 历史 PR 参考

### 2.12.0 版本周期 PR 清单

| Step | PR | 仓库 | 说明 |
|------|-----|------|------|
| 1 | [!34141](https://gitcode.com/Ascend/pytorch/pull/34141) | pytorch | bump-dev: 2.12.0.dev20260217 → 20260317 |
| 2 | [!35103](https://gitcode.com/Ascend/pytorch/pull/35103) | pytorch | switch-rc: 切 2.12.0 RC, _dynamo fix, lint exclusion |
| 3 | [!35159](https://gitcode.com/Ascend/pytorch/pull/35159) | pytorch | bump-next: 2.12.0 → 2.13.0.dev20260416 |
| 3 | [!4923](https://gitcode.com/Ascend/op-plugin/pull/4923) | op-plugin | op-plugin 兼容 2.13 PTA 分支 |
| 4 | [!35503](https://gitcode.com/Ascend/pytorch/pull/35503) | pytorch | cut-release: v2.12.0 分支, 全量文档/Docker 更新 |

### 关键 commit diff 速查

| 文件 | PR | 变更要点 |
|------|-----|---------|
| `requirements.txt` | 34141 | `torch==2.12.0.dev20260217` → `torch==2.12.0.dev20260317` |
| `requirements.txt` | 35103 | `nightly/cpu` → `test/cpu`, dev → `2.12.0` |
| `test/requirements.txt` | 35103 | `torchvision==0.26.0.dev` → `0.27.0`, `onnx==1.20.0` → `1.21.0` |
| `torch_npu/utils/_dynamo.py` | 35103 | `Config(default=..., value_type=...)` → `Config("name", default=..., type=...)` |
| `version.txt` | 35159 | `2.12.0` → `2.13.0` |
| `ci/docker/X86/Dockerfile` | 35503 | FROM 镜像, python 3.9→3.10/3.11/3.12, torch 2.1→2.12 |
| `README.md` | 35503 | 大量版本号/日期/链接/表格更新 |
