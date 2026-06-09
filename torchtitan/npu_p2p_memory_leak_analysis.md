# Ascend NPU Pipeline P2P 显存泄漏分析报告

## 1. 问题概述

| 项目 | 说明 |
|------|------|
| 环境 | Ascend 910B3, torch 2.11, torch_npu 2.11 |
| 模型 | Qwen3-8B (4 层 debug 模型) |
| 并行配置 | PP=2, DP=1, TP=1, DualPipeV schedule, 4 microbatches |
| 现象 | 每步泄漏 ~134 MB (rank 0) / ~159 MB (rank 1)，200 步后 OOM |
| 根源 | HCCL `dist.irecv` 每步分配新的内部 receive buffer，C 层引用阻止 Python GC 释放 |

---

## 2. 排查链

### 2.1 确认泄漏存在

**手段**：per-step `torch.npu.memory_allocated()` 采样

```
[MEM_TRACE] step=N BEGIN allocated=19.5GB → step=N END allocated=26.1GB
[MEM_TRACE] step=N+1 BEGIN allocated=19.6GB  ← baseline 持续上涨
```

allocated 在 step 之间的 **baseline 线性增长**，确认有泄漏。

### 2.2 排除非泄漏源

| 检查项 | 结论 |
|--------|------|
| `optimizer.step()` 前后 allocated 差值 | Δ = 0.0 MB，排除 optimizer |
| `fwd_cache` / `bwd_cache` 大小 | 稳定，不是缓存膨胀 |
| `output_chunks` | 稳定 |

### 2.3 定位泄漏形状

**手段**：`gc.get_objects()` 全量扫描，按 `(shape, dtype)` 分组

```
[GC_SCAN] step=3: [402.7MB x48] 1x1024x4096 torch.bfloat16
[GC_SCAN] step=4: [536.9MB x64] 1x1024x4096 torch.bfloat16  (+16 tensors)
[GC_SCAN] step=5: [671.1MB x80] 1x1024x4096 torch.bfloat16  (+16 tensors)
```

每步严格递增 16 个 `[1, 1024, 4096] bf16` tensor（8.4 MB/个）。

### 2.4 定位持有者

**手段**：BFS 从 `self._stages` 遍历所有属性，追踪到 `PipelineStage.args_recv_info[*][*].buffer`

```
[GC_HOLD2] walk from known API objects:
  ✓ self._stages[1] → _stages[1].args_recv_info
```

Tensor 属性：
```
grad_fn=None, requires_grad=False, is_leaf=True, _version=0
```
纯 P2P 接收 buffer，不在 autograd 图中，不会被 `backward()` 自动释放。

### 2.5 确认 Python GC 无法释放

**尝试的 Python 侧清理方案（全部失败）**：

| 方案 | 结果 | 失败原因 |
|------|------|----------|
| `r.buffer = None` | 崩溃 | `clear_runtime_states()` 执行 `r.buffer.grad = None`，NoneType 报错 |
| `r.buffer = dummy_tensor` | 不崩但不生效 | 旧 tensor 被 HCCL C 层持有引用，`gc.get_referrers()` 看不到 C 层引用 |
| 清空 `args_recv_info` list | 不生效 | 同上，C 层引用在 Python GC 视野之外 |
| `gc.collect()` + `del` | 不生效 | Python GC 无法跨越 C 层引用屏障 |

**关键发现**：`gc.get_referrers()` 只能追踪 Python 对象到 Python 对象的引用链。HCCL C 代码持有的 tensor 引用对 Python GC 完全不可见。这是 Python 无法跨越的墙。

### 2.6 确认绕过 C 层引用的方法

`.storage().resize_(0)` 直接调用 NPU 驱动释放显存，不依赖 Python refcnt 和 C 层引用计数。

### 2.7 确认 resize 单独不生效的原因

resize 后 `allocated` baseline 稳定（不再增长），但 **reserved 内存持续膨胀**：

```
PYTORCH_NPU_ALLOC_CONF="expandable_segments:True" 下：
每步 alloc(16×8.4MB) → resize(16个洞) → 洞无法复用 → expandable_segments 不断开辟新段
200步后 reserved 碎片填满 NPU → OOM
```

### 2.8 最终组合方案验证

| 配置项 | 作用 |
|--------|------|
| `.storage().resize_(0)` | 绕过 C 层引用，强制释放 NPU 显存 |
| `expandable_segments:False` | 传统 caching allocator，可复用 resize 释放的内存块 |
| `empty_cache()` | 将完全空闲的 segment 归还 OS |

验证结果：
```
[MEM_TRACE] step=10  BEGIN allocated=9.264GB
[MEM_TRACE] step=50  BEGIN allocated=9.264GB  ← 完全稳定
[MEM_TRACE] step=100 BEGIN allocated=9.264GB  ← 纹丝不动
[EMPTY_CACHE] step=N reserved 22.1GB→12.4GB (freed 9732.9MB) ← 每步回收
```

---

## 3. 根因分析

### 3.1 直接原因

HCCL 的 `dist.irecv` 实现在 Ascend NPU 上每步分配**新的内部 buffer tensor**，P2P 完成后这些 tensor 的引用留在 C 层，Python GC 无法回收。
self.args_recv_info 在 pipeline 完成后仍持有旧 buffer 的引用。

### 3.2 代码路径

```
torchtitan/train.py
  → trainer.train_step()
    → pp_schedule.step()
      → _PipelineScheduleRuntime._step_microbatches()
        → DualPipeV schedule
          → _batch_p2p(stage.get_fwd_recv_ops(mb_idx))
            → stage.get_fwd_recv_ops()                           # stage.py:460
              → self._get_recv_ops(recv_infos)                   # stage.py:348
                → dist.P2POp(dist.irecv, info.buffer, ...)       # stage.py:369
                  → torch_npu / HCCL 层
                    → 内部分配新 buffer（BUG），接收数据，拷贝到 info.buffer
                    → 新 buffer 引用留在 C 层（泄漏）
```

### 3.3 PTA 源码缺陷位置

| 组件 | 缺陷 |
|------|------|
| **torch_npu** `dist.irecv` 实现 | 未复用传入的 buffer，内部分配新 tensor 用于接收 |
| **HCCL** P2P receive | 完成接收后未释放内部临时 buffer 的引用 |

正常行为（CUDA/NCCL）：
```
dist.irecv(tensor, peer)  →  NCCL 直接写入 tensor 的 storage，不创建新 tensor
```

Ascend NPU 实际行为：
```
dist.irecv(tensor, peer)  →  HCCL 创建新 tensor → 接收数据 → 拷贝到 tensor
                            → 新 tensor 的 C 层引用不释放（泄漏）
```

### 3.4 为什么每步恰好泄漏 16 个 tensor

```
PP=2, 4 microbatches, DualPipeV:
  Stage 0 → Stage 1: 4 × fwd activation send/recv = 8 ops
  Stage 1 → Stage 0: 4 × bwd gradient send/recv   = 8 ops
  Total                                             = 16 ops/step
```

每个 `dist.irecv` 泄漏 1 个 `[1, seq_len, hidden_dim]` bf16 tensor。

---

## 4. torchtitan 侧修复策略（workaround）

文件：`torchtitan/_npu_workarounds.py`

### Patch 1：`autograd.grad` 强制 `allow_unused=True`

NPU autograd 对未使用参数的检测不准确，会导致误报 "does not appear to be used in the graph" 错误。

### Patch 2：P2P recv buffer 泄漏 workaround

```
每步 _step_microbatches 的 finally 块中：

1. cleanup unconsumed P2P ops
   ↓
2. clear_runtime_states() + 替换 args_recv_info/grad_recv_info buffer 为 dummy tensor
   目的：防止 clear_runtime_states 在旧 buffer 上累积 grad
   ↓
3. .storage().resize_(0) 清零所有泄漏 tensor 的 storage
   通过 GC 扫描找到目标 shape（auto-detected）的 NPU tensor
   直接 release 底层显存分配，绕开 Python/C 引用计数
   ↓
4. torch.npu.empty_cache()
   配合 expandable_segments:False，回收空闲 segment 给 OS
```

### 通用性设计

- **自适应 shape 检测**：step 2→3 的 GC diff 自动发现增长最快的 NPU tensor shape，不硬编码
- 适配任意 seq_len、hidden_dim、PP 度
- 不依赖模型结构、层数
- 不推荐，最好在PTA的cpp侧后端底层修改

---

## 5. PTA 侧修复建议

### 5.1 torch_npu 修复

**文件**：torch_npu 中 `dist.irecv` / `P2POp` 的实现

**修复方向**：
1. `irecv` 应直接写入传入的 tensor storage，而非内部分配新 buffer
2. 如果必须使用中间 buffer（HCCL 限制），应在 P2P 完成后立即释放，不保留引用

**代码链接**
https://gitcode.com/Ascend/pytorch/pull/36161

### 5.2 HCCL 修复

**文件**：HCCL P2P receive 相关代码

**修复方向**：
1. 确保 receive 操作复用外部传入的内存，不分配额外 buffer
2. 如有内部分配，在 wait/同步点后释放

### 5.3 验证方法

修复后，以下指标应全部满足：
- per-step GC 扫描：`[1, seq_len, hidden_dim]` tensor 数量不随步数增长
- `memory_allocated()` baseline 稳定
- `memory_reserved()` 不持续膨胀
- 不依赖 `_npu_workarounds.py` 即可稳定训练 1000+ 步
