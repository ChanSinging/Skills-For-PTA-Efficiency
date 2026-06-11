# Ascend NPU Pipeline P2P 显存泄漏分析报告

## 1. 问题概述

| 项目 | 说明 |
|------|------|
| 环境 | Ascend 910B, torch 2.11rc1, torch_npu 2.11 |
| 模型 | Qwen3-8B (4 层 debug 模型) |
| 并行配置 | PP=2, DP=1, TP=1, DualPipeV schedule, 4 microbatches |
| 现象 | 每步泄漏 ~134 MB (rank 0) / ~159 MB (rank 1)，200 步后 OOM |
| 根源 | HCCL `dist.irecv` 每步分配新的内部 receive buffer，C 层引用阻止 Python GC 释放 |

---

## 2. 排查链

### 2.1 确认泄漏存在

**手段**：在 `_PipelineScheduleRuntime._step_microbatches` 中插入 patch，step 前后各采样一次 `memory_allocated()`：

```python
def _patched_step(self, *args, **kwargs):
    stats_begin = torch.npu.memory_allocated()
    try:
        return _orig_step(self, *args, **kwargs)
    finally:
        stats_end = torch.npu.memory_allocated()
        # BEGIN-baseline 线性增长 → 泄漏存在
```

输出：

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

**手段**：`gc.get_objects()` 全量扫描，按 `(shape, dtype)` 分组统计 count 和 total_bytes。在 step 2→3 各采样一次做 diff，找出 count 增长 ≥4 的 shape：

```python
def _gc_scan():
    torch.npu.synchronize()
    gc.collect()
    groups = collections.defaultdict(lambda: [0, 0])  # key → (count, total_bytes)
    for obj in gc.get_objects():
        if type(obj) is not torch.Tensor or obj.device.type != "npu":
            continue
        sz = obj.numel() * obj.element_size()
        key = (tuple(obj.shape), obj.dtype)
        g = groups[key]
        g[0] += 1
        g[1] += sz
    return dict(groups)

# 2-step diff 发现泄漏 shape
if step in (2, 3):
    cur_groups = _gc_scan()
    if prev_groups:
        for key in set(prev_groups.keys()) | set(cur_groups.keys()):
            delta = cur_cnt - prev_cnt
            if delta >= 4:                   # 增长 ≥4 → 候选泄漏 shape
                candidates.append((delta, key))
    prev_groups = cur_groups
```

输出：

```
[GC_SCAN] step=3: [402.7MB x48] 1x1024x4096 torch.bfloat16
[GC_SCAN] step=4: [536.9MB x64] 1x1024x4096 torch.bfloat16  (+16 tensors)
[GC_SCAN] step=5: [671.1MB x80] 1x1024x4096 torch.bfloat16  (+16 tensors)
```

每步严格递增 16 个 `[1, 1024, 4096] bf16` tensor（8.4 MB/个）。

### 2.4 定位持有者

**手段**：BFS 从 `self._stages` 遍历所有属性，追踪到 `PipelineStage.args_recv_info[*][*].buffer`

```python
def _trace_holders(stages):
    """BFS 遍历 stages 全部属性，找到持有的 NPU tensor"""
    for i, stage in enumerate(stages):
        visited = set()
        queue = [(f"stages[{i}]", stage)]
        while queue:
            path, obj = queue.pop(0)
            if id(obj) in visited:
                continue
            visited.add(id(obj))
            if type(obj) is torch.Tensor and obj.device.type == "npu":
                print(f"[GC_HOLD2] {path} → tensor[{tuple(obj.shape)}] {obj.dtype}")
            if hasattr(obj, "__dict__"):
                for attr, val in vars(obj).items():
                    queue.append((f"{path}.{attr}", val))
            elif isinstance(obj, (list, tuple)):
                for j, item in enumerate(obj):
                    queue.append((f"{path}[{j}]", item))
```

输出：

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

**关键发现**：用 `gc.get_referrers()` 验证引用链 — 泄漏 tensor 的引用者全部显示为空或不可达：

```python
for obj in gc.get_objects():
    if type(obj) is torch.Tensor and tuple(obj.shape) == (1, 1024, 4096):
        referrers = gc.get_referrers(obj)
        # referrers 中没有任何 Python 对象持有该 tensor
        # → C 层引用对 gc.get_referrers() 完全不可见
```

`gc.get_referrers()` 只能追踪 Python 对象到 Python 对象的引用链。HCCL C 代码持有的 tensor 引用对 Python GC 完全不可见。这是 Python 无法跨越的墙。

### 2.6 确认绕过 C 层引用的方法

在 finally 块中对自动检测到的目标 shape 的 NPU tensor 调用 `.storage().resize_(0)`：

```python
_resize_targets = [((1, 1024, 4096), torch.bfloat16)]  # 来自 2.3 的 auto-detect
target_set = set(_resize_targets)
for obj in gc.get_objects():
    if type(obj) is not torch.Tensor or obj.device.type != "npu":
        continue
    if (tuple(obj.shape), obj.dtype) not in target_set:
        continue
    obj.storage().resize_(0)   # 绕过 Python + C 层引用直接释放 NPU 显存
    resized_count += 1
```

`.storage().resize_(0)` 直接调用 NPU 驱动释放显存，不依赖 Python refcnt 和 C 层引用计数。

### 2.7 确认 resize 单独不生效的原因

resize 后分别在 `empty_cache()` 前后采样 `memory_reserved()`，对比 freed 量：

```python
stats_before = torch.npu.memory_reserved()
torch.npu.empty_cache()
stats_after = torch.npu.memory_reserved()
freed = stats_before - stats_after
# expandable_segments:True  → freed ≈ 0，reserved 持续增长
# expandable_segments:False → freed > 0，reserved 可回收
```

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
[MEM_TRACE] step=50  BEGIN allocated=9.264GB   完全稳定
[MEM_TRACE] step=100 BEGIN allocated=9.264GB   纹丝不动
[EMPTY_CACHE] step=N reserved 22.1GB→12.4GB (freed 9732.9MB)  每步回收
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

## 4. PTA 侧修复

### 4.1 torch_npu 修复

**文件**：torch_npu 中 `dist.irecv` / `P2POp` 的实现

**修复方向**：
1. `irecv` 应直接写入传入的 tensor storage，而非内部分配新 buffer
2. 如果必须使用中间 buffer（HCCL 限制），应在 P2P 完成后立即释放，不保留引用

**代码链接**
https://gitcode.com/Ascend/pytorch/pull/32317

### 4.2 前后修复PR对比
从最近的百来次PR，经过二分定位，定位到!32317PR成功修复了泄露

#### 泄露根因

PR [#32267] 中 NPU 侧的 `_batch_isend_irecv` 直接调用 C++ 的 `batch_isend_irecv`，每个 P2P op 独立创建 Work 对象，缺乏统一的 coalescing 框架管理生命周期：

```python
# PR #32267 — 泄露路径
return [_group.batch_isend_irecv(op_type, tensors, remote_rank_list)]
```

C++ 侧 `pointToPoint()` 中 `recorded_inputs_`、`recordStream` 无条件执行，Work 对象粒度过细，tensor 存储引用无法及时回收。

#### 修复方案

PR [#32317] 在 Python 侧对支持设备（Ascend910B1+ / CANN >= 9.0.0）改用 coalescing manager，C++ 侧引入 `coalescing_state_` 状态机：

```python
# PR #32317 — 修复路径
if cann_version >= "9.0.0" and is_supported_device_name:
    with _coalescing_manager(group=group, device=device, async_ops=True) as cm:
        for p2p_op in p2p_op_list:
            p2p_op.op(...)
    return cm.works
else:
    return [_group.batch_isend_irecv(op_type, tensors, remote_rank_list)]
```

#### C++ 侧关键变更（ProcessGroupHCCL.cpp）

| 改动点 | PR #32267 | PR #32317 |
|---|---|---|
| Work 创建 | 每个 op 独立创建 | coalescing 时延迟到 `endCoalescing` |
| tensor 暂存 | 无 | `coalescedTensors_.stash(tensors)` |
| `recorded_inputs_` | 无条件 push | `!coalescing_state_` 守卫 |
| `recordStream` | 无条件执行 | `!coalescing_state_` 守卫 |
| seqP2P_ 递增 | 每个 op 递增 | coalescing 时按 group 递增一次 |

核心思路：将多个 P2P ops 合并为一个 coalescing group 统一调度，从架构层面解决 Work 对象粒度和 tensor 生命周期管理问题。
