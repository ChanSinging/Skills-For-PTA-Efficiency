# Pipeline Parallelism: release/2.12 → release/2.13 差异分析

## 涉及模块

- `torch/distributed/pipelining/` — 分布式 pipeline 并行（PyTorch 原生模块）
- `torch/_inductor/` — 编译期 Triton software pipelining（非分布式，但相关）

文件结构在 2.12 和 2.13 之间无变化，共 11 个文件：

```
torch/distributed/pipelining/__init__.py
torch/distributed/pipelining/_backward.py
torch/distributed/pipelining/_debug.py
torch/distributed/pipelining/_IR.py
torch/distributed/pipelining/_schedule_visualizer.py
torch/distributed/pipelining/_unflatten.py
torch/distributed/pipelining/_utils.py
torch/distributed/pipelining/microbatch.py
torch/distributed/pipelining/schedules.py
torch/distributed/pipelining/stage.py
torch/distributed/pipelining/README.md
```

总体 diff 量：7 个文件变更，~612 行新增，~171 行删除。

---

## 一、新增功能（3 个 PR）

### 1. #178815 Defer pipeline RECV ops with rank-parity deadlock avoidance

**影响文件**: `schedules.py`, `_schedule_visualizer.py`

在部分平台（如 AMD ROCm）上，pending RECV 会阻塞同一 communication fabric 上的无关 compute ops，产生 pipeline bubble。

**新增**:
- `overlap_pp_comm` 参数（默认 `True`），添加到所有 multi-schedule 构造函数：`ScheduleLoopedBFS`、`ScheduleInterleaved1F1B`、`ScheduleInterleavedZeroBubble`、`ScheduleZBVZeroBubble`、`ScheduleDualPipeV`
- 当设为 `False` 时，调用 `_defer_recv_ops()` 将 RECV 推迟到消费其数据的 compute op 之前执行

**死锁避免**: 通过 rank-parity P2P ordering —— 低 rank 方（`rank < peer`）在向 peer SEND 前先 flush deferred RECV，确保 RECV 先于 SEND 被 post。

附 464 行新测试。

### 2. #181057 [PP] Support extra loss_fn kwargs in pipeline schedules

**影响文件**: `schedules.py`, `stage.py`

**新增** `loss_kwargs` 参数，全链路透传：

```
step() → _step_microbatches() → _maybe_compute_loss() → _compute_loss()
_initialize_stage() → _initialize_pp_stages() → _prepare_backward_infra() → _backward_metadata_inference()
```

**使用场景**: chunked cross-entropy loss 需要 `global_valid_tokens` 等额外参数：

```python
schedule.step(..., loss_kwargs={"global_valid_tokens": tokens})
```

kwargs 作为参数传递而非存储在 schedule 对象上，因为它们随 step 变化，不应作为持久状态。

### 3. #179293 Guard pipeline schedule adjacency

**影响文件**: `schedules.py`

在 `PipelineScheduleMulti._initialize_stages()` 中新增 `_validate_adjacent_stage_communication()`。

**背景**: `PipelineScheduleMulti` 假设通信仅发生在相邻 stage 之间（`stage_idx ± 1`），但此前未强制执行此约束。非相邻通信（如 skip-connection 拓扑）会静默产生错误结果或 cryptic 运行时错误。

**实现**: 遍历 `args_recv_info` 和 `act_send_info`，检查 recv sources 和 send destinations 是否在预期的相邻 stage 集合中，不符合则抛出清晰的 `RuntimeError`。

在 init 时执行一次，不影响热路径性能。

---

## 二、Bug 修复（5 个 PR）

### 4. #181365 Fix shared-weight gradient double-counting in zero-bubble pipeline schedules

**影响文件**: `_backward.py`（重构 ~158 行）

**根因**: `ScheduleInterleavedZeroBubble` 将 backward 拆分为 `stage_backward_input` 和 `stage_backward_weight` 两阶段。当 stage 内的 module 有共享权重（tied weights）时，weight phase 会重复累加梯度。

**修复**: 在 weight phase 中注册 pre-hook：当 param group 有多个 intermediates 时，用 input phase 保存的梯度值替换 incoming gradients，防止双重计算。

同时 `stage_backward_input` 做了以下改进：
- 过滤 `valid_outputs` 只保留 `requires_grad=True` 的
- 使用 `allow_unused=True` 调用 `autograd.grad`
- 对非 Tensor input 返回 `None` grad
- 添加 try/except/finally 确保 hook 被正确清理

### 5. #182182 Fix None gradient handling in backward send/recv

**影响文件**: `stage.py`, `_backward.py`, `_utils.py`, `schedules.py`

修复多个 backward 梯度通信问题，与 #181365 配合使用：

- **保守分配 grad recv buffers**: 对可微的 plain tensor 预分配 recv buffer，使后续 microbatch 能发送真实梯度（即使推理 microbatch 产生了 None）
- **DTensor 梯度元数据不再从 activation 元数据推测**: 若 inference 记录 None 而 runtime 产生 DTensor 梯度，直接报错，而不是静默丢弃或猜测 placement
- **Runtime backward send 三层分发**:
  1. metadata 分配了 buffer 且 runtime 返回真实梯度 → 发送
  2. metadata 分配了 buffer 但 runtime 返回 None → 发送零梯度
  3. 无 recv buffer 但 runtime 返回真实梯度 → 报错
- **Split input-backward 对齐 full backward**: 在 `autograd.grad` 前过滤非可微输出，允许 unused inputs
- **非 Tensor stage input 处理**: `FlexAttention BlockMask` 等对象不再调用 `.requires_grad`，直接返回 `None`

新增方法 `_get_grad_send_meta()` 到 `_PipelineStageBase`。

### 6. #182644 Fix pipelining crash when split_module interleaves get_attr with placeholder

**影响文件**: `_IR.py`

**根因**: `split_module` 可能产生 `get_attr` 与 `placeholder` 交错排列的分区子图，而 `_ModuleFrame.run_outer()` 假设 placeholders 是连续前缀。

**修复**: 在 `Pipe.__init__()` 的 `_outline_submodules` 之前，新增 `_move_placeholders_to_front()` 函数：

```python
def _move_placeholders_to_front(graph: fx.Graph):
    """Ensure all placeholder nodes form a contiguous prefix of the graph."""
    # 找到最后一个 placeholder，将之前的所有非 placeholder 节点移到它之后
```

验证通过 SmolLM2 模型（来自 issue #162898）。

### 7. #183582 Add guards for non-float tensors when building pipeline

**影响文件**: `stage.py`, `_utils.py`

**用户影响最大的修复**。此前 pipeline parallelism 对 HuggingFace transformers 模型完全不可用。

**根因**: `input_ids`（`torch.long`）和 `attention_mask`（`torch.bool`）等非浮点张量在 pipeline 中被无条件设置 `requires_grad=True`，而 PyTorch 只允许浮点张量可求导。

**修复 4 处**:
1. `stage.py` → `create_recv_tensor`: 用 `self.has_backward and example_value.is_floating_point()` 替换 `self.has_backward`
2. `stage.py` → `_create_act_send_info`: 同上
3. `_utils.py` → `_TensorMeta.to_tensor()`: 同守卫
4. `_utils.py` → `_DTensorMeta.to_tensor()`: 同守卫

### 8. #180937 [DTensor][PP] Correctly shard the input to preserve DTensor placements

**影响文件**: `schedules.py`, `microbatch.py`

**根因**: `torch.split` 对 DTensor 会改变 placements（`Shard(0)` → `Replicate()` — 隐式 all-gather）。

**修复**:
- `schedules.py`: target splitting 路径改用 PP 已有的 `_split_tensor` 工具函数
- `microbatch.py`: `_split_tensor`、`_expand_chunks`、`merge_chunks` 完全重写，移除 `local_map` 用法，改为直接 `DTensor.from_local()` 调用加显式 `shape`/`stride` 参数

**行为变更** (target-splitting 路径):
- DTensor target with `Shard(0)` 现在保留 `Shard(0)` 而不是隐式 gathered 到 `Replicate()`
- `target.requires_grad and target.is_leaf` 时 split chunks 调用 `retain_grad()`
- `target.size(0) < n_microbatches` 时抛出 `AssertionError`（之前静默返回较少 chunks）

---

## 三、Inductor 编译期 Software Pipelining（非分布式）

### #180892 Enable software pipelining for mix-order reduction

**影响文件**: `torch/_inductor/config.py`, `torch/_inductor/runtime/triton_heuristics.py`

**类型**: 编译器后端优化，非分布式并行策略。属于 Triton kernel 级别的指令流水线。

启用 `mix_order_reduction_allow_multi_stages` 默认值，使 mix-order reduction kernel 受益于 Triton software pipelining（`NUM_STAGES > 1`）。

**fallback 机制**: 在 `CachingAutotuner._make_launchers` 中，当所有 configs 均因 `OutOfResources` / `OutOfMemoryError` 失败且 pipelining 处于活动状态时，自动以 `num_stages=1` 重新编译重试。保证大 kernel 能正常运行，小 kernel 获得流水线性能收益。

---

## 四、代码变更量统计

| 文件 | 变更 |
|------|------|
| `schedules.py` | +325, -72 (最⼤变更) |
| `_backward.py` | +182, -49 |
| `stage.py` | +103, -28 |
| `microbatch.py` | +92, -36 |
| `_utils.py` | +45, -20 |
| `_IR.py` | +26, -4 |
| `_schedule_visualizer.py` | +10, -2 |

---

## 五、其他次要提交

以下提交也触及了 `torch/distributed/pipelining/`，但不涉及功能性变更：

| 提交 | 描述 |
|------|------|
| `64413b0038f` | 修复 `fork_rng` 指定 device type，支持非 CUDA 设备 |
| `a9926f51c7b` / `0464301f944` | 自动化 ruff PERF102 清理（移除不必要的 list→generator) |
| `4fec9b8dd8f` | 修复注释和 docstring 中的拼写错误 |
| `38a7633d42d` | 修复代码库中的重复单词拼写错误 |
| `8a55f2e606c` | 移除不再需要的 noqa 指令 |

---

## 六、总结

| 维度 | 关键变化 |
|------|----------|
| **新功能** | RECV 延迟策略（消除 ROCm pipeline bubble）、loss_fn kwargs 支持、adjacency 校验 |
| **正确性修复** | ZBV 共享权重梯度修复、None 梯度处理、非浮点张量守卫、DTensor placements 保留 |
| **兼容性** | HF transformers 模型从完全不可用变为可用 |
| **架构变更** | 无新增公共 API 类，无文件结构变化。`defer_pp_recv` + `loss_kwargs` 穿透全调度链路 |
| **API 变更** | 所有 multi-schedule 构造函数新增 `overlap_pp_comm` 参数；`step()` 新增 `loss_kwargs` |
