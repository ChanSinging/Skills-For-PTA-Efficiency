## PTA v2.13.0 PP 适配分析

**总体结论：没有必须修改的 PTA 代码，但需要验证 + 补充测试。**

PTA 的 PP 架构是"薄封装"模式——NPU 自定义的只有 `pipeline()` API（`_pipelining_patch.py`）和设备适配（`transfer_to_npu.py`），所有 Schedule/Stage/Backward 逻辑直接继承上游。上游 2.13 的 8 个 PP 变更中，大部分自动生效。

---

### 一、自动生效，无需适配（6 项）

| PR | 变更 | 原因 |
|---|---|---|
| #181057 loss_kwargs | `step()` 新增参数全链路透传 | PTA 直接使用上游 schedules，参数自动继承 |
| #179293 adjacency 校验 | `_validate_adjacent_stage_communication()` | 在上游 `PipelineScheduleMulti._initialize_stages()` 中，自动生效 |
| #181365 共享权重梯度修复 | `_backward.py` 重构 | PTA 使用上游的 backward 函数，无需改动 |
| #182182 None 梯度处理 | `stage.py` 新增 `_get_grad_send_meta()` | 该方法在 `_PipelineStageBase` 上，不涉及 device 引用，`transfer_to_npu.py` 无需额外包装 |
| #180937 DTensor placements | `microbatch.py` 重写 `_split_tensor` 等 | 上游工具函数，PTA 直接使用 |
| #180892 Inductor software pipelining | `torch/_inductor/` | 编译器层面优化，非分布式 PP，不相关 |

---

### 二、需要验证的（2 项）

#### 1. #178815 overlap_pp_comm + `_defer_recv_ops()` — 需要 HCCL 兼容性验证

这是最需要关注的一项。所有 multi-schedule 构造函数（`ScheduleLoopedBFS`、`ScheduleInterleaved1F1B`、`ScheduleInterleavedZeroBubble`、`ScheduleZBVZeroBubble`）新增 `overlap_pp_comm` 参数：

- `overlap_pp_comm=True`（默认）：行为与 2.12 一致，通信和计算 overlap
- `overlap_pp_comm=False`：调用 `_defer_recv_ops()` 推迟 RECV，通过 rank-parity P2P ordering 避免死锁

**风险点**：rank-parity 死锁避免策略是针对 NCCL/ROCm 设计的，HCCL 的 P2P RECV/SEND ordering 语义可能不同，`overlap_pp_comm=False` 在 NPU 上可能死锁。

**建议**：
- 测试 `overlap_pp_comm=False` 在 HCCL 多卡下的表现
- 如果 HCCL 行为与 NCCL 一致 → 无需任何代码改动
- 如果死锁 → 需要 PTA 自定义 `_defer_recv_ops()` 实现（当前 PTA 没有任何 `_defer_recv` 相关代码）

#### 2. `_modify_graph_op_device` 兼容性 — 需要确认上游未变更

PTA 的 `_quiet_modify_graph_op_device()`（`_pipelining_patch.py:172`）通过 `_apply_quiet_graph_device_patch()` 替换上游 `_IR._modify_graph_op_device`。

**风险点**：如果 2.13 的上游 `_IR.py` 重命名或删除了 `_modify_graph_op_device`，PTA 的 patch 会静默失效（有守卫检查 `original is None` 时会跳过）。虽然比较文档显示 `_IR.py` 只加了 `_move_placeholders_to_front()`，但还是需要确认。

**建议**：对比 2.13 上游 `_IR.py` 中 `_modify_graph_op_device` 是否存在且签名一致。

---

### 三、transfer_to_npu.py 的包装是否需要更新

当前包装：
- `torch.distributed.pipelining.stage.PipelineStage.__init__` = `_wrapper_cuda(...)`
- `_device_wrapper(torch.distributed.pipelining.stage, ['build_stage'])`

分析：
- `PipelineStage.__init__`：上游参数无变化，`_wrapper_cuda` 继续生效
- `build_stage`：上游签名无变化，`_device_wrapper` 继续生效
- 2.13 新增的 `_get_grad_send_meta()` 不引用 CUDA device，无需额外包装

**结论：不需要修改。**

---

### 四、需要补充的测试

| 优先级 | 测试内容 |
|---|---|
| 高 | `overlap_pp_comm=False` 在 HCCL 多卡下的正确性与死锁测试 |
| 高 | `loss_kwargs` 在 NPU 上的功能测试（含非浮点张量参数如 `input_ids`） |
| 中 | 非浮点张量守卫（#183582）在 NPU 上 HF 模型的集成测试 |
| 中 | ZBV schedule + 共享权重模型在 NPU 上的梯度正确性测试 |
| 低 | `_move_placeholders_to_front()` 与 PTA `_prune_none_root_placeholders()` 的交互验证 |

---

### 总结

| 类别 | 数量 |
|---|---|
| 需要改 PTA 代码 | 0 项 |
| 需要验证兼容性 | 2 项（overlap_pp_comm HCCL 行为 + `_modify_graph_op_device` 签名） |
| 需要补充测试 | 4 项 |

最关键的验证是 `overlap_pp_comm=False` 在 HCCL 上的行为——如果 HCCL 的 P2P 语义与 NCCL 有差异，可能需要在 PTA 中实现 NPU 专属的 `_defer_recv_ops()`。
