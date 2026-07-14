# torch/distributed/pipelining 代码导读

## 1. 模块概览

`torch.distributed.pipelining` 是 PyTorch 的**流水线并行（Pipeline Parallelism, PP）**实现。核心思路是将一个大的模型按层切分成多个 stage，每个 stage 放在不同的 GPU 上，通过 P2P 通信传递中间激活，以 micro-batch 为单位流水线式执行，从而提高吞吐。

**总代码量**: 约 10,000 行 Python，10 个文件。

## 2. 目录结构与文件职责

```
torch/distributed/pipelining/
├── __init__.py          # 公开 API 导出
├── _IR.py              # 1280 行 - 自动分阶段前端（torch.export tracer + FX IR 切分）
├── stage.py            # 2200 行 - Stage 核心实现（手动/自动两种前端共用）
├── schedules.py        # 3900 行 - 各种流水线调度策略
├── _backward.py        #  480 行 - 自定义 backward 引擎（stage_backward 等）
├── _utils.py           #  940 行 - 元数据、工具函数、InferenceMode
├── microbatch.py       #  660 行 - micro-batch 切分与合并（含 DTensor 感知）
├── _debug.py           #   22 行 - 调试信息格式化
├── _unflatten.py       #   30 行 - FX graph module 子模块提取
└── _schedule_visualizer.py # 455 行 - 调度可视化工具（需 matplotlib）
```

## 3. 核心架构：两种创建前端

### 3.1 双前端设计

pipelining 提供两种创建 pipeline stage 的方式：

| 前端 | 入口 | 适用场景 |
|------|------|---------|
| **自动切分** (`_IR.py`) | `pipeline()` + `Pipe` + `pipe_split` | 对已有模型自动切分，无需手动指定每个 submodule |
| **手动切分** (`stage.py`) | `PipelineStage()` + `build_stage()` | 用户已手动将模型切分为 submodule 列表 |

两者最终都产生 `_PipelineStageBase` 的子类实例，由 schedule 统一驱动。

### 3.2 类继承关系

```
_PipelineStageBase (ABC)              # 公共基类：P2P、metadata、backward 基础设施
├── _PipelineStage                    # 自动切分前端：由 _IR.py 的 Pipe 创建
│                                     #   输入 PipeInfo，内部管理 fx.GraphModule
└── PipelineStage                     # 手动切分前端：用户直接提供 nn.Module
                                      #   支持 STATIC/DYNAMIC 两种 InferenceMode
```

### 3.3 自动切分前端工作流 (`_IR.py`)

```
用户模型
  → torch.export.export()            # 导出为 ExportedProgram（ATen 方言）
  → _sink_params()                    # 将参数/缓冲区下沉为 graph 输入
  → _outline_submodules()             # 展平嵌套子模块结构
  → split_module() + _split_on_graph  # 在 pipe_split 标记处切分 FX graph
  → _PipelineStage 列表              # 每个切分后的子图封装为一个 stage
```

关键类和函数：

- **`Pipe`** (line ~150): 顶层 API，接受 `nn.Module` + `pipe_split` 标记，自动完成 trace→split→build 全流程
- **`pipe_split()`**: 在模型 forward 中插入的标记 token，指示切分点
- **`SplitPoint`**: 切分点描述（stage_id, 是否需要 loss）
- **`pipeline()`**: 装饰器，将普通模块标记为 pipeline 模块
- **`PipeInfo`** (定义在 `_utils.py`): 存储 pipeline 结构信息（stage 数量、输入输出关系等）

### 3.4 手动切分前端工作流 (`stage.py`)

用户将模型手动切分为 `[submod_0, submod_1, ..., submod_N]`，然后：

```python
stages = [
    PipelineStage(
        submod,
        stage_index=i,
        num_stages=N,
        device=device,
        input_args=example_input,    # 可选，用于 STATIC 模式
        output_args=example_output,  # 可选
    )
    for i, submod in enumerate(submodules)
]
```

**`PipelineStage` 核心能力**:
- `_prepare_forward_infra()`: 推导前向元数据，分配 recv buffer，建立 P2P 通信链路
- `_prepare_backward_infra()`: 推导反向梯度元数据
- `_warmup_forward_vote()` / `_warmup_backward_result()`: P2P 投票协议，确定 STATIC/DYNAMIC 模式

## 4. 数据流：从 step() 调用到 backward

### 4.1 整体流程

```
schedule.step(x)
  ├── 1. split_args_kwargs_into_chunks()   # 将 batch 切分为 micro-batch
  ├── 2. _initialize_pp_stages()           # 首次执行：P2P 预热 + metadata inference
  ├── 3. [schedule 核心循环]               # 按调度策略交错执行 F/B/W
  │     ├── stage._forward()                #   recv → fwd → send
  │     ├── stage._backward_input()         #   recv grad → bwd_input
  │     └── stage._backward_weight()        #   bwd_weight → send grad
  ├── 4. merge_chunks()                     # 合并 micro-batch 结果
  └── 5. return loss
```

### 4.2 关键数据类

**`_TensorMeta` / `_DTensorMeta`** (`_utils.py`):
- `_TensorMeta`: 普通 tensor 的元数据 (shape, stride, dtype, requires_grad)
- `_DTensorMeta`: DTensor 的元数据（额外包含 placements, mesh 信息）
- 用于 recv buffer 预分配和运行时 shape/dtype 校验

**`_StageMeta`** (`_utils.py`):
- 聚合一个 stage 的全部元数据：`input_metas`, `output_metas`, `input_grads`, `output_grads`
- `is_complete_for_forward()`: 判断是否满足 STATIC 模式条件

**`PipeInfo`** (`_utils.py`):
- 描述 pipeline 的拓扑结构：stage 间的输入/输出对应关系
- 由 `_IR.py` 的 `Pipe` 创建，被 `_PipelineStage` 消费

**`InferenceMode`** (`_utils.py`):
```python
class InferenceMode(Enum):
    STATIC = "static"    # 跳过运行时元数据推导，直接使用静态配置
    DYNAMIC = "dynamic"  # 首次执行时动态推导 tensor shape/dtype
```
> 注意：这里的 "Inference" 是"元数据推断"之意，与 训练/推理（training/inference）无关。

### 4.3 P2P 通信抽象

**`_RecvInfo`** (`stage.py:93`):
- 描述一个输入 tensor 的来源信息（哪个上游 stage、buffer 位置、tensor_meta）

**P2P 操作**：
- `_send_tensor()` / `_recv_tensor()`: 封装 `dist.send` / `dist.recv`，支持 DTensor
- `_batch_p2p()`: 批量执行 P2P ops，用 `_batch_p2p` 的**同构快速路径**（相同 op 类型）和**混合路径**

## 5. 自定义 Backward 引擎 (`_backward.py`)

Pipelining **不使用** PyTorch 原生的 autograd backward，而是实现了自定义的 backward 拆分：

```
原生 backward:        loss.backward() → 一次性反向传播整个计算图
Pipeline backward:    拆分为 backward_input + backward_weight 两个阶段
```

核心 API：

```python
stage_backward(stage_outputs, output_grads, input_values, weights)
  → 返回 (grad_inputs, grad_weights)

# 也可分步调用：
stage_backward_input(stage_outputs_or_loss, output_grads, input_values, weights)
  → 返回 grad_inputs, param_groups

stage_backward_weight(weights, param_groups)
  → 填充 weight.grad
```

**关键设计**：
- `_null_coalesce_accumulate()`: 合并多个 micro-batch 的梯度累加
- `construct_reverse_graph()` + `reverse_closure()`: 从 loss 出发反向遍历 autograd graph，找出所需的中间节点
- `_get_grad_fn_or_grad_acc()`: 获取 tensor 的 grad_fn 或触发 AccumulateGrad 节点创建

## 6. Micro-batch 切分 (`microbatch.py`)

### 6.1 核心概念

将一个大 batch 切成多个 micro-batch，以流水线方式交错执行：

```python
batch (B=8) → [mb0(B=2), mb1(B=2), mb2(B=2), mb3(B=2)]
```

### 6.2 关键类与函数

**`TensorChunkSpec`**: 指定 tensor 沿哪个维度切分
```python
TensorChunkSpec(split_dim=0)   # 沿 batch 维度切分
TensorChunkSpec.from_tuple((0, 0, 1))  # 批量创建
```

**`_Replicate`**: 标记不切分、直接复制的输入（如模型参数、配置）

**`split_args_kwargs_into_chunks()`** (line 378): 主入口
- 输入：args, kwargs, chunks 数量, chunk spec
- 输出：`([args_chunk_0, ...], [kwargs_chunk_0, ...])`
- 支持嵌套结构（list/tuple/dict），使用 `pytree` 处理
- **DTensor 感知**：`_split_tensor()` 对 DTensor 走 `to_local() → tensor_split → from_local()` 路径，正确维护 global shape

**`merge_chunks()`** (line 493): 反向操作
- 将多个 micro-batch 的输出合并回一个大 batch
- 同样支持 DTensor（local cat → from_local with merged global shape）

## 7. 调度策略 (`schedules.py`)

### 7.1 类层次

```
_PipelineSchedule (ABC)                  # 公共基类：micro-batch 管理、loss 计算
├── PipelineScheduleSingle              # 单卡多 stage（用于调试/小规模）
│   ├── _ScheduleForwardOnly            # 纯推理（无 backward）
│   ├── ScheduleGPipe                   # GPipe：全部 F 完成后再全部 B
│   └── Schedule1F1B                    # 1F1B：交替执行 F 和 B
└── PipelineScheduleMulti               # 多卡多 stage（生产用）
    └── _PipelineScheduleRuntime        # 通用运行时调度框架
        ├── ScheduleLoopedBFS           # 循环 BFS 调度
        ├── ScheduleInterleaved1F1B     # 交错 1F1B（多 stage 每 rank）
        ├── ScheduleInterleavedZeroBubble  # 交错 Zero-Bubble
        ├── ScheduleZBVZeroBubble       # ZBV Zero-Bubble
        └── ScheduleDualPipeV           # DualPipeV 双向流水线
```

### 7.2 关键计算类型

```python
class _ComputationType(str, Enum):
    FORWARD = "F"            # 前向计算
    BACKWARD_INPUT = "I"     # 反向：计算 input grads（传给上游）
    BACKWARD_WEIGHT = "W"    # 反向：计算 weight grads（更新本地参数）
    UNSHARD = "UNSHARD"      # FSDP unshard
    RESHARD = "RESHARD"      # FSDP reshard
    SEND_F / RECV_F          # 前向通信
    SEND_B / RECV_B          # 反向通信
    FULL_BACKWARD = "B"      # 完整 backward（GPipe 用）
    OVERLAP_F_B              # 前向/反向重叠
    REDUCE_GRAD              # 梯度 reduce
```

### 7.3 调度执行模型

**`_PipelineScheduleRuntime`** 使用通用的 action 抽象：

```python
class _Action(NamedTuple):
    computation_type: _ComputationType
    microbatch_index: int
    stage_index: int
```

调度策略 = 生成一个 `_Action` 的序列/拓扑，然后逐项执行。这使得**调度策略与执行逻辑解耦**。

### 7.4 各调度策略特点

| 调度 | 特点 | 适用场景 |
|------|------|---------|
| **GPipe** | F 全部完成 → B 全部完成，peak memory 高 | 简单场景 |
| **1F1B** | 交替 F/B，控制 in-flight micro-batch 数量 | 平衡内存与效率 |
| **Interleaved1F1B** | 每 rank 多个 stage，交错调度 | 减少 bubble |
| **InterleavedZeroBubble** | 将 W 延迟到 bubble 区间执行 | 进一步减少 bubble |
| **ZBVZeroBubble** | 基于 Zero-Bubble 变体优化 | 最小化空闲时间 |
| **LoopedBFS** | BFS 执行顺序 | 特定拓扑 |
| **DualPipeV** | 双向流水线，支持前向/反向重叠 | 最高吞吐场景 |

### 7.5 _warmup_p2p 投票协议 (schedules.py:327)

在所有 stage 是 `PipelineStage` 类型时，通过一次前向/反向投票确定每个 stage 的 `InferenceMode`：

```
forward vote (rank0 → rank1 → ... → rankN):
    每个 stage 判断自己是否需要 DYNAMIC，将结果（AND 累积）发送给下游
    → 同时预热 2-rank sub-communicator

backward result (rankN → ... → rank1 → rank0):
    最终结果传回每个 stage，设置 _inference_mode
```

若任一 stage 需要 DYNAMIC，全局就是 DYNAMIC（全票通过才 STATIC）。

## 8. DTensor 集成

Pipelining 与 DTensor 深度集成，支持流水线并行 + 张量并行的混合并行：

- **`_split_tensor()`** / **`merge_chunks()`**: 正确处理 DTensor 的 local/global shape
- **`_DTensorMeta`**: 存储 DTensor 特有的 placements 和 mesh 信息
- **_forward_metadata_inference()**: DYNAMIC 模式下，通过 `DTensor.from_local(grad_placements=...)` 推导梯度 metadata
- **`GetMeshCallback`**: 回调接口，允许 DYNAMIC 模式下按需创建 DeviceMesh
- **`_MeshCache`**: 缓存 DeviceMesh 实例，避免重复创建
- **FSDP 集成**: `UnshardHandle` / `reshard` / `unshard` 在 1F1B warmup 和 cooldown 阶段使用

## 9. 典型使用流程

### 9.1 自动切分方式

```python
from torch.distributed.pipelining import pipeline, Pipe, pipe_split, Schedule1F1B

class MyModel(nn.Module):
    def forward(self, x):
        x = self.layer1(x)
        pipe_split()        # 标记切分点
        x = self.layer2(x)
        pipe_split()
        x = self.layer3(x)
        return x

mod = MyModel()
x = torch.randn(batch, ...)

# 自动切分 + 创建 schedule
pipe = Pipe(mod, chunks=num_microbatches)
schedule = Schedule1F1B(pipe, num_microbatches, loss_fn=loss_fn)

# 训练循环
for data in dataloader:
    loss = schedule.step(data)
    optimizer.step()
```

### 9.2 手动切分方式

```python
from torch.distributed.pipelining import PipelineStage, Schedule1F1B

submodules = [stage0_mod, stage1_mod, stage2_mod]
stages = [
    PipelineStage(sub, i, len(submodules), device)
    for i, sub in enumerate(submodules)
]
schedule = Schedule1F1B(stages, n_microbatches, loss_fn=my_loss)

loss = schedule.step(data)
```

## 10. 关键设计决策与注意事项

1. **不使用 `torch.autograd.backward()`**：为了将 backward 拆分为 I（计算 input grads）和 W（计算 weight grads），pipelining 实现了自己的 autograd graph 遍历，将 backward 过程打断，从而可以跨 micro-batch 交错执行。

2. **双前端但统一 runtime**：无论 _PipelineStage（自动）还是 PipelineStage（手动），都由同一套 schedule 驱动，核心接口一致（`_forward`, `_backward_input`, `_backward_weight`）。

3. **InferenceMode 不是训练/推理**：`InferenceMode.STATIC` vs `DYNAMIC` 决定运行时是否需要推导 tensor shape/dtype。命名容易与 train/eval 混淆。

4. **P2P 通信的两种路径**：`_batch_p2p` 对全 send 或全 recv 使用同构快速路径（2-rank sub-communicator）；混合 send+recv 走 group communicator 路径。

5. **v2.12 重大重构**：PR #177728 引入了 DTensor-aware metadata 体系（`_TensorMeta` / `_DTensorMeta` / `_StageMeta`），废弃了旧的 `PipeliningShapeError` → `PipeliningMetadataError`。

6. **Zero-Bubble 调度的 W 延迟**：在 Zero-Bubble 系列调度中，weight gradient 计算（W）被推迟到原本的 bubble 时间段执行，需要 `dw_builder` 生成 deferred weight update runner。
