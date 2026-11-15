# Kimi K3 7B 在 Ascend 950DT 上的 Zero-Bubble PP 与重计算实验报告

> 日期：2026-11-13 ～ 2026-11-14
> 平台：8 x Ascend 950DT，torchtitan 0.4.0 + torchtitan-ascend
> 目标：评估 InterleavedZeroBubble / ZBVZeroBubble / DualPipeV 与激活重计算的组合，寻找 32768 tokens x 8 microbatches 的可行配置

---

## 1. 结论摘要

| # | 结论 | 证据 |
|---|------|------|
| 1 | 零气泡（ZB）调度在当前 NPU 栈上可行，但显存上限远低于目标：InterleavedZeroBubble 无重计算时 2048 x 4 可稳定跑 10 步（55.5-56.7 GiB），4096 x 4 已到 87.6 GiB / 95.5% HBM 并在第 4 步崩溃 | 第 4.1、4.2 节 |
| 2 | 在 split backward 下用 `torch.utils.checkpoint` 做"可重放 AC"会**增加**显存而不是节省：同一 IZB 2048 x 4 从 55.5 GiB 涨到 79.1 GiB 并 OOM | 第 4.3、5.1 节 |
| 3 | ZBVZeroBubble / DualPipeV 只有在关闭 Kimi K3 的 attention-residual block cache（`attn_res_cache=False`）后才能启动；关闭后每次 hop 搬运整个 block stack。1024 x 4 实测 ZBV 43.2/72.4 GiB、185 tok/s/device，DualPipeV 43.4/72.1 GiB、489 tok/s/device | 第 4.4、5.3 节 |
| 4 | 32768 x 8（1,048,576 tokens/step）目前唯一验证可行的配置仍是 **1F1B + FullActivationRematConfig**：约 3.9k tok/s/device、86 TFLOPS/device、27.5% MFU、76-88 GiB reserved | 第 4.5 节 |
| 5 | "ZB 必须设置 `activation_remat=None`"这一结论需要修正：上游 torchtitan 的 GPU CI 明确把 split backward 与 `FullActivationRematConfig` 组合测试；我们在 NPU 上的失败用的是 `DefaultActivationRematConfig` + 17 层 debug 模型。IZB + Full 策略的 NPU 对照实验尚未完成 | 第 7 节 |

---

## 2. 实验环境

| 项目 | 配置 |
|------|------|
| 加速器 | 8 x Ascend 950DT，单卡 98304 MB（可用约 91.75 GiB） |
| CANN | 9.2.0-950 |
| PyTorch | 2.15.0+cpu |
| torch_npu | 2.15.0 |
| torchtitan | 0.4.0（release 版，含 `_torch_compat` 进程内 backport） |
| torchtitan-ascend | main 分支 4d7181b，editable 安装 |
| torch_remat | 0.2.2 |
| Python | 3.12（conda env torch215） |
| 自定义算子 | fla_npu（`ASCEND_CUSTOM_OPP_PATH`） |

**环境注意事项**

- 主机上所有容器共享同一组 `/dev/davinci0-7`，显存测量只有在 `npu-smi info` 没有外部进程时才可信。曾出现本进程已分配 46.94 GiB、剩余 35 MiB 即申请失败的情况，推断约 43 GiB 被其他容器占用；进程全部退出后仍可能有约 4.79 GiB 残留。
- 启动时出现 `Communication_Error_Bind_IP_Port EI0020 ... port 16666 already bound` 通常是另一个任务占用了 HCCL socket，不是代码问题。
- 服务器时区为 UTC，本地为 UTC+8，日志时间需要 +8 小时换算。

---

## 3. 模型、并行与复现入口

### 3.1 模型

Kimi K3 7.47B（`kimi_k3_7b_npu` recipe）：

- 13 层，hidden dim 4608
- 32 个 routed experts，top-2；2 个 shared experts
- `attn_res_block_size=4`（4 个 attention-residual block）
- full attention layers = {3, 7, 11, 12}
- q_lora 896，kv_lora 288，dense_hidden 19456
- 含 vision encoder

debug 模型为 17 层版本（`kimi_k3_debugmodel_npu`），用于小规模功能验证。

### 3.2 并行配置

8 卡：DP-replicate 1 / DP-shard 4 / CP 1 / TP 1 / EP 4 / PP 2。

- 32 experts / EP 4 = 每 rank 8 experts
- InterleavedZeroBubble 每 rank 2 个 virtual stage，共 4 个
- PP=2 下 ZBVZeroBubble / DualPipeV 同样得到 4 个 virtual stage

2 机 16 卡（`kimi_k3_7b_npu_1f1b_ac_2node`）：DP-shard 8 / EP 4 / PP 2，world_size 16；EP 从 dense dp_shard 区域切出，edp_shard = 8/4 = 2，每 rank 4 experts。

### 3.3 复现入口

配置文件：服务器上的 `kimi_k3_pp_config.py`（通过 `PYTHONPATH` 被 `torchtitan_ascend.train --module kimi_k3_pp_config --config <factory>` 导入）。

启动脚本：`run_kimi_k3_7b_pp.sh`，环境变量：

| 变量 | 含义 | 默认 |
|------|------|------|
| `CONFIG` | factory 名 | `kimi_k3_7b_npu_interleaved_zerobubble` |
| `K3_TOKENS` | 每 microbatch 每 DP rank 的 token 槽位 | 依 factory |
| `K3_MBS` | 每 step 的 pipeline microbatch 数 | 依 factory |
| `K3_STEPS` | 训练步数 | 10 |
| `K3_LOCAL_COMPILE` | 是否启用 local compile（NPU 上默认 off） | off |
| `K3_SCHEDULE` | V 形 factory 的调度选择 | `ZBVZeroBubble` |
| `K3_MEM_PROBE` | 每步打印显存探针 | off |
| `OUT` | 输出目录 | - |
| `LOG_RANK` | torchrun 打印哪些 rank | `0,4` |
| `NGPU` | 每机卡数 | 8 |

配置模块中的 factory：

| factory | 调度 | 重计算 | 用途 |
|---------|------|--------|------|
| `kimi_k3_debugmodel_npu_interleaved_zerobubble` | IZB | None | 17 层 debug 模型功能验证，8192 x 4 |
| `kimi_k3_7b_npu_interleaved_zerobubble` | IZB | None | 7B 基线，token/microbatch 可调 |
| `kimi_k3_7b_npu_1f1b_ac` | 1F1B | FullActivationRematConfig | 32k x 8 目标配置 |
| `kimi_k3_7b_npu_izb_replay_ac` | IZB | ReplayableBlockCheckpoint | 可重放 AC 对照实验 |
| `kimi_k3_7b_npu_vshape_replay_ac` | ZBV / DualPipeV | ReplayableBlockCheckpoint | V 形调度实验 |
| `kimi_k3_7b_npu_1f1b_ac_2node` | 1F1B | FullActivationRematConfig | 2 机 16 卡验证 |
| `kimi_k3_7b_npu_izb_full_ac` | IZB | FullActivationRematConfig | 与上游策略对齐的对照实验（待跑） |

模块级设置：

```python
# split backward 通过 torch.autograd.grad(retain_graph=True) 反传两次，
# AOTAutograd 不允许在 retain_graph 下 donate buffer。
torch._functorch.config.donated_buffer = False
```

NPU 上所有 factory 都设置：

```python
config.model.local_compile_regions = []
```

### 3.4 关键补丁（均在配置模块内，未修改 torchtitan 核心代码）

| 补丁 | 作用 |
|------|------|
| `ReplayableBlockCheckpoint` | 用非重入 `torch.utils.checkpoint` 包裹每个 transformer block，使 input pass 与 weight pass 各自重放 |
| `_force_whole_stack_transport()` | monkeypatch `KimiK3Model.pipeline`，以 `attn_res_cache=False` 调用 `pipeline_kimi_k3`，让 V 形调度可以启动 |
| `_install_memory_probe()` | 每步打印 allocated / reserved / max 显存 |
| `torch._functorch.config.donated_buffer = False` | 见上 |
| `local_compile_regions = []` | 规避 Ascend inductor/triton 崩溃 |

---

## 4. 实测结果

所有数字均为 8 x Ascend 950DT、10 个训练步、无外部进程干扰时的测量；显存为单卡峰值。

### 4.1 InterleavedZeroBubble（无重计算）

| 配置 | 结果 |
|------|------|
| 2048 x 4 | 10/10 步通过；55.5-56.7 GiB；2,550 tok/s/device；56 TFLOPS；18% MFU；loss 8.069 -> 8.067；grad_norm 14.4 -> 14.0 |
| 4096 x 4（2026-11-13） | 前 3 步通过，87.6 GiB / 95.5% HBM；第 4 步在 `npu_chunk_kda_bwd` 中崩溃：`aoti_torch_empty_strided ... acl_meta.h line 469` |

### 4.2 InterleavedZeroBubble + `ReplayableBlockCheckpoint`

| 配置 | 结果 |
|------|------|
| 4096 x 4 | 第 1 步 OOM，85.46 GiB，崩溃点在 `backward_weight_one_chunk` 的 checkpoint unpack hook 重算 |
| 2048 x 4 | 第 1 步 OOM，79.10 GiB，同一位置 |
| 1024 x 4 | 10/10 步通过，63.5-72.6 GiB |

> 首次 1024 x 4 尝试在 46.94 GiB allocated / 35 MiB free 处失败，原因是其他容器的任务占用了约 43 GiB；换到空闲设备后通过。

**对照：同一 2048 x 4 大小，无重计算 55.5 GiB，加重计算 79.1 GiB。可重放 AC 在 split backward 下是显存负优化。**

### 4.3 ZBVZeroBubble / DualPipeV

两者都是 V 形调度，`pipeline_kimi_k3` 在 `attn_res_cache=True` 时会直接拒绝（`_require_loop_style`）。用 `_force_whole_stack_transport()` 关闭 block cache 后，两个 factory 均使用 `ReplayableBlockCheckpoint`（以下显存数字包含它的开销）：

| 调度（1024 x 4，10 步） | rank 0 峰值 | rank 4 峰值 | 吞吐 |
|--------------------------|-------------|-------------|------|
| ZBVZeroBubble | 43.2 GiB | 72.4 GiB | 185 tok/s/device |
| DualPipeV | 43.4 GiB | 72.1 GiB | 489 tok/s/device |

- rank 4 承担 stages [1, 2]，即 13 层中的 8 层，负载高度倾斜。
- DualPipeV 的吞吐明显更高，因为同一 rank 上的 V 对可以同时进行两个方向的通信/计算重叠。
- 但两者都达不到 32768 x 8：零气泡的驻留机制与 IZB 相同，关闭 cache 还会增加每次 hop 的传输量。

### 4.4 1F1B + FullActivationRematConfig（32k x 8 目标配置）

| 指标 | 数值 |
|------|------|
| tokens/step | 1,048,576（32768 x 8 x 4 DP ranks） |
| 吞吐 | 约 3.9k tok/s/device |
| 算力 | 86 TFLOPS/device |
| MFU | 27.5% |
| 显存 | 76-88 GiB reserved |
| 步数 | 10 步稳定 |

不加 AC 的 1F1B 在 32768 tokens in-flight 时即 OOM。

### 4.5 2 机 16 卡验证（进行中）

`kimi_k3_7b_npu_1f1b_ac_2node`（DP-shard 8 / EP 4 / PP 2，16 卡，1F1B + FullActivationRematConfig）已于 2026-11-14 启动，报告撰写时仍处于 inductor benchmark / 编译阶段，尚无 step 结果。

---

## 5. 机制分析

### 5.1 为什么可重放 AC 在 split backward 下反而更费显存

零气泡调度把每个 microbatch 的反向拆成两遍：

1. input pass：`stage_backward_input` 计算对输入的梯度；
2. weight pass：`stage_backward_weight` 计算对权重的梯度。

`stage_backward_weight` 内部调用 `torch.autograd.grad(..., retain_graph=True)`，而 stage 会持有 forward 输出直到 weight pass 结束。非重入 `torch.utils.checkpoint` 在 input pass 中重放 block forward 时，会把整块的重算图 graft 到外层图中；由于 `retain_graph=True`，这张图（即 block 的完整重算激活）必须一直驻留到 weight pass，**叠加在 checkpoint 自己保存的 inputs/outputs 之上**。因此：

- 不加重计算：forward 保存的激活在 weight pass 中按需释放；
- 加重计算：除了 checkpoint 的输入/输出，还要多保留一整份重算图。

实测差距（2048 x 4：55.5 GiB -> 79.1 GiB）与此一致。

> 该结论针对 `torch.utils.checkpoint`。torch_remat 0.2.2 对 `retain_graph` 有专门的 re-recompute 设计，行为不同，见第 7 节。

### 5.2 torch_remat 的 saved input 单次消费

2026-11-12 在 17 层 debug 模型、PP=2、IZB、4 microbatches、`DefaultActivationRematConfig` 下，weight pass 报：

```
RuntimeError: No saved input saved_input.0 for layers.9::attention.q_proj.linear.
Either it was already unpacked once in this backward (read ctx.saved_tensors only once),
or forward and recompute followed different code paths.
```

- 崩溃发生在 `stage_backward_weight` -> `torch.autograd.grad` -> torch_remat `_load_saved_input`。
- 当时 torch_remat 0.2.2、`_torch_compat` backport [199884, 200238, 200199] 均已加载，因此不是版本缺失。
- 错误信息本身给出两种可能：同一 backward 中重复读取 saved tensor，或 forward 与 recompute 走了不同代码路径。K3 的 attention-residual cache / 区域策略都可能影响后者。

**该失败只证明"DefaultActivationRematConfig + K3 + 当前 NPU 栈"不可用，不能推广为"torch_remat 无法用于 split backward"**（见第 7 节）。

### 5.3 V 形调度与 attention-residual cache

`pipeline_kimi_k3` 的 block cache 假设 loop 式 stage 分配（stage s 位于 rank s mod pp）。ZBVZeroBubble / DualPipeV 是 V 形分配（rank i 持有 stage i 与 stage n-1-i），缓存复用会失效，因此模型直接拒绝：

```
ValueError: ... attn_res_cache is not supported with V-shaped placement ...
```

关闭 cache 后每次 hop 需要搬运整个 block stack（最多 4 个 block 的 `[T, D]`），这是 V 形调度的额外代价。

### 5.4 吞吐差异

- DualPipeV 在 1024 x 4 下达到 489 tok/s/device，是 ZBVZeroBubble（185）的 2.6 倍，原因是 V 对的两个方向可以在同一 rank 上重叠。
- 但 V 形调度在 PP=2 下与 IZB 一样是 4 个 virtual stage，没有额外收益；负载倾斜（rank 4 承担 8/13 层）也放大了最慢 rank 的瓶颈。

---

## 6. 调度 x 重计算支持矩阵（NPU 实测）

| 调度 | backward 形态 | torch_remat | 可重放 AC | 额外补丁 | NPU 实测 |
|------|---------------|-------------|-----------|----------|----------|
| 1F1B | 单次完整 backward | 支持 | 不需要 | 无 | 32k x 8 通过 |
| Interleaved1F1B | 单次完整 backward | 支持 | 不需要 | 无 | 未单独测 |
| InterleavedZeroBubble | input/weight 两遍 | Default 策略失败 | 可用但费显存 | 无（loop 式，cache 可用） | 2048 x 4 通过（无 AC） |
| ZBVZeroBubble | input/weight 两遍 | 未验证 | 可用但费显存 | `attn_res_cache=False` | 1024 x 4 通过 |
| DualPipeV | input/weight 两遍 | 未验证 | 可用但费显存 | `attn_res_cache=False` | 1024 x 4 通过 |

通用前提：`donated_buffer=False`、`local_compile_regions=[]`（NPU）。

---

## 7. GPU / 上游社区对照

### 7.1 上游明确支持 ZB + 重计算的组合

torchtitan 0.4.0 的 GPU 集成测试把零气泡和 `FullActivationRematConfig` 直接组合：

- `tests/integration_tests/features.py:415-424`：`llama3_debugmodel_pp4_zero_bubble`，InterleavedZeroBubble + `FullActivationRematConfig()`，4 GPU。
- `tests/integration_tests/features.py:427-436`：`llama3_debugmodel_pp2_zbv`，ZBVZeroBubble + `FullActivationRematConfig()`，2 GPU。
- `features.py:940-951`：注册为 `pp_looped_zero_bubble` / `pp_zbv`，`use_real_pg=True`。
- `.github/workflows/integration_test.yaml:129-155`：real-PG job 在 8 GPU 上运行 features 套件。
- `tests/unit_tests/cpu/test_integration_test_definitions.py:134-145`：断言这些测试未被禁用，且要求 varlen CUDA graph。

框架层还有两处专门支持：

- torch_remat 0.2.2 文档字符串明确处理 `retain_graph` 下的二次反传：
  - `_api.py:1523`："Free each output slot as it is served, except under retain_graph where a later backward reads the slot again"
  - `_api.py:1623`："... clearing it on load keeps it from staying resident under retain_graph, and the next replay refills it"
  - `_region.py:67`："the state resyncs at every SAVE boundary and a retain_graph re-recompute ..."
- torchtitan 进程内 backport PyTorch PR #200199（`torchtitan/_torch_compat/_pipelining_edits.py:549-587`），注释明确说明 activation-checkpoint boundary 的输出视图在 split backward 下要保留到 weight pass 之后。NPU 日志确认该 backport 已加载：

```
Applied TorchTitan process-local compatibility backports [199884, 200238, 200199] to torch 2.15.0+cpu
```

### 7.2 各问题的归属

| 问题 | GPU 上是否存在 | 归属 |
|------|----------------|------|
| ZB + `FullActivationRematConfig` | 否，是上游 CI 组合 | 上游支持 |
| ZB + `DefaultActivationRematConfig`（本机实测） | 未验证 | 上游未覆盖，不保证 |
| V 形 + `attn_res_cache` | 是 | 模型层限制（Kimi K3），GPU 同样拒绝 |
| `attn_res_cache=False` 的 monkeypatch | 是 | 模型没有把开关暴露到 config，GPU 同样需要 |
| `ReplayableBlockCheckpoint` | 行为一致 | 自研备选方案，上游无对应物 |
| `donated_buffer=False` | 是（仅开 compile 时） | PyTorch AOTAutograd 约束 |
| `local_compile_regions=[]` | 否 | **NPU 专属**：Ascend inductor/triton 崩溃 |
| HCCL 16666 / 共享卡 | 否 | 环境专属 |

### 7.3 结论修正

此前记录的"torch_remat 不能与零气泡一起使用"过度概括。准确表述应为：

1. 上游 torchtitan 在 GPU 上支持 split backward + `FullActivationRematConfig`，有 CI 覆盖；
2. 我们在 NPU 上的失败发生在 `DefaultActivationRematConfig` + 17 层 debug 模型，可能与该策略的 save/recompute 区域划分或 K3 的重放路径有关；
3. `FullActivationRematConfig` + IZB 在 NPU 上尚未验证，已加入 factory `kimi_k3_7b_npu_izb_full_ac`，待 2 机任务结束后运行。

如果该对照通过，则：

- IZB/ZBV/DualPipeV 下不再需要 `ReplayableBlockCheckpoint`；
- 需要重新测量 ZB + torch_remat 的显存上限，"ZB 必须 `activation_remat=None`"的结论作废；
- 但仍需注意：即使功能可用，零气泡在 weight pass 前保留 in-flight microbatch 的机制不变，32768 x 8 的显存可行性仍需实测（按当前数据估算，ZB 驻留约为每 token-in-flight 每 rank 2.6 MiB 激活、每 stage 约 4 MiB，32768 x 8 约为预算的 15 倍）。

---

## 8. 结论与建议

1. **32768 x 8 目标**：继续使用 `kimi_k3_7b_npu_1f1b_ac`（1F1B + `FullActivationRematConfig`），这是目前唯一验证过的配置。
2. **零气泡定位**：IZB 适合 2048 x 4 级别的显存预算（约 56 GiB），不要用于 32k x 8；4096 x 4 会先撞显存上限，随后可能触发 `npu_chunk_kda_bwd` 崩溃。
3. **不要使用可重放 AC 省显存**：`torch.utils.checkpoint` 在 split backward 下会保留整块重算图，实测负优化；保留该 factory 仅用于复现对照。
4. **V 形调度**：ZBV/DualPipeV 在 PP=2 下没有 virtual stage 收益，且需要关闭 block cache、增加传输、负载倾斜；如果追求吞吐，DualPipeV 在 1024 x 4 下有 2.6 倍优势，但需接受 8/13 层集中在 rank 4 的显存与计算倾斜。
5. **NPU 专属 workaround**：`local_compile_regions=[]` 必须保留；`donated_buffer=False` 在启用 compile 的 split backward 场景下需要保留。
6. **重计算策略选择**：在 1F1B 上优先 `FullActivationRematConfig`；零气泡上先验证 `FullActivationRematConfig`，再决定是否继续使用 `ReplayableBlockCheckpoint`。

---

## 9. 待办

- [ ] 运行 `kimi_k3_7b_npu_izb_full_ac`（建议 1024 x 4、3-10 步），确认 IZB + `FullActivationRematConfig` 在 NPU 上是否可用。
- [ ] 若通过，重新测量 IZB / ZBV / DualPipeV + torch_remat 的显存上限，更新第 6 节矩阵。
- [ ] 收集 2 机 16 卡 `kimi_k3_7b_npu_1f1b_ac_2node` 的 step 结果（显存、吞吐、loss）。
- [ ] 评估 PP=4 / 双机场景下 ZB + Full AC 相对 1F1B 的收益（需要先解决 V 形 cache 与负载均衡问题）。
- [ ] 确认 `npu_chunk_kda_bwd` 在 4096 x 4 下的失败是显存边界问题还是算子问题（AOE/acl_meta 报错）。

---

## 10. 附录

### 10.1 关键错误签名

| 错误 | 含义 |
|------|------|
| `No saved input saved_input.N for <op>. Either it was already unpacked once ...` | torch_remat saved input 在 weight pass 中已被消费或重放路径不一致 |
| `Communication_Error_Bind_IP_Port EI0020 ... port 16666 already bound` | 另一个任务占用 HCCL socket |
| `aoti_torch_empty_strided ... acl_meta.h line 469`（`npu_chunk_kda_bwd`） | Ascend 算子在显存边界附近分配失败 |
| `ValueError ... attn_res_cache ... V-shaped placement` | Kimi K3 的 block cache 不支持 ZBV/DualPipeV |
| AOTAutograd buffer donation 与 `retain_graph` 冲突 | 开启 local compile + split backward 时触发；本项目全局关闭 `donated_buffer` 规避 |

### 10.2 日志与输出目录

- 训练输出：`outputs_ascend/<run_name>/`，包含 `structured_logs/`、`torchrun_logs/`、`comm_traces/`
- 每 rank 的 stdout/stderr：`torchrun_logs/<rendezvous>/attempt_0/<rank>/stdout.log|stderr.log`
- 显存探针：设置 `K3_MEM_PROBE=on`，每步打印 `[memprobe] after step ...`
- 吞吐/算力：structured log 中的 `tflops`、`tokens_per_second` 字段

### 10.3 复现命令示例

```bash
# 1F1B + Full AC，32k x 8 目标配置
OUT=/path/to/outputs/k3_1f1b_ac \
CONFIG=kimi_k3_7b_npu_1f1b_ac \
K3_TOKENS=32768 K3_MBS=8 K3_STEPS=10 \
bash run_kimi_k3_7b_pp.sh

# IZB 基线（无重计算）
OUT=/path/to/outputs/k3_izb_2048x4 \
CONFIG=kimi_k3_7b_npu_interleaved_zerobubble \
K3_TOKENS=2048 K3_MBS=4 K3_STEPS=10 \
bash run_kimi_k3_7b_pp.sh

# IZB + Full AC（待验证）
OUT=/path/to/outputs/k3_izb_full_ac \
CONFIG=kimi_k3_7b_npu_izb_full_ac \
K3_TOKENS=1024 K3_MBS=4 K3_STEPS=10 \
bash run_kimi_k3_7b_pp.sh

# V 形调度（二选一）
OUT=/path/to/outputs/k3_dualpipev \
CONFIG=kimi_k3_7b_npu_vshape_replay_ac \
K3_SCHEDULE=DualPipeV K3_TOKENS=1024 K3_MBS=4 K3_STEPS=10 \
bash run_kimi_k3_7b_pp.sh
```

### 10.4 术语

| 术语 | 说明 |
|------|------|
| 1F1B | one-forward-one-backward，单次完整 backward |
| IZB | InterleavedZeroBubble，loop 式零气泡，每 rank 2 个 virtual stage |
| ZBV | ZBVZeroBubble，V 形零气泡 |
| DualPipeV | V 形双方向零气泡，同一 rank 的 V 对可双向重叠 |
| split backward | 反向拆成 input pass 与 weight pass 两遍 |
| virtual stage | 一个 PP rank 上的逻辑 stage |
| MFU | model FLOPs utilization |
| `attn_res_cache` | Kimi K3 的 attention-residual block 缓存，loop 式调度下复用 block 输出 |
