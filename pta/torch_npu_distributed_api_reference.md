# PTA `torch_npu/distributed/` API 参考文档

> 基于 PTA v2.12.0 版本，统计 `torch_npu/distributed/` 下所有分布式 API

---

## 1. `torch_npu.distributed` 公开入口（`__init__.py`）

| API | 类型 | 功能说明 |
|-----|------|----------|
| `is_available()` | 函数 | 判断分布式模块是否可用，检查 `_c10d_npu_init` 属性 |
| `is_hccl_available()` | 函数 | 判断 HCCL 通信后端是否在 Backend 列表中注册 |
| `reinit_process_group()` | 函数 | 重新初始化 HCCL 通信组，支持重建链接或仅恢复通信 |
| `reduce_scatter_tensor_uneven()` | 函数 | 非均匀张量的 reduce_scatter 操作（HCCL 自定义） |
| `all_gather_into_tensor_uneven()` | 函数 | 非均匀张量的 all_gather 操作（HCCL 自定义） |
| `ParallelStore` | 类 | 并行 TCP 存储后端，用于分布式 rendezvous |

---

## 2. 分布式通信 `distributed_c10d.py`

### 公开 API

| API | 功能说明 |
|-----|----------|
| `is_hccl_available()` | 检查 HCCL 后端是否可用（`"hccl" in Backend.backend_list`） |
| `reinit_process_group(group, rebuild_link)` | 重新初始化 ProcessGroup，支持 `rebuild_link=False`（仅恢复通信，不重建链接）和 `rebuild_link=True`（完全重建 HCCL 通信） |

### 内部 API（`_` 前缀，被集成到 `torch.distributed` 中）

| API | 功能说明 |
|-----|----------|
| `_batch_isend_irecv(p2p_op_list)` | NPU 上的批量 P2P 收发操作，对 NPU 设备使用 HCCL batch_isend_irecv 或 coalescing manager 合并执行 |
| `_gather(tensor, gather_list, dst, group, async_op)` | NPU 上的 gather 操作，默认使用 all_gather 替代实现（HCCL 不原生支持 gather），兼容模式下使用 send/recv |
| `_gather_object(obj, object_gather_list, dst, group)` | NPU 上的 Python 对象 gather 操作，内部使用 `_gather` 实现，传输前序列化、传输后反序列化 |
| `_clear_pg_cache_in_torch(group)` | 清除指定 ProcessGroup 在 PyTorch 内部全局状态中的缓存（pg_map, pg_names, tags 等） |
| `_comm_switch_nic(ranks, useBackup)` | 切换通信网卡，控制 NPU 通信使用主/备网卡链路 |
| `_reduce_scatter_tensor_uneven(output, input, input_split_sizes, op, group, async_op)` | 非均匀尺寸的 reduce_scatter 操作，支持指定每个 rank 的输入 split 大小 |
| `_all_gather_into_tensor_uneven(output, input, output_split_sizes, group, async_op)` | 非均匀尺寸的 all_gather 操作，支持指定每个 rank 的输出 split 大小 |
| `_trigger__get_addr_and_port_decorator(func)` | 装饰器：当使用 parallel rendezvous 后端时，临时将 backend 改为 static 获取地址端口后再恢复 |
| `_trigger_rendezvous_decorator(func)` | 装饰器：当 `TORCH_NPU_USE_PARALLEL_TCPSTORE=True` 时，将 `env://` URL 改写为 `parallel://` 格式 |
| `_destructor_process_group()` | NPU 退出时清理所有 ProcessGroup，调用 shutdown 并清空全局状态 |
| `_hccl_get_sequence_number_for_group(self)` | 获取 HCCL ProcessGroup 的序列号（用于调试和排序通信操作） |
| `_patched_new_process_group_helper(...)` | PTA 补丁版 ProcessGroup 创建函数，替换 `torch.distributed.distributed_c10d._new_process_group_helper`，增加 HCCL 后端的 NPU 设备绑定、split_from 支持 |
| `_hccl_add_ephemeral_timeout_for_all_pgs(timeout)` | 临时为所有 HCCL ProcessGroup 设置额外的超时时间（ephemeral timeout），第一个集合通信完成后自动重置 |

---

## 3. 分布式 Rendezvous `rendezvous.py`

| API | 功能说明 |
|-----|----------|
| `_parallel_rendezvous_handler(url, timeout)` | 并行 TCP 的 rendezvous handler：解析 `parallel://host:port?rank=X&world_size=Y` URL，创建 ParallelStore |
| `_create_parallel_handler(params)` | 从 RendezvousParameters 创建 `_ParallelTCPRendezvous` 实例 |
| `_create_c10d_store(hostname, port, rank, world_size, timeout)` | 创建 c10d Store（支持 agent store 复用、分层优化），使用 ParallelStore 作为底层存储 |
| `_rendezvous_init()` | 注册 "parallel" rendezvous backend handler |

### 类

| API | 功能说明 |
|-----|----------|
| `_ParallelTCPRendezvous` | 并行 TCP Rendezvous Handler，封装 ParallelStore 的创建和管理，支持 `get_backend()`, `next_rendezvous()`, `is_closed()`, `shutdown()` |

---

## 4. 分布式启动 `run.py`

| API | 功能说明 |
|-----|----------|
| `parse_args(args)` | 解析命令行参数，在 PyTorch 原生参数基础上增加 `--enable_tiered_parallel_tcpstore` 参数（分层并行 TCP Store） |
| `_main(args)` | NPU 分布式训练入口，设置 `rdzv_backend='parallel'` 后调用 `torch.distributed.run.run()` |

---

## 5. 对称内存 `_symmetric_memory/__init__.py`

| API | 功能说明 |
|-----|----------|
| `_enable_symm_mem_for_group(group_name)` | 为指定 ProcessGroup 启用对称内存（symmetric memory）功能，设置 `_SymmetricMemory.set_group_info` |

---

## 6. FSDP 补丁 `fsdp/_add_fsdp_patch.py`

### 公开 API

| API | 功能说明 |
|-----|----------|
| `fully_shard(*args, **kwargs)` | NPU 版 FSDP fully_shard 入口，内部先应用 FSDP 增强补丁，再调用 PyTorch 原生的 `torch_fully_shard` |
| `move_attr(src_obj, src_attr, dst_obj, dst_attr)` | 对象属性迁移工具函数（将属性值从源对象移到目标对象并清空源） |

### 内部 API

| API | 功能说明 |
|-----|----------|
| `_patched_finalize_backward(self)` | 补丁版 FSDP backward 完成处理，增加 grad_offload_event 同步等待 |
| `_get_param_all_gather_inputs(fsdp_params)` | 获取 FSDP 参数的 all_gather 输入张量，NPU 上使用 `non_blocking=True` |
| `_patched_root_pre_forward(self, module, args, kwargs)` | 补丁版 FSDP root 前向传播，增加 `"npu"` 设备类型支持 |
| `_patched_fsdp_param_group_init` | 装饰器工厂：为 FSDP 参数组设置 `_use_mem_cache` 标志 |
| `_custom_all_gather_copy_in(all_gather_inputs, ...)` | 自定义 all_gather copy_in 操作，使用 FSDP 内存缓存（`_fsdp_mem_cache`）分配输出 |
| `_patched_foreach_all_gather(fsdp_params, ...)` | 补丁版批量 all_gather，NPU 上支持内存缓存复用 |
| `_patched_wait_all_gather_streams_on_event` | 补丁版等待 all_gather 流，增加内存缓存释放逻辑 |
| `_patched_foreach_reduce(fsdp_params, ...)` | 补丁版批量 reduce scatter，NPU 上支持内存缓存复用和自定义 shard_dim 处理 |
| `_patched_post_forward(original_post_forward)` | 补丁版 FSDP post_forward 装饰器，支持 `_skip_post_forward` 优化 |
| `_patched_post_backward(original_post_backward)` | 补丁版 FSDP post_backward 装饰器，重置 `_skip_post_forward` |
| `_patched_fsdp_state_post_forward(original_post_forward)` | 补丁版 FSDP 状态 post_forward，管理前向预取状态备份/恢复 |
| `_apply_fsdp_patch()` | 应用基础 FSDP 补丁（finalize_backward、all_gather_inputs、root_pre_forward） |
| `_apply_fsdp_enhance_patch()` | 应用增强 FSDP 补丁（内存缓存、通信优化、冗余 all-gather 消除） |

### 类

| API | 功能说明 |
|-----|----------|
| `FSDPMemCache` | FSDP 内存缓存管理器，支持缓冲区按 dtype 管理和重用（allocate/free/clear） |

---

## 7. FSDP 集合通信 `fsdp/_fsdp_collectives.py`

| API | 功能说明 |
|-----|----------|
| `chunk_cat(tensors, dim, num_chunks, out)` | PrivateUse1 设备上的 chunk_cat 算子实现 |
| `all_gather_copy_in_npu(all_gather_inputs, inp_split_sizes, ...)` | PrivateUse1 设备上的 FSDP all_gather copy_in 算子实现 |
| `split_with_sizes_copy(all_gather_output, all_gather_input_split_sizes, dim, out)` | PrivateUse1 设备上的 FSDP split_with_sizes_copy 算子实现 |

---

## 8. Pipeline 并行 `pipelining/_pipelining_fx.py`

### 公开 API

| API | 功能说明 |
|-----|----------|
| `populate_boundary_metadata(traced_gm, example_args, ...)` | 通过 eager warmup 执行，为 FX 图中每个节点填充 `meta['val']` 元数据 |
| `symbolic_trace_for_pipelining(module, example_args, ...)` | 使用流水线专用 FX tracer 对模块进行符号追踪，支持 `atomic_units`（原子单元模块不展开） |
| `inject_pipeline_splits(traced, split_spec, pipe_split_op)` | 在 FX 图中插入流水线分割点（`pipe_split_op`），支持 `BEGINNING` 和 `END` 两种切分位置 |

### 内部 API

| API | 功能说明 |
|-----|----------|
| `_find_tp_group_ranks_from_model(model)` | 从模型参数中推断 Tensor Parallel 的 rank 列表 |
| `_move_to_device(value, device)` | 递归将嵌套结构中的张量移动到目标设备 |
| `_tree_to_meta(value)` | 将运行时值树转换为 meta 张量树 |
| `_module_has_meta_tensors(module)` | 检查模块是否包含 meta 设备参数或 buffer |
| `_materialize_meta_module_(module, device)` | 将 meta 参数初始化到实际设备并用随机值填充 |
| `_pack_meta_tree(value)` | 将 meta 张量打包为可序列化的 Python 数据 |
| `_unpack_meta_tree(value)` | 将序列化元数据解包恢复为 meta 张量 |
| `_pack_graph_meta(graph_module)` | 打包 FX 图中所有节点的 `meta['val']` |
| `_apply_packed_meta(graph_module, packed)` | 将打包的元数据应用到对应节点 |
| `_warmup_device()` | 选择 warmup 使用的设备（NPU > CUDA > CPU） |
| `_seed_placeholder_meta_from_examples(graph_module, example_args, ...)` | 用真实输入为 placeholder 节点填充 meta 信息 |
| `_contains_tensor(value)` | 检查嵌套值是否包含 Tensor |
| `_validate_atomic_units(module, atomic_units)` | 验证原子单元名称对应的子模块存在 |
| `_mark_atomic_units(module, atomic_units)` | 标记原子单元模块（设置 `_torch_npu_pipelining_atomic_unit` 属性） |
| `_clear_atomic_units(module, atomic_units)` | 清除原子单元标记 |
| `_concrete_args_from_examples(module, example_args, example_kwargs)` | 从示例输入中提取 FX 非张量参数（concrete_args） |

### 类

| API | 功能说明 |
|-----|----------|
| `RealPyTreeInfo` | PyTree 结构元数据 dataclass（`in_spec`, `out_spec`） |
| `WarmupMetaResult` | Eager warmup 结果 dataclass（`real_output`, `meta_output`） |
| `_MetadataInterpreter` | FX 解释器子类，运行图并为每个节点附加 meta 张量值 |
| `_FXPipeliningTracer` | 流水线专用 FX tracer，对于 `torch.nn` 模块、Sequential 模块和原子单元模块有特殊 leaf 策略 |

---

## 9. Pipeline 并行入口 `pipelining/_pipelining_patch.py`

### 公开 API

| API | 功能说明 |
|-----|----------|
| `pipeline(module, mb_args, ...)` | 流水线并行入口函数，将模块拆分为流水线阶段。支持 `mode="export"`（原生导出）和 `mode="fx"`（FX 追踪），返回 Pipe IR 对象 |

### 内部 API

| API | 功能说明 |
|-----|----------|
| `_preserve_fx_call_modules_for_from_traced(ir_module)` | 上下文管理器：在 Pipe 构建期间保留 FX `call_module` 子模块 |
| `_from_fx_traced(ir_module, module, traced, split_policy)` | 从 FX 追踪的 GraphModule 构建原生 PyTorch Pipe |
| `_prune_none_root_placeholders(pipe)` | 删除值为 None 的 placeholder 节点 |
| `_quiet_modify_graph_op_device(gm, new_device)` | 递归更新 FX 图和子模块中的设备参数 |
| `_apply_quiet_graph_device_patch(ir_module)` | 打补丁替换 `_modify_graph_op_device` 为静默版本 |
| `_pipeline_fx(ir_module, module, ...)` | FX 模式的流水线构建核心：追踪 → 填充元数据 → 注入分割点 → 构建 Pipe |
| `_init_pipelining_backend()` | 延迟初始化流水线后端，导入 `torch.distributed.pipelining._IR` 并打补丁 |

### 类

| API | 功能说明 |
|-----|----------|
| `_FXExportedProgramShim` | FX ExportedProgram 的适配器，提供 Pipe 构建所需的 `module()` 接口 |

---

## 10. RPC 后端注册 `rpc/backend_registry.py`

| API | 功能说明 |
|-----|----------|
| `_get_device_count_info()` | 获取 NPU 等第三方设备的 count 信息 |
| `_init_device_state(custom_backend_name)` | 初始化第三方设备状态 |
| `_tensorpipe_validate_devices(devices, device_count)` | 验证设备列表格式（CPU 或有效索引的 NPU 设备） |
| `_validate_device_maps(all_names, all_device_counts, ...)` | 验证多节点的设备映射关系（1-to-1 映射、source/target 设备合法性） |
| `_get_device_infos()` | 从 RPC agent 获取设备信息（device_count, device_maps, devices） |
| `_tensorpipe_exchange_and_check_all_device_maps(my_name, ...)` | 通过 all_gather_object 交换并校验所有 worker 的设备映射 |
| `_set_devices_and_reverse_device_map(agent)` | 为新加入的 worker 设置反向设备 map 和设备列表 |
| `_backend_type_repr(self)` | BackendType 的字符串表示 |
| `_construct_rpc_backend_options(backend, rpc_timeout, init_method, **kwargs)` | 构造 RPC 后端配置选项 |
| `_init_backend(backend, *args, **kwargs)` | 初始化 RPC 后端 |
| `_npu_tensorpipe_construct_rpc_backend_options_handler(rpc_timeout, init_method, ...)` | 构造 NPU TensorPipe RPC 后端选项（创建 `NPUTensorPipeRpcBackendOptions`） |
| `_npu_tensorpipe_init_backend_handler(store, name, rank, world_size, ...)` | 初始化 NPU TensorPipe RPC 后端（创建 TensorPipeAgent，分配设备、校验设备映射） |
| `_faulty_tensorpipe_construct_rpc_backend_options_handler(...)` | 构造 Faulty TensorPipe RPC 后端选项（用于故障注入测试） |
| `_faulty_tensorpipe_init_backend_handler(...)` | 初始化 Faulty TensorPipe RPC 后端（用于故障注入测试） |
| `_rpc_backend_registry()` | RPC 后端注册入口，注册 "NPU_TENSORPIPE" 和 "NPU_FAULTY_TENSORPIPE" 后端 |

---

## 11. RPC 选项 `rpc/options.py`

| API | 功能说明 |
|-----|----------|
| `_to_device(device)` | 将设备字符串/索引转换为 `torch.device`，验证是否属于 NPU 设备类型 |
| `_to_device_map(device_map)` | 将设备映射字典转换为 `{torch.device: torch.device}` 格式，验证 1-to-1 映射 |
| `_to_device_list(devices)` | 将设备列表转换为 `[torch.device, ...]` |

### 类

| API | 功能说明 |
|-----|----------|
| `NPUTensorPipeRpcBackendOptions` | NPU TensorPipe RPC 后端选项类，支持 `set_device_map()` 和 `set_devices()` 配置设备映射 |

---

## 12. 神经网络分布式辅助 `nn/functional.py`

| API | 功能说明 |
|-----|----------|
| `_allgather_base_backward_hccl(ctx, grad_output)` | `_AllGatherBase` 的 NPU/HCCL 反向传播函数（PyTorch 原生仅支持 NCCL，PTA 增加 HCCL 支持） |

---

## 13. DTensor 公共工具 `tensor/_common.py`

| API | 功能说明 |
|-----|----------|
| `get_redistributed_local_args(op_info, output_sharding)` | 根据 sharding 方案将张量参数 redistribute 到目标 placement 并获取本地张量 |
| `get_redistributed_local_kwargs(kwargs_spec_infer_func, op_info, output_sharding)` | 根据 sharding 方案推断并 redistribute 关键字参数到目标 placement |
| `get_empty_local_results(op_info, output_sharding)` | 为不参与当前 mesh 的 rank 返回空张量结果 |

---

## 14. DTensor 补丁 `tensor/_dtensor_patch.py`

| API | 功能说明 |
|-----|----------|
| `_patched_kwargs_strategy(self)` | 补丁版 `OpSchema.kwargs_strategy` 属性（为旧版 PyTorch 增加 kwargs pytree 支持） |
| `_patched_expand_to_full_mesh_op_strategy(mesh, op_schema, ...)` | 补丁版 `expand_to_full_mesh_op_strategy` 函数（为旧版 PyTorch 增加 kwargs 策略展开） |
| `_patched_register_tuple_strategy()` | 补丁版 `TupleStrategy` pytree 节点注册（为旧版 PyTorch 增加 tensor list 扁平化） |
| `_apply_dtensor_patch()` | 应用 DTensor 兼容性补丁（根据 PyTorch 版本自动选择需要的补丁） |

---

## 15. Sharded Tensor 补丁 `tensor/_sharded_tensor_patch.py`

| API | 功能说明 |
|-----|----------|
| `_patched_sharded_tensor_npu(self, device, non_blocking, memory_format, process_group)` | 补丁版 `ShardedTensor.npu()` 方法，将 CPU 上的 ShardedTensor 各分片移动到对应 NPU 设备 |
| `_apply_sharded_tensor_npu_patch()` | 为 `ShardedTensor` 类添加 `.npu()` 方法 |

---

## 16. DTensor 矩阵运算 Sharding 策略 `tensor/_matrix_ops.py`

### `@register_sharding` 装饰的 Sharding 策略

| API | 对应算子 | 功能说明 |
|-----|----------|----------|
| `custom_matmul_strategy(tensor1, tensor2)` | `aten.matmul.default` | 自定义 matmul sharding 策略，支持 1D×1D / 1D×ND / ND×1D / ND×ND 各种维度组合的 Shard/Partial/Replicate 切分 |
| `custom_matmul_backward_strategy(grad, tensor1, tensor2, mask)` | `aten.matmul_backward.default` | 自定义 matmul 反向 sharding 策略，覆盖各种输入维度组合的梯度 Shard/Partial 切分 |
| `npu_all_gather_base_mm_strategy(x1, x2, hcom, world_size, ...)` | `npu.npu_all_gather_base_mm.default` | NPU all_gather + matmul 融合算子的 sharding 策略，支持 (S0,R)→R 和 (S0,S1)→S1 两种策略 |
| `npu_mm_reduce_scatter_base_strategy(x1, x2, hcom, world_size, ...)` | `npu.npu_mm_reduce_scatter_base.default` | NPU matmul + reduce_scatter 融合算子的 sharding 策略，使用 (S1,S0)→S0 策略 |

### `@register_op_strategy` 装饰的 OpStrategy

| API | 对应算子 | 功能说明 |
|-----|----------|----------|
| `npu_grouped_matmul_strategy(op_schema)` | `npu.npu_grouped_matmul.default` / `.List` | 分组矩阵乘法的 sharding 策略，支持多对 (x,weight) 的 Replicate/Partial/Shard 切分组合 |
| `npu_apply_adam_w_strategy(op_schema)` | `npu.npu_apply_adam_w.default` / `.out` | NPU AdamW 优化器的 sharding 策略，将 grad 的 placement 传播到各输出 |
| `custom_dropout_forward_sharding(op_schema)` | `aten.native_dropout.default` | 自定义 dropout 前向 sharding 策略（支持 Shard/Replicate，小维度回退到 Replicate） |
| `custom_dropout_backward_sharding(op_schema)` | `aten.native_dropout_backward.default` | 自定义 dropout 反向 sharding 策略 |
| `custom_bmm_strategy(op_schema)` | `npu.npu_bmmV2.default` | 批量矩阵乘法的 sharding 策略，桥接到 `_mm_like_strategy` |

### 自定义 Op Handler（注册在 `DTensor._op_dispatcher._custom_op_handlers`）

| Handler | 功能说明 |
|-----|----------|
| `_npu_grouped_matmul_handler(op_call, args, kwargs)` | npu_grouped_matmul 的完整 sharding 传播 handler，处理 tensor list 展开、sharding 传播、本地计算、结果包装 |
| `npu_comm_mm_fusion_handler(op_call, args, kwargs)` | all_gather_base_mm / mm_reduce_scatter_base 的 handler，自动修正输出 meta shape（除以/乘以 world_size） |
| `_npu_apply_adam_w_handler(op_call, args, kwargs)` | npu_apply_adam_w 的 handler，支持 inplace out 变体 |

---

## 17. DTensor 数学运算 Sharding 策略 `tensor/_math_ops.py`

### `@register_op_strategy` 装饰

| API | 对应算子 | 功能说明 |
|-----|----------|----------|
| `npu_rms_norm_strategy(op_schema)` | `npu.npu_rms_norm.default` | RMS Normalization 的 sharding 策略，normalized dims 必须为 Replicate，batch dims 可 Shard |
| `npu_rms_norm_backward_strategy(op_schema)` | `npu.npu_rms_norm_backward.default` | RMS Normalization 反向的 sharding 策略，对 gamma 梯度进行 reduction |
| `npu_add_rms_norm_strategy(op_schema)` | `npu.npu_add_rms_norm.default` | Add + RMS Norm 融合算子的 sharding 策略，x1/x2/y 使用相同 placement |
| `custom_cross_entropy_loss_sharding(op_schema)` | `npu.npu_cross_entropy_loss.default` | 交叉熵损失的 sharding 策略，支持 reduction='none'/'mean'/'sum' 的 Shard/Partial |
| `custom_cross_entropy_loss_backward_strategy(op_schema)` | `npu.npu_cross_entropy_loss_backward.default` | 交叉熵损失反向的 sharding 策略 |

### `@register_sharding` 装饰

| API | 对应算子 | 功能说明 |
|-----|----------|----------|
| `npu_rotary_mul_strategy(x, r1, r2, rotary_mode)` | `npu.npu_rotary_mul.default` | 旋转位置编码乘法的 sharding 策略，支持 B/N 维度的 Shard（r1/r2 的 1-dim 广播处理） |
| `npu_rotary_mul_backward_strategy(grad, x, r1, r2, ...)` | `npu.npu_rotary_mul_backward.default` | 旋转位置编码反向的 sharding 策略 |
| `custom_npu_gather_backward_strategy(grad, input_shape, ...)` | `npu.npu_gather_backward.default` | Gather 反向（scatter_add）的 sharding 策略，支持非 gather 维度的 Shard |
| `custom_npu_swiglu_strategy(x, dim)` | `npu.npu_swiglu.default` | SwiGLU 激活函数的 sharding 策略，除 dim 维度外皆可 Shard |
| `custom_npu_swiglu_backward_strategy(grad, x, dim)` | `npu.npu_swiglu_backward.default` | SwiGLU 反向的 sharding 策略 |
| `custom_npu_conv2d_strategy(x, weight, bias, ...)` | `npu.npu_conv2d.default` | NPU 2D 卷积的 sharding 策略，支持 DP（N 维度 Shard）和 TP（Co 维度 Shard，groups==1） |
| `custom_npu_conv2d_backward_strategy(x, grad_output, weight, ...)` | `npu.npu_conv2d_backward.default` | NPU 2D 卷积反向的 sharding 策略，支持 DP 和 TP 的梯度计算 |
| `custom_grouped_matmul_add__strategy(y, x, weight, ...)` | `npu.npu_grouped_matmul_add_.default` | 分组矩阵乘法加法的 sharding 策略，支持 D 维度的 Shard |
| `custom_npu_repeat_interleave_self_int_strategy(x, repeat, dim, ...)` | `aten.repeat_interleave.self_int` | 自定义 repeat_interleave 的 sharding 策略，支持 dim 和非 dim 维度 Shard |
| `custom_npu_repeat_interleave_backward_int_strategy(grad, x, repeats, dim)` | `npu.repeat_interleave_backward_int.default` | 自定义 repeat_interleave 反向的 sharding 策略 |
| `custom_KLDivLoss_forward_strategy(x, target, reduction, ...)` | `aten.kl_div.default` | KL 散度损失的 sharding 策略，支持 'none'(Shard) / 'mean'(Partial avg) / 'sum'(Partial sum) |
| `custom_KLDivLoss_backward_strategy(grad_out, x, target, reduction, ...)` | `npu.kl_div_backward.default` | KL 散度损失反向的 sharding 策略 |

### 工具函数

| API | 功能说明 |
|-----|----------|
| `is_tensor_evenly_shardable(shape, spec)` | 检查给定形状是否能被 spec 中的 placements 均匀切分 |

---

## 18. DTensor Attention Sharding 策略 `tensor/_attention.py`

### `@register_sharding` 装饰

| API | 对应算子 | 功能说明 |
|-----|----------|----------|
| `npu_fusion_attention_strategy(query, key, value, head_num, ...)` | `npu.npu_fusion_attention.default` | NPU 融合注意力前向的 sharding 策略，支持 DP（B 维度 Shard）和 TP（N 维度 Shard），自动处理 atten_mask 分片 |
| `npu_fusion_attention_grad_strategy(query, key, value, dy, ...)` | `npu.npu_fusion_attention_grad.default` | NPU 融合注意力反向的 sharding 策略 |
| `npu_fusion_attention_v3_strategy(query, key, value, ...)` | `npu.npu_fusion_attention_v3.default` | NPU 融合注意力 v3 版本前向的 sharding 策略 |
| `npu_fusion_attention_grad_v3_strategy(query, key, value, dy, ...)` | `npu.npu_fusion_attention_grad_v3.default` | NPU 融合注意力 v3 版本反向的 sharding 策略 |

### 内部函数

| API | 功能说明 |
|-----|----------|
| `_infer_npu_fusion_attention_grad_kwargs_spec(op_schema, output_sharding)` | 从 output spec 推断注意力反向 kwargs 的 target placement（处理 atten_mask/softmax_max/sum/attention_in） |
| `_npu_fusion_attention_handler(op_call, args, kwargs)` | 注意力算子的通用 DTensor handler，自动将普通 tensor 转为 DTensor，处理 head_num 重计算、构造 kwargs spec |

---

## 19. DTensor MoE 算子 Sharding 策略 `tensor/_moe_ops.py`

| API | 对应算子 | 功能说明 |
|-----|----------|----------|
| `npu_moe_token_permute_strategy(tokens, indices, ...)` | `npu.npu_moe_token_permute.default` | MoE token 重排的 sharding 策略，支持 hidden_size 维度 Shard（Shard(1)） |
| `npu_moe_token_permute_grad_strategy(tokens, grad_permuted_tokens, ...)` | `npu.npu_moe_token_permute_grad.default` | MoE token 重排反向的 sharding 策略，hidden_size 维度可 Shard |
| `npu_moe_token_permute_grad_v2_strategy(grad_permuted_tokens, sorted_indices, ...)` | `npu.npu_moe_token_permute_grad_v2.default` | MoE token 重排反向 v2 的 sharding 策略 |
| `npu_moe_token_unpermute_strategy(permuted_tokens, sorted_indices, ...)` | `npu.npu_moe_token_unpermute.default` | MoE token 恢复排列的 sharding 策略，支持 hidden_size 维度 Shard |
| `npu_moe_token_unpermute_grad_strategy(permuted_tokens, grad_unpermuted_tokens, ...)` | `npu.npu_moe_token_unpermute_grad.default` | MoE token 恢复排列反向的 sharding 策略，grad_unpermuted_tokens 为 Partial |

---

## 20. DTensor 逐点运算 `tensor/_pointwise_ops.py`

| API | 功能说明 |
|-----|----------|
| `custom_linear_pointwise_strategy(op_schema)` | 自定义线性逐点运算 sharding 策略，应用于 npu_dtype_cast 及其反向、_npu_dtype_cast 及其反向 |

### 注册为 `pointwise_strategy` 的逐点算子

`fast_gelu`, `npu_fast_gelu`, `npu_layer_norm_eval`, `npu_fast_gelu_backward`, `aten.isclose.default`, `aten.isfinite.default`

---

## 21. Context Parallel（上下文并行）`tensor/experimental/_context_parallel/`

### 公开 API（`__init__.py`）

| API | 功能说明 |
|-----|----------|
| `context_parallel(*args, **kwargs)` | 上下文并行上下文管理器，包装 PyTorch 原生 `_torch_context_parallel` |
| `context_parallel_unshard(*args, **kwargs)` | 上下文并行解切分（unshard），包装 PyTorch 原生实现 |
| `set_rotate_method(*args, **kwargs)` | 设置上下文并行的张量旋转方法 |

### 内部 API（`_npu_attention.py`）

| API | 功能说明 |
|-----|----------|
| `npu_enable_cp_dtensor_dispatcher()` | 注册 NPU SDPA 前向/反向 handler 到 DTensor op_dispatcher |
| `npu_disable_cp_dtensor_dispatcher()` | 从 DTensor op_dispatcher 移除 NPU SDPA handler，清理 step_cache_stack |
| `_npu_fa_v3_handler(op_call, args, kwargs)` | NPU 融合注意力 v3 的 Context Parallel 前向 handler，执行 ring attention 分步计算 |
| `_npu_fa_grad_v3_handler(op_call, args, kwargs)` | NPU 融合注意力 v3 的 Context Parallel 反向 handler，从 step_cache_stack 恢复状态执行反向 |
| `_make_forward_op(step_caches, pt)` | 创建 ring attention 单步前向闭包，支持 online softmax 合并（Python/Ops 两种路径） |
| `_make_backward_op(step_caches, pt)` | 创建 ring attention 单步反向闭包，消费 step_cache_stack LIFO |
| `_merge_softmax_stats_python(prev_max, prev_sum, cur_max, cur_sum)` | Python 实现的多步 softmax stats 合并 |
| `_merge_softmax_stats_with_ring_update(prev_attn, ...)` | 使用硬件算子 `npu_ring_attention_update` 合并多步 attention 输出和 softmax stats |
| `_convert_softmax_to_logsumexp(softmax_max, softmax_sum)` | 将 softmax stats 转换为 log-sum-exp 格式 |
| `_validate_bnsd_layout(query, key, value, input_layout, ...)` | 验证 q/k/v 满足 BNSD 布局要求 |
| `_validate_cp_passthrough_args(args, ix, ...)` | 验证不支持 ring step 变换的参数（pse, padding_mask, prefix, sink 等） |
| `_validate_cp_sparse_args(args, ix, ...)` | 验证稀疏 attention mask 模式 |
| `_bnsd_to_sbh(attn_out)` / `_sbh_to_bnsd(attn_out, head_num)` | BNSD ↔ SBH 布局转换 |

---

## 统计汇总

| 分类 | 数量 |
|------|------|
| 模块级函数定义（总计） | **170** |
| 公开 API（无 `_` 前缀） | **59** |
| 内部函数（`_` 前缀） | **111** |
| 类定义 | **8** |
| `@register_sharding` 装饰的 sharding 策略 | **24** |
| `@register_op_strategy` 装饰的 op strategy | **11**（12 个注册点） |
| 自定义 op handler（注册到 `_custom_op_handlers`） | **5** |

### 按模块分布

| 模块 | 公开 API | 内部 API | 类 |
|------|----------|----------|-----|
| `distributed_c10d.py` | 2 | 13 | - |
| `rendezvous.py` | - | 5 | 1 |
| `run.py` | 1 | 1 | - |
| `_symmetric_memory/` | - | 1 | - |
| `fsdp/_add_fsdp_patch.py` | 2 | 12 | 1 |
| `fsdp/_fsdp_collectives.py` | 3 | - | - |
| `pipelining/_pipelining_fx.py` | 3 | 17 | 4 |
| `pipelining/_pipelining_patch.py` | 1 | 7 | 1 |
| `rpc/backend_registry.py` | - | 16 | - |
| `rpc/options.py` | - | 3 | 1 |
| `nn/functional.py` | - | 1 | - |
| `tensor/_common.py` | 3 | - | - |
| `tensor/_dtensor_patch.py` | - | 4 | - |
| `tensor/_sharded_tensor_patch.py` | - | 2 | - |
| `tensor/_matrix_ops.py` | 10 | 7 | - |
| `tensor/_math_ops.py` | 18 | - | - |
| `tensor/_attention.py` | 4 | 2 | - |
| `tensor/_moe_ops.py` | 5 | - | - |
| `tensor/_pointwise_ops.py` | 1 | - | - |
| `tensor/experimental/_context_parallel/` | 5 | 20 | - |
| `__init__.py` | 4 | - | - |
