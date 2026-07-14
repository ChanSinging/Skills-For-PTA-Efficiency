基于 PTA 项目的实际代码，Framework 集成工程师把 Ascend NPU kernel 接到 PyTorch 需要走以下五条通路：

### 一、设备后端注册（让 PyTorch 认识 NPU）

核心原理：PyTorch 预留了 PrivateUse1 作为第三方设备插槽，PTA 把它重命名为 npu。
#### C++ 层静态注册
torch_npu/csrc/core/npu/impl/NPUGuardImpl.cpp:
```c++
 // 模块加载时自动执行
  c10::register_privateuse1_backend("npu");           // PrivateUse1 → "npu"
  at::RegisterPrivateUse1HooksInterface(get_npu_hooks()); // 注册设备钩子
  c10::SetStorageImplCreate(PrivateUse1, make_npu_storage_impl); // 存储分配
```
  NPUHooksInterface 实现 PrivateUse1HooksInterface 虚接口，提供：
  - isAvailable() — NPU 是否可用
  - getDeviceFromPtr() — 从指针反查设备
  - getNewGenerator() — 创建随机数生成器
  - hasPrimaryContext() — 设备上下文管理

#### Python 层注册
```python
# backend.py
torch.utils.rename_privateuse1_backend("npu")
torch._register_device_module("npu", torch_npu.npu)  # 使 torch.npu 可用
torch.utils.generate_methods_for_privateuse1_backend(for_tensor=True, for_module=True, for_storage=True)
# 自动生成 Tensor.npu(), Module.npu(), Storage.npu()
```
效果：调用这三行后，torch.npu.is_available()、tensor.npu()、model.npu() 全部可用。

### 算子注册（让 aten 调度到 NPU kernel）
#### TORCH_LIBRARY_IMPL 注册

在 PyTorch dispatcher 的 PrivateUse1 调度键上注册 NPU 实现：
```c++
  // VariableFallbackKernel.cpp — 自动求导回退
  TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
      m.impl("abs.out", torch_npu::npu_abs_out);  // NPU kernel
  }

  TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
      // 对没有专门 autograd 公式的算子，自动回退到正向+反向组合
  }

  TORCH_LIBRARY_IMPL(aten, AutocastPrivateUse1, m) {
      // AMP 混合精度策略：哪些算子降精度到 fp16，哪些保持 fp32
  }
```

#### 自定义 NPU 算子

  PTA 新增的 Ascend 特有算子（如 npu_fusion_attention）通过 TORCH_LIBRARY 注册：

```c++
TORCH_LIBRARY(npu, m) {
  m.def("npu_fusion_attention(...) -> Tensor");
  m.def("npu_rms_norm(Tensor x, Tensor gamma, float epsilon) -> (Tensor, Tensor)");
}
```

Python 侧自动暴露为 torch.ops.npu.npu_fusion_attention，再通过 _exports.py 映射到 torch_npu.npu_fusion_attention。

### 分布式后端接入（让 HCCL 替代 NCCL）

ProcessGroupHCCL 继承 c10d::Backend，把 HCCL C API 包装为 PyTorch c10d 接口：
```c++
  class ProcessGroupHCCL : public c10d::Backend {
      c10::intrusive_ptr<Work> allreduce(std::vector<at::Tensor>& tensors, ...);
      c10::intrusive_ptr<Work> broadcast(std::vector<at::Tensor>& tensors, ...);
      c10::intrusive_ptr<Work> allgather(...);
      c10::intrusive_ptr<Work> reduce_scatter(...);
      void groupStart();   // HCCL group 合并下发
      void groupEnd();
  };
```
#### PyBind11 绑定
  Init.cpp 把 ProcessGroupHCCL 暴露给 Python：

```c++
  auto processGroupHCCL = intrusive_ptr_no_gil_destructor_class_<ProcessGroupHCCL>(
      module, "ProcessGroupHCCL", dist.attr("Backend"))
      .def("allreduce", ...)
      .def("_group_start", &ProcessGroupHCCL::groupStart)
      .def("_group_end", &ProcessGroupHCCL::groupEnd);
```

#### Python 层注册为 c10d Backend

```python
  torch.distributed.Backend.register_backend(
      "hccl",
      lambda opts, pg_options: _new_process_group_hccl_helper(...),
      extended_api=True, devices=["npu"])
```

### DTensor 分片策略注册（让分布式张量支持 NPU 算子）
NPU 自定义算子在 DTensor 下做分布式推理/训练时，需要告诉 DTensor 如何切分：

```python
  # _math_ops.py — register_op_strategy 方式（完整策略）
  @register_op_strategy(npu.npu_rms_norm.default)
  def npu_rms_norm_strategy(op_schema):
      # 指定 normalized dims 必须 Replicate，batch dims 可 Shard

  # _matrix_ops.py — register_sharding 方式（声明式）
  @register_sharding(aten.matmul.default)
  def custom_matmul_strategy(tensor1, tensor2):
      return [
          ([Replicate()], [Replicate(), Replicate()]),          # 全复制
          ([Shard(0)], [Shard(0), Replicate()]),                # 按 batch 切分
          ([Partial()], [Shard(-1), Shard(-2)]),                # 部分和
      ]
```      
PTA 为 35+ 个 NPU 算子注册了分片策略（RMS Norm、Rotary、SwiGLU、MoE、FusionAttention 等）。

### FSDP / Pipeline 等高层 API 适配

  不是重新实现，而是补丁式适配：

```python
  # 必需补丁：让 FSDP 认识 NPU 设备
  FSDPState._root_pre_forward = _patched_root_pre_forward
  # 原始代码: device.type in ["cuda", "hpu", "xpu", "mtia"]
  # 补丁后:   device.type in ["cuda", "hpu", "xpu", "mtia", "npu"]

  # 算子补丁：注册 NPU 上的 FSDP 集合通信算子
  @torch.library.impl(lib, "all_gather_copy_in", "PrivateUse1")
  def all_gather_copy_in_npu(...): ...

  # 性能优化补丁：NPU 显存缓存复用
  _patched_foreach_all_gather  # 使用 FSDPMemCache 避免反复分配
```

### 总结
五步走：设备注册 → 算子注册 → 分布式后端注册 → DTensor 分片策略 → 高层 API 补丁。每一步都有清晰的对标接口和代码模式。
