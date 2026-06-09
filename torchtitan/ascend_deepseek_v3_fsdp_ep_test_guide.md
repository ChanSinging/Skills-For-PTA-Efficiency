# Ascend NPU 上 DeepSeek V3 FSDP+EP 测试指南

本文档汇总了在昇腾 910B 上运行 DeepSeek V3 FSDP+EP 泛化测试的完整流程，包括数据准备、运行测试、查看性能指标以及精度验证。

---

## 1. 数据准备（enwiki）

### 1.1 直接用项目脚本预处理（推荐）
跳过 WikiExtractor 的大量小文件中间步骤，直接由 XML bz2 生成 torchtitan 所需的 JSONL：

```bash
python scripts/preprocess_enwiki.py \
    --input ~/data/enwiki-latest-pages-articles.xml.bz2 \
    --output ~/data/enwiki_jsonl/train.jsonl
```

### 1.2 若已使用 WikiExtractor
执行合并脚本：

```bash
python scripts/merge_enwiki_to_jsonl.py \
    --input ~/data/enwiki_extracted \
    --output ~/data/enwiki_jsonl/train.jsonl
```

---

## 2. 运行 FSDP+EP 测试

### 2.1 全部配置（默认）
```bash
bash scripts/test_deepseek_v3_fsdp_ep.sh
```

### 2.2 只跑单个配置
```bash
bash scripts/test_deepseek_v3_fsdp_ep.sh fsdp8_ep2
```

可选的 `test_case` 参数：
- `fsdp8` — Pure FSDP（8-way shard）
- `fsdp8_ep2` — FSDP8 + EP2
- `fsdp8_ep4` — FSDP8 + EP4
- `fsdp8_ep8` — FSDP8 + EP8（Pure EP）
- `hsdp2_fsdp4_ep2` — HSDP2 + FSDP4 + EP2
- `fsdp4_tp2_ep2` — FSDP4 + TP2 + EP2
- `fsdp8_ep2_no_sp` — FSDP8 + EP2，关闭 Sequence Parallel
- `all` — 依次跑完全部 7 组（默认）

---

## 3. 查看关键性能指标

脚本已配置 `--metrics.log_freq 1`，每步都会打印以下指标：

| 指标 | 日志字段 | 含义 |
|------|----------|------|
| **MFU** | `mfu: 23.45%` | Model FLOPs Utilization |
| **显存峰值** | `memory: 45.67GiB(78.12%)` | max_reserved 显存及占比 |
| **吞吐量** | `tps: 12,345` | tokens/sec per device |
| **TFLOPS** | `tflops: 123.45` | 单卡每秒浮点运算量 |
| **Loss** | `loss: 6.12345` | 全局平均 loss |
| **Grad Norm** | `grad_norm: 1.2345` | 裁剪后梯度范数 |

### 3.1 TensorBoard 持久化
脚本默认开启 `--metrics.enable_tensorboard true`，日志保存在：
```
outputs_ascend/deepseek_v3_fsdp_ep/<test_name>/tb/
```

启动 TensorBoard：
```bash
tensorboard --logdir outputs_ascend/deepseek_v3_fsdp_ep
```

打开 `http://localhost:6006` 查看：
- `mfu(%)`
- `throughput(tps)`
- `memory/max_reserved(GiB)` / `memory/max_reserved(%)`
- `loss_metrics/global_avg_loss`
- `grad_norm`

---

## 4. 精度验证（数值一致性）

### 4.1 导出 baseline golden 值
选一个最简配置（如 `fsdp8`）作为 golden：

```bash
python scripts/loss_compare.py . . \
    --baseline-module deepseek_v3 \
    --baseline-config deepseek_v3_debugmodel \
    --baseline-options '--dataloader.dataset enwiki --dataloader.dataset_path ~/data/enwiki_jsonl --parallelism.data_parallel_shard_degree 8 --training.steps 5' \
    --steps 5 \
    --export-result ds3_fsdp8_golden.txt
```

### 4.2 对比其他并行配置
```bash
python scripts/loss_compare.py . . \
    --baseline-module deepseek_v3 \
    --baseline-config deepseek_v3_debugmodel \
    --baseline-options '--dataloader.dataset enwiki --dataloader.dataset_path ~/data/enwiki_jsonl --parallelism.data_parallel_shard_degree 4 --parallelism.expert_parallel_degree 2 --training.steps 5' \
    --steps 5 \
    --assert-equal \
    --import-result ds3_fsdp8_golden.txt
```

说明：
- `loss_compare.py` 会自动追加 `--debug.seed=42 --debug.deterministic --metrics.log_freq=1`。
- 它会从 TensorBoard 提取**完整精度**的 `global_avg_loss`，逐 step 做 bit-wise 对比。
- 如果 loss 完全一致，输出 `Assertion passed!`；否则打印差值并返回非 0 退出码。

---

## 5. 常见问题

### WikiExtractor 跑几个小时正常吗？
正常。完整 enwiki dump 超过 1200 万页，压缩包 20GB+，解压后 80GB+。若未加 `--processes`，或磁盘 I/O 较慢，耗时几小时是常见现象。推荐直接使用 `scripts/preprocess_enwiki.py` 一步完成。

### 终端 loss 只有 5 位数字，够判断精度吗？
不够。stdout 为了可读性做了截断。严格精度验证请用 `loss_compare.py` 或 TensorBoard 原始数据。

---

## 6. 相关文件

- `scripts/test_deepseek_v3_fsdp_ep.sh` — FSDP+EP 测试入口
- `scripts/preprocess_enwiki.py` — 直接由 XML bz2 生成 JSONL
- `scripts/merge_enwiki_to_jsonl.py` — 合并 WikiExtractor 输出
- `scripts/loss_compare.py` — 精度对比工具
- `torchtitan/components/metrics.py` — 指标计算逻辑
