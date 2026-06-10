#!/usr/bin/bash
# FSDP2 reshard_after_forward generalization test — Qwen3 + Qwen3-MoE on Ascend NPU
#
# reshard_after_forward:
#   "always" → true  (Zero3, default): reshard after each forward, re-all-gather next step
#   "never"  → false (Zero2):         keep full params on GPU after forward, skip re-all-gather
#
# CLI: --parallelism.fsdp_reshard_after_forward always|never
#
# Usage: bash scripts/run_qwen3_reshard_test.sh

set -euo pipefail

cd "$(dirname "$0")/.."

: ${PYTIORCH_NPU_ALLOC_CONF:="expandable_segments:False"}
export PYTIORCH_NPU_ALLOC_CONF

NGPU=8
STEPS=1000
DUMP_BASE="./outputs_ascend/npu_reshard_test"
LOG_RANK=0

run_one() {
    local name="$1" config="$2" reshard="$3"
    shift 3
    local dump_dir="${DUMP_BASE}/${name}"
    mkdir -p "${dump_dir}"

    printf "===== %s %s =====\n" "$(date '+%H:%M:%S')" "${name}"
    printf "  model=%-20s reshard_after_forward=%s\n" "${config}" "${reshard}"

    torchrun \
        --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
        --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
        -m torchtitan.train \
        --module qwen3 --config "${config}" \
        --training.steps ${STEPS} \
        --training.local_batch_size 4 \
        --training.seq_len 2048 \
        --training.mixed_precision_param bfloat16 \
        --training.mixed_precision_reduce float32 \
        --parallelism.data_parallel_shard_degree ${NGPU} \
        --parallelism.fsdp_reshard_after_forward "${reshard}" \
        --metrics.log_freq 10 \
        --debug.seed 42 \
        --checkpoint.no-enable \
        --dump_folder "${dump_dir}" \

    echo "  [PASS] ${name}"
    echo ""
}

passed=0
failed=0

test_one() {
    local name="$1" config="$2" reshard="$3"
    shift 3
    if run_one "${name}" "${config}" "${reshard}" "$@"; then
        passed=$((passed + 1))
    else
        failed=$((failed + 1))
    fi
}

# ---- Qwen3 dense, pure FSDP2 ----
test_one "qwen3_zero3"     qwen3_8b      always
test_one "qwen3_zero2"     qwen3_8b      never

# ---- Qwen3-MoE, pure FSDP2 ----
test_one "qw3moe_zero3"    qwen3_30b_a3b always
test_one "qw3moe_zero2"    qwen3_30b_a3b never

# ---- Qwen3 dense + PP2 (pp=2, fsdp=4) ----
# PP on → default = Zero2（避免 microbatch 间反复 all-gather）
test_one "qwen3_pp2_default" qwen3_8b default \
    --parallelism.data_parallel_shard_degree 4 \
    --parallelism.pipeline_parallel_degree 2
test_one "qwen3_pp2_always"  qwen3_8b always \
    --parallelism.data_parallel_shard_degree 4 \
    --parallelism.pipeline_parallel_degree 2
test_one "qwen3_pp2_never"   qwen3_8b never \
    --parallelism.data_parallel_shard_degree 4 \
    --parallelism.pipeline_parallel_degree 2

# ---- Qwen3-MoE + PP2 (pp=2, fsdp=4) ----
test_one "qw3moe_pp2_default" qwen3_30b_a3b default \
    --parallelism.data_parallel_shard_degree 4 \
    --parallelism.pipeline_parallel_degree 2
test_one "qw3moe_pp2_always"  qwen3_30b_a3b always \
    --parallelism.data_parallel_shard_degree 4 \
    --parallelism.pipeline_parallel_degree 2
test_one "qw3moe_pp2_never"   qwen3_30b_a3b never \
    --parallelism.data_parallel_shard_degree 4 \
    --parallelism.pipeline_parallel_degree 2

echo ""
echo "============================================================"
echo " reshard_after_forward Test Summary: Passed=${passed} Failed=${failed}"
echo "============================================================"

[ ${failed} -eq 0 ]
