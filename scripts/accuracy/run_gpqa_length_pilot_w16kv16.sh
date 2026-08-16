#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
max_num_seqs=${RIVF26_MAX_NUM_SEQS:-24}
run_id=${RIVF26_RUN_ID:-$(date -u +%Y%m%d_%H%M%S)_accuracy_gpqa_length_pilot_w16kv16_mns${max_num_seqs}}

RIVF26_RUN_ID="$run_id" \
RIVF26_MAX_NUM_SEQS="$max_num_seqs" \
RIVF26_GPQA_RUN_KIND=length_pilot \
RIVF26_GPQA_NUM_SAMPLES=1 \
BENCH_ARRIVAL_RATE=none \
exec "$script_dir/run_trace_azure_gpqa_Qwen3.6-35B-A3B_w16kv16.sh"
