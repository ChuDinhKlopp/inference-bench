#!/usr/bin/bash

cd ~/repos/inference-bench/rivf26
export RIVF26_ROOT="$PWD"
export RIVF26_VENV_BIN="$HOME/repos/vllm/.venv/bin"
export PATH="$RIVF26_VENV_BIN:$PATH"
export RIVF26_BULK_ROOT="$HOME/rivf26-bulk"

# Script defaults (80/50 GiB) assume far more free disk than this host
# currently has (~24 GiB free, 95% full); actual observed runs of this size
# use well under 1 GiB, so 10/10 leaves comfortable margin without tripping
# the preflight capacity gate.
export RIVF26_ESTIMATED_OUTPUT_GIB=10
export RIVF26_SAFETY_RESERVE_GIB=10
export RIVF26_LCB_PERFORMANCE_REQUESTS=1055
export RIVF26_AZURE_LCB_TRACE_CSV="$PWD/traces/processed/azure_multimodal_bursty_1055.csv"
export RIVF26_MAX_NUM_SEQS=128
export RIVF26_MAX_MODEL_LEN=262144
export RIVF26_MAX_NUM_BATCHED_TOKENS=8192
export RIVF26_LCB_PERFORMANCE_THINKING_TOKEN_BUDGET=24576
export RIVF26_LCB_PERFORMANCE_MAX_GEN_TOKS=32768
export RIVF26_PLOT_BIN_SECONDS=0.5

RIVF26_RUN_ID="$(date -u +%Y%m%d_%H%M%S)_performance_livecodebench_v6_w16kv16_mns128" \
./scripts/performance/run_trace_azure_livecodebench_Qwen3.6-35B-A3B_w16kv16.sh

RIVF26_RUN_ID="$(date -u +%Y%m%d_%H%M%S)_performance_livecodebench_v6_w8kv8_mns128" \
./scripts/performance/run_trace_azure_livecodebench_Qwen3.6-35B-A3B_w8kv8.sh

RIVF26_RUN_ID="$(date -u +%Y%m%d_%H%M%S)_performance_livecodebench_v6_w8kv16_mns128" \
./scripts/performance/run_trace_azure_livecodebench_Qwen3.6-35B-A3B_w8kv16.sh

RIVF26_RUN_ID="$(date -u +%Y%m%d_%H%M%S)_performance_livecodebench_v6_w16kv8_mns128" \
./scripts/performance/run_trace_azure_livecodebench_Qwen3.6-35B-A3B_w16kv8.sh


