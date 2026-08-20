#!/usr/bin/env bash
set -euo pipefail

cd ~/repos/inference-bench/rivf26
export RIVF26_ROOT="$PWD"
export RIVF26_VENV_BIN="$HOME/repos/vllm/.venv/bin"
export PATH="$RIVF26_VENV_BIN:$PATH"
export RIVF26_BULK_ROOT="$HOME/rivf26-bulk"

RIVF26_PERFORMANCE_REQUESTS=1000 \
RIVF26_PUBMED_WORKLOAD="$RIVF26_ROOT/datasets/processed/pubmed_azure_bursty_1000_longest.jsonl" \
RIVF26_MAX_NUM_SEQS=256 \
RIVF26_MAX_MODEL_LEN=131072 \
RIVF26_MAX_NUM_BATCHED_TOKENS=32768 \
RIVF26_THINKING_TOKEN_BUDGET=6144 \
RIVF26_PLOT_BIN_SECONDS=0.5 \
RIVF26_ESTIMATED_OUTPUT_GIB=5 \
RIVF26_SAFETY_RESERVE_GIB=5 \
RIVF26_RUNTIME_MIN_HOST_AVAILABLE_GIB=150 \
HIP_VISIBLE_DEVICES=0,1 \
CUDA_VISIBLE_DEVICES=0,1 \
RIVF26_PORT=8000 \
RIVF26_RUN_ID=$(date -u +%Y%m%d_%H%M%S)_part1_performance_pubmed_1000_longest_w16kv16_mns256 \
./scripts/performance/run_trace_azure_pubmed_gpt-oss-120b_w16kv16.sh
