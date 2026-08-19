#!/usr/bin/env bash
set -euo pipefail

run_one() {
  local variant=$1
  local launcher=$2
  local stamp
  stamp=$(date -u +%Y%m%d_%H%M%S)

  echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] Starting Part 2 profile: ${variant}"
  RIVF26_PROFILER_AUTO_SHUTDOWN=1 \
  RIVF26_ENABLE_TORCH_PROFILER=1 \
  RIVF26_PROFILER_START_DELAY_SECONDS=180 \
  RIVF26_TORCH_PROFILER_DELAY_ITERS=20 \
  RIVF26_TORCH_PROFILER_WARMUP_ITERS=10 \
  RIVF26_TORCH_PROFILER_MAX_ITERS=10 \
  RIVF26_TORCH_PROFILER_WITH_STACK=0 \
  RIVF26_TORCH_PROFILER_DIR=/run/user/1009/ducct/rivf26/results/part2/${variant}/torch_profiler \
  RIVF26_PERFORMANCE_REQUESTS=1000 \
  RIVF26_PUBMED_WORKLOAD=/run/user/1009/ducct/rivf26/datasets/processed/pubmed_azure_bursty_1000_longest.jsonl \
  RIVF26_MAX_NUM_SEQS=256 \
  RIVF26_MAX_MODEL_LEN=262144 \
  RIVF26_MAX_NUM_BATCHED_TOKENS=32768 \
  RIVF26_THINKING_TOKEN_BUDGET=6144 \
  RIVF26_PLOT_BIN_SECONDS=0.5 \
  RIVF26_RUN_ID=${stamp}_part2_performance_pubmed_1000_longest_${variant}_mns256 \
  "$launcher" || {
    # Auto-shutdown intentionally stops the client once the requested profile
    # has flushed, so the workload may return non-zero because the server is no
    # longer accepting requests.  Continue to the next precision arm while
    # preserving the failed/incomplete run artifacts for auditability.
    local rc=$?
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] ${variant} profiler launcher exited ${rc}; continuing to next arm"
  }
}

# Sequential order: w8kv8 (w8a8), w8kv16 (w8a16), w16kv8 (w16a8), w16kv16.
run_one w8kv8 ./scripts/performance/run_trace_azure_pubmed_Qwen3.6-35B-A3B_w8kv8.sh
run_one w8kv16 ./scripts/performance/run_trace_azure_pubmed_Qwen3.6-35B-A3B_w8kv16.sh
run_one w16kv8 ./scripts/performance/run_trace_azure_pubmed_Qwen3.6-35B-A3B_w16kv8.sh
run_one w16kv16 ./scripts/performance/run_trace_azure_pubmed_Qwen3.6-35B-A3B_w16kv16.sh
