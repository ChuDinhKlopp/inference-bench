#!/usr/bin/env bash
set -euo pipefail

test_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
workload_env=$test_dir/../scripts/common/workload_env.sh

env -u MAX_GEN_TOKS -u RIVF26_MAX_GEN_TOKS -u GPQA_MAX_GEN_TOKS \
  -u PUBMED_MAX_GEN_TOKS -u BENCH_ARRIVAL_RATE \
  bash -c 'source "$1"; rivf26_set_workload_env accuracy; [[ $MAX_GEN_TOKS == 32768 && $GPQA_MAX_GEN_TOKS == 32768 && $BENCH_ARRIVAL_RATE == none && $RIVF26_REASONING_EFFORT == high ]]' \
  _ "$workload_env"

env -u MAX_GEN_TOKS -u RIVF26_MAX_GEN_TOKS -u GPQA_MAX_GEN_TOKS \
  -u PUBMED_MAX_GEN_TOKS -u BENCH_ARRIVAL_RATE \
  bash -c 'source "$1"; rivf26_set_workload_env performance; [[ $MAX_GEN_TOKS == 10240 && $PUBMED_MAX_GEN_TOKS == 10240 && $BENCH_ARRIVAL_RATE == azure && $RIVF26_REASONING_EFFORT == low ]]' \
  _ "$workload_env"

if MAX_GEN_TOKS=10240 bash -c 'source "$1"; rivf26_set_workload_env accuracy' _ "$workload_env"; then
  echo "accuracy accepted the performance MAX_GEN_TOKS" >&2
  exit 1
fi

if MAX_GEN_TOKS=32768 bash -c 'source "$1"; rivf26_set_workload_env performance' _ "$workload_env"; then
  echo "performance accepted the accuracy MAX_GEN_TOKS" >&2
  exit 1
fi

if RIVF26_MAX_MODEL_LEN=32768 bash -c 'source "$1"; rivf26_set_workload_env accuracy' _ "$workload_env"; then
  echo "accuracy accepted a context limit with no prompt headroom" >&2
  exit 1
fi

echo "workload environment tests passed"
