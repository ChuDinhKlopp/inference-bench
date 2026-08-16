#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
rivf26_root=${RIVF26_ROOT:-$(cd -- "$script_dir/../.." && pwd)}
source "$rivf26_root/scripts/common/paths.sh"

matrix_id=${RIVF26_MATRIX_ID:-$(date -u +%Y%m%d_%H%M%S)_performance_pubmed_matrix}
status_dir=$RIVF26_BULK_ROOT/logs/$matrix_id
status_file=$status_dir/status.jsonl
mkdir -p "$status_dir"
if [[ -e "$status_file" ]]; then
  echo "refusing to overwrite matrix status: $status_file" >&2
  exit 2
fi

# Predeclared alternating order limits monotonic precision/thermal bias while
# retaining the full-precision baseline as the first operational reference.
precisions=(w16kv16 w8kv8 w8kv16 w16kv8)
printf '{"event":"matrix_start","matrix_id":"%s","epoch_s":%(%s)T,"order":["w16kv16","w8kv8","w8kv16","w16kv8"]}\n' \
  "$matrix_id" -1 >> "$status_file"

for precision in "${precisions[@]}"; do
  run_id=$(date -u +%Y%m%d_%H%M%S)_performance_pubmed_${precision}_mns128
  printf '{"event":"run_start","run_id":"%s","precision":"%s","epoch_s":%(%s)T}\n' \
    "$run_id" "$precision" -1 >> "$status_file"
  wrapper=$script_dir/run_trace_azure_pubmed_Qwen3.6-35B-A3B_${precision}.sh
  if RIVF26_RUN_ID="$run_id" BENCH_ARRIVAL_RATE=azure "$wrapper"; then
    printf '{"event":"run_complete","run_id":"%s","precision":"%s","status":"PASS","epoch_s":%(%s)T}\n' \
      "$run_id" "$precision" -1 >> "$status_file"
  else
    rc=$?
    printf '{"event":"run_complete","run_id":"%s","precision":"%s","status":"FAIL","exit_code":%d,"epoch_s":%(%s)T}\n' \
      "$run_id" "$precision" "$rc" -1 >> "$status_file"
    exit "$rc"
  fi
done

printf '{"event":"matrix_complete","matrix_id":"%s","status":"PASS","epoch_s":%(%s)T}\n' \
  "$matrix_id" -1 >> "$status_file"
echo "performance matrix PASS: $matrix_id"
