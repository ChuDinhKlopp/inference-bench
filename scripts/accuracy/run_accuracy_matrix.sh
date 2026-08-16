#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
rivf26_root=${RIVF26_ROOT:-$(cd -- "$script_dir/../.." && pwd)}
source "$rivf26_root/scripts/common/venv.sh"
source "$rivf26_root/scripts/common/paths.sh"

matrix_id=${RIVF26_MATRIX_ID:-$(date -u +%Y%m%d_%H%M%S)_accuracy_gpqa_matrix}

# Interleave precision order across concurrency levels to reduce monotonic
# precision/thermal bias while retaining w16kv16/mns24 as the first reference.
cells=(
  w16kv16:24 w8kv8:24 w8kv16:24 w16kv8:24
  w16kv8:48 w8kv16:48 w8kv8:48 w16kv16:48
  w16kv16:96 w8kv8:96 w8kv16:96 w16kv8:96
)

if [[ ${RIVF26_DRY_RUN:-0} == 1 ]]; then
  printf 'accuracy matrix: %s\n' "$matrix_id"
  for cell in "${cells[@]}"; do
    precision=${cell%%:*}
    max_num_seqs=${cell##*:}
    run_id=DRYRUN_accuracy_gpqa_${precision}_mns${max_num_seqs}
    wrapper=$script_dir/run_trace_azure_gpqa_Qwen3.6-35B-A3B_${precision}.sh
    printf '\n### %s\n' "$run_id"
    RIVF26_RUN_ID="$run_id" RIVF26_MAX_NUM_SEQS="$max_num_seqs" \
      BENCH_ARRIVAL_RATE=none "$wrapper"
  done
  exit 0
fi

status_dir=$RIVF26_BULK_ROOT/logs/$matrix_id
status_file=$status_dir/status.jsonl
mkdir -p "$status_dir"
if [[ -e "$status_file" ]]; then
  echo "refusing to overwrite matrix status: $status_file" >&2
  exit 2
fi

printf '{"event":"matrix_start","matrix_id":"%s","epoch_s":%(%s)T,"total_runs":12}\n' \
  "$matrix_id" -1 >> "$status_file"

for cell in "${cells[@]}"; do
  precision=${cell%%:*}
  max_num_seqs=${cell##*:}
  run_id=$(date -u +%Y%m%d_%H%M%S)_accuracy_gpqa_${precision}_mns${max_num_seqs}
  wrapper=$script_dir/run_trace_azure_gpqa_Qwen3.6-35B-A3B_${precision}.sh
  printf '{"event":"run_start","run_id":"%s","precision":"%s","max_num_seqs":%d,"epoch_s":%(%s)T}\n' \
    "$run_id" "$precision" "$max_num_seqs" -1 >> "$status_file"
  if RIVF26_RUN_ID="$run_id" RIVF26_MAX_NUM_SEQS="$max_num_seqs" \
    BENCH_ARRIVAL_RATE=none "$wrapper"; then
    printf '{"event":"run_complete","run_id":"%s","precision":"%s","max_num_seqs":%d,"status":"PASS","epoch_s":%(%s)T}\n' \
      "$run_id" "$precision" "$max_num_seqs" -1 >> "$status_file"
  else
    rc=$?
    printf '{"event":"run_complete","run_id":"%s","precision":"%s","max_num_seqs":%d,"status":"FAIL","exit_code":%d,"epoch_s":%(%s)T}\n' \
      "$run_id" "$precision" "$max_num_seqs" "$rc" -1 >> "$status_file"
    exit "$rc"
  fi
done

printf '{"event":"matrix_complete","matrix_id":"%s","status":"PASS","epoch_s":%(%s)T}\n' \
  "$matrix_id" -1 >> "$status_file"
echo "accuracy matrix PASS: $matrix_id"
