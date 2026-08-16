#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
rivf26_root=${RIVF26_ROOT:-$(cd -- "$script_dir/../.." && pwd)}
source "$rivf26_root/scripts/common/venv.sh"
source "$rivf26_root/scripts/common/paths.sh"

matrix_id=${RIVF26_MATRIX_ID:-$(date -u +%Y%m%d_%H%M%S)_accuracy_gpqa_matrix}
selected_mns=${RIVF26_ACCURACY_MAX_NUM_SEQS:-}
if [[ ! "$selected_mns" =~ ^[1-9][0-9]*$ ]] || (( selected_mns > 4096 )); then
  echo "set RIVF26_ACCURACY_MAX_NUM_SEQS to the value selected from the 198-request length pilot" >&2
  exit 2
fi

# Use one pilot-selected concurrency for all formats. Alternating weight/KV
# precision limits monotonic precision and thermal bias.
precisions=(w16kv16 w8kv8 w8kv16 w16kv8)

if [[ ${RIVF26_DRY_RUN:-0} == 1 ]]; then
  printf 'accuracy matrix: %s\n' "$matrix_id"
  for precision in "${precisions[@]}"; do
    run_id=DRYRUN_accuracy_gpqa_${precision}_mns${selected_mns}
    wrapper=$script_dir/run_trace_azure_gpqa_Qwen3.6-35B-A3B_${precision}.sh
    printf '\n### %s\n' "$run_id"
    RIVF26_RUN_ID="$run_id" RIVF26_MAX_NUM_SEQS="$selected_mns" \
      RIVF26_GPQA_RUN_KIND=official RIVF26_GPQA_NUM_SAMPLES=5 \
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

printf '{"event":"matrix_start","matrix_id":"%s","epoch_s":%(%s)T,"total_runs":4,"max_num_seqs":%d}\n' \
  "$matrix_id" -1 "$selected_mns" >> "$status_file"

for precision in "${precisions[@]}"; do
  run_id=$(date -u +%Y%m%d_%H%M%S)_accuracy_gpqa_${precision}_mns${selected_mns}
  wrapper=$script_dir/run_trace_azure_gpqa_Qwen3.6-35B-A3B_${precision}.sh
  printf '{"event":"run_start","run_id":"%s","precision":"%s","max_num_seqs":%d,"epoch_s":%(%s)T}\n' \
    "$run_id" "$precision" "$selected_mns" -1 >> "$status_file"
  if RIVF26_RUN_ID="$run_id" RIVF26_MAX_NUM_SEQS="$selected_mns" \
    RIVF26_GPQA_RUN_KIND=official RIVF26_GPQA_NUM_SAMPLES=5 \
    BENCH_ARRIVAL_RATE=none "$wrapper"; then
    printf '{"event":"run_complete","run_id":"%s","precision":"%s","max_num_seqs":%d,"status":"PASS","epoch_s":%(%s)T}\n' \
      "$run_id" "$precision" "$selected_mns" -1 >> "$status_file"
  else
    rc=$?
    printf '{"event":"run_complete","run_id":"%s","precision":"%s","max_num_seqs":%d,"status":"FAIL","exit_code":%d,"epoch_s":%(%s)T}\n' \
      "$run_id" "$precision" "$selected_mns" "$rc" -1 >> "$status_file"
    exit "$rc"
  fi
done

printf '{"event":"matrix_complete","matrix_id":"%s","status":"PASS","epoch_s":%(%s)T}\n' \
  "$matrix_id" -1 >> "$status_file"
echo "accuracy matrix PASS: $matrix_id"
