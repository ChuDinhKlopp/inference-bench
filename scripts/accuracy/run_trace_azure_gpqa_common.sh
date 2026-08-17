#!/usr/bin/env bash
set -euo pipefail

precision=${1:?precision is required}
shift
case "$precision" in
  w16kv16|w8kv16|w8kv8|w16kv8) ;;
  *) echo "unsupported precision: $precision" >&2; exit 2 ;;
esac
if (( $# )); then
  echo "this matrix harness accepts configuration through audited RIVF26_* variables, not passthrough arguments" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
rivf26_root=${RIVF26_ROOT:-$(cd -- "$script_dir/../.." && pwd)}
parent_root=$(cd -- "$rivf26_root/.." && pwd)
source "$rivf26_root/scripts/common/venv.sh"
source "$rivf26_root/scripts/common/paths.sh"
source "$rivf26_root/scripts/common/scheduler_env.sh"
source "$rivf26_root/scripts/common/workload_env.sh"
source "$rivf26_root/scripts/common/hbm_env.sh"

export RIVF26_MAX_MODEL_LEN=${RIVF26_MAX_MODEL_LEN:-262144}
rivf26_set_scheduler_env
rivf26_set_workload_env accuracy

num_samples=${RIVF26_GPQA_NUM_SAMPLES:-1}
run_kind=${RIVF26_GPQA_RUN_KIND:-official}
if [[ "$run_kind" == length_pilot ]]; then
  max_num_seqs=${RIVF26_MAX_NUM_SEQS:-24}
else
  max_num_seqs=${RIVF26_MAX_NUM_SEQS:-256}
fi
if [[ ! "$max_num_seqs" =~ ^[1-9][0-9]*$ ]] || (( max_num_seqs > 4096 )); then
  echo "GPQA requires a positive RIVF26_MAX_NUM_SEQS no larger than 4096; got $max_num_seqs" >&2
  exit 2
fi
case "$run_kind:$num_samples" in
  official:1|length_pilot:1) ;;
  *)
    echo "GPQA run kind/sample mismatch: official and length_pilot both require 1 repeat" >&2
    exit 2
    ;;
esac
if [[ "$run_kind" == official && "$max_num_seqs" != 256 ]]; then
  echo "the selected GPQA accuracy configuration requires RIVF26_MAX_NUM_SEQS=256; got $max_num_seqs" >&2
  exit 2
fi
total_requests=$((198 * num_samples))
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$HIP_VISIBLE_DEVICES}
IFS=',' read -ra hbm_visible_array <<< "$HIP_VISIBLE_DEVICES"
hbm_expected_gpu_count=${#hbm_visible_array[@]}

port=${RIVF26_PORT:-8000}
if [[ "$run_kind" == length_pilot ]]; then
  default_run_id=$(date -u +%Y%m%d_%H%M%S)_accuracy_gpqa_length_pilot_${precision}_mns${max_num_seqs}
else
  default_run_id=$(date -u +%Y%m%d_%H%M%S)_accuracy_gpqa_${precision}_mns${max_num_seqs}
fi
run_id=${RIVF26_RUN_ID:-$default_run_id}
revision=633f5ee89ab8ad4522a9f850766b73f62147ffdd
gpqa_sha256=41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305
gpqa_csv=${RIVF26_GPQA_CSV:-$HOME/.cache/huggingface/hub/datasets--Idavidrein--gpqa/snapshots/$revision/gpqa_diamond.csv}
smoke_gate=${RIVF26_SMOKE_MATRIX:-$rivf26_root/manifests/smoke_matrix_mbt16384_20260816.json}
case "$precision" in
  w8kv16|w8kv8) local_tokenizer=${RIVF26_FP8_MODEL_PATH:-/dev/shm/Qwen3.6-35B-A3B-FP8} ;;
  *) local_tokenizer=${RIVF26_BF16_MODEL_PATH:-/dev/shm/Qwen3.6-35B-A3B} ;;
esac

run_dir=${RIVF26_RUN_DIR:-$rivf26_root/results/part1/accuracy/$run_id}
bulk_run_dir=${RIVF26_BULK_RUN_DIR:-$RIVF26_BULK_ROOT/results/part1/accuracy/$run_id}
manifest_dir=$rivf26_root/manifests/$run_id
preflight=$manifest_dir/preflight.json
post_server=$manifest_dir/post_server.json
dataset_metadata=$manifest_dir/gpqa_source.json
server_log=$bulk_run_dir/logs/server.log
client_log=$bulk_run_dir/logs/client.log
resource_guard_log=$bulk_run_dir/logs/resource_guard.log
resource_guard_csv=$bulk_run_dir/raw/resource_guard.csv
bench_results=$bulk_run_dir/raw/bench_results.json
generations=$bulk_run_dir/raw/generations.json
metrics_prefix=$bulk_run_dir/raw/bench_results
hbm_kernel_dir=$bulk_run_dir/raw/hbm_kernel
hbm_csv=$bulk_run_dir/raw/hbm.csv
server_launcher=$rivf26_root/scripts/servers/run_server_Qwen3.6-35B-A3B_${precision}.sh

validate_cmd=(
  "$RIVF26_VENV_BIN/python" "$script_dir/run_gpqa.py"
  --gpqa-local-csv "$gpqa_csv"
  --local-tokenizer "$local_tokenizer"
  --gpqa-expected-sha256 "$gpqa_sha256"
  --gpqa-expected-rows 198
  --gpqa-revision "$revision"
  --sampling-top-k 20
  --dataset-metadata-json "$dataset_metadata"
  --validate-only
)
preflight_cmd=(
  "$RIVF26_VENV_BIN/python" "$rivf26_root/scripts/utilities/preflight.py"
  --run-id "$run_id"
  --mode accuracy
  --precision "$precision"
  --max-num-seqs "$max_num_seqs"
  --max-num-batched-tokens "$RIVF26_MAX_NUM_BATCHED_TOKENS"
  --estimated-output-gib "${RIVF26_ESTIMATED_OUTPUT_GIB:-80}"
  --safety-reserve-gib "${RIVF26_SAFETY_RESERVE_GIB:-50}"
  --min-host-available-gib "${RIVF26_RUNTIME_MIN_HOST_AVAILABLE_GIB:-256}"
  --port "$port"
)
client_cmd=(
  "$RIVF26_VENV_BIN/python" "$script_dir/run_gpqa.py"
  --gpqa-local-csv "$gpqa_csv"
  --local-tokenizer "$local_tokenizer"
  --gpqa-expected-sha256 "$gpqa_sha256"
  --gpqa-expected-rows 198
  --gpqa-revision "$revision"
  --sampling-top-k 20
  --dataset gpqa
  --gpqa-dataset Idavidrein/gpqa
  --gpqa-config gpqa_diamond
  --gpqa-split train
  --gpqa-max-gen-toks "$GPQA_MAX_GEN_TOKS"
  --num-prompts 198
  --num-samples "$num_samples"
  --seed 42
  --model Qwen3.6-35B-A3B
  --base-url "http://127.0.0.1:$port"
  --max-concurrency "$max_num_seqs"
  --temperature 1.0
  --top-p 0.95
  --timeout "${RIVF26_REQUEST_TIMEOUT:-86400}"
  --enable-thinking
  --server-log-file "$server_log"
  --server-metrics-poll-interval "${RIVF26_SERVER_METRICS_POLL_INTERVAL:-0.2}"
  --iteration-metrics-prefix "$metrics_prefix"
  --save-generations "$generations"
  --output "$bench_results"
)
post_cmd=(
  "$RIVF26_VENV_BIN/python" "$rivf26_root/scripts/utilities/post_server_validate.py"
  --run-id "$run_id"
  --precision "$precision"
  --server-log "$server_log"
  --preflight "$preflight"
  --output "$post_server"
  --port "$port"
  --max-num-batched-tokens "$RIVF26_MAX_NUM_BATCHED_TOKENS"
)
if [[ "$precision" == *kv8 ]]; then
  post_cmd+=(--accept-fp8-kv-scale-one)
fi

print_command() {
  local label=$1
  shift
  printf '%s:' "$label"
  printf ' %q' "$@"
  printf '\n'
}

if [[ ${RIVF26_DRY_RUN:-0} == 1 ]]; then
  print_command "GPQA dataset validation" "${validate_cmd[@]}"
  print_command "Stage A preflight" "${preflight_cmd[@]}"
  RIVF26_DRY_RUN=1 RIVF26_RUN_ID="$run_id" RIVF26_PORT="$port" \
    RIVF26_MAX_NUM_SEQS="$max_num_seqs" RIVF26_MODE=accuracy \
    "$server_launcher"
  print_command "Stage B validation" "${post_cmd[@]}"
  print_command "GPQA client" "${client_cmd[@]}"
  printf 'MAX_GEN_TOKS=%s THINKING_TOKEN_BUDGET=%s MAX_NUM_BATCHED_TOKENS=%s BENCH_ARRIVAL_RATE=%s reasoning_effort=%s run_kind=%s repeats=%s total_requests=%s\n' \
    "$MAX_GEN_TOKS" "$RIVF26_THINKING_TOKEN_BUDGET" "$RIVF26_MAX_NUM_BATCHED_TOKENS" "$BENCH_ARRIVAL_RATE" \
    "$RIVF26_REASONING_EFFORT" "$run_kind" "$num_samples" "$total_requests"
  exit 0
fi

if [[ -e "$run_dir" || -e "$bulk_run_dir" || -e "$manifest_dir" ]]; then
  echo "refusing to overwrite an existing run ID: $run_id" >&2
  exit 2
fi
if [[ ! -x "$server_launcher" ]]; then
  echo "missing server launcher: $server_launcher" >&2
  exit 2
fi
if [[ ! -f "$smoke_gate" ]]; then
  echo "missing four-precision scheduler-budget smoke gate: $smoke_gate" >&2
  exit 2
fi
if ! "$RIVF26_VENV_BIN/python" -c \
  'import json,sys; d=json.load(open(sys.argv[1])); required={"w16kv16","w8kv16","w8kv8","w16kv8"}; runs=d.get("runs",{}); sys.exit(0 if d.get("status")=="PASS" and d.get("max_num_batched_tokens")==16384 and set(runs)==required and all(runs[p].get("status")=="PASS" for p in required) else 2)' \
  "$smoke_gate"; then
  echo "four-precision scheduler-budget smoke gate is not valid: $smoke_gate" >&2
  exit 2
fi

mkdir -p "$run_dir" "$bulk_run_dir/logs" "$bulk_run_dir/raw" "$manifest_dir"
for artifact_kind in logs raw; do
  ln -s "$bulk_run_dir/$artifact_kind" "$run_dir/$artifact_kind"
done

server_pid=
guard_pid=
run_completed=0
cleanup() {
  local rc=$?
  trap - EXIT
  if [[ -n ${guard_pid:-} ]] && kill -0 "$guard_pid" 2>/dev/null; then
    kill -TERM "$guard_pid" 2>/dev/null || true
    wait "$guard_pid" 2>/dev/null || true
  fi
  if [[ -n ${server_pid:-} ]]; then
    rivf26_stop_rocprof_server "$server_pid" "$port"
  fi
  if (( run_completed == 0 )); then
    printf '{"status":"FAIL","exit_code":%d,"run_id":"%s"}\n' "$rc" "$run_id" > "$run_dir/failure.json"
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Dataset identity is checked before GPU allocation. This is deliberately
# offline: the harness never downloads a fallback dataset or model.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
"${validate_cmd[@]}"
"${preflight_cmd[@]}"

export RIVF26_RUN_ID=$run_id
export RIVF26_PORT=$port
export RIVF26_MAX_NUM_SEQS=$max_num_seqs
export RIVF26_MODE=accuracy
export RIVF26_RUN_DIR=$run_dir
export RIVF26_BULK_RUN_DIR=$bulk_run_dir
export RIVF26_PREFLIGHT_JSON=$preflight

setsid "$server_launcher" > "$server_log" 2>&1 &
server_pid=$!
ready=0
for ((attempt=0; attempt<${RIVF26_SERVER_READY_ATTEMPTS:-1800}; attempt++)); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "vLLM exited before readiness; see $server_log" >&2
    tail -100 "$server_log" >&2 || true
    exit 2
  fi
  if "$RIVF26_VENV_BIN/python" -c \
    'import sys,urllib.request; sys.exit(0 if urllib.request.urlopen(sys.argv[1], timeout=2).status == 200 else 1)' \
    "http://127.0.0.1:$port/health" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 1
done
if (( ready == 0 )); then
  echo "vLLM did not become ready within the configured timeout" >&2
  exit 2
fi

"$RIVF26_VENV_BIN/python" "$rivf26_root/scripts/monitoring/resource_guard.py" \
  --path "$bulk_run_dir" \
  --output "$resource_guard_csv" \
  --interval-s "${RIVF26_RESOURCE_GUARD_INTERVAL_S:-30}" \
  --min-free-gib "${RIVF26_RUNTIME_MIN_FREE_GIB:-20}" \
  --min-host-available-gib "${RIVF26_RUNTIME_MIN_HOST_AVAILABLE_GIB:-128}" \
  --required-pid "$server_pid" \
  --interrupt-pid "$server_pid" \
  > "$resource_guard_log" 2>&1 &
guard_pid=$!

post_cmd+=(--monitor-pid "$guard_pid")
"${post_cmd[@]}"
if ! "$RIVF26_VENV_BIN/python" -c \
  'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d["status"] == "PASS" and d.get("long_run_eligible") else 2)' \
  "$post_server"; then
  echo "Stage B did not pass every request-release gate" >&2
  exit 2
fi

{
  printf '%q ' "${client_cmd[@]}"
  printf '\n'
} > "$run_dir/client_command.txt"
started_epoch_s=$(
  "$RIVF26_VENV_BIN/python" -c 'import time; print(time.time())'
)
set +e
"${client_cmd[@]}" \
  2>&1 | tee "$client_log"
client_rc=${PIPESTATUS[0]}
set -e
ended_epoch_s=$(
  "$RIVF26_VENV_BIN/python" -c 'import time; print(time.time())'
)
if (( client_rc != 0 )); then
  echo "GPQA client failed with exit code $client_rc" >&2
  exit "$client_rc"
fi
if ! kill -0 "$guard_pid" 2>/dev/null; then
  wait "$guard_pid" || guard_rc=$?
  echo "resource guard exited during the run (status=${guard_rc:-0})" >&2
  exit 2
fi
kill -TERM "$guard_pid"
wait "$guard_pid" 2>/dev/null || true
guard_pid=

# rocprof only finishes writing kernel_dispatches.csv once its wrapped vLLM process
# exits, so the server must be stopped (gracefully, via the vLLM PID specifically --
# see scripts/common/hbm_env.sh) before HBM data can be parsed.
rivf26_stop_rocprof_server "$server_pid" "$port"
server_pid=

"$RIVF26_VENV_BIN/python" "$rivf26_root/scripts/monitoring/parse_rocprof_hbm.py" \
  --kernel-csv "$hbm_kernel_dir/kernel_dispatches.csv" \
  --reference-json "$hbm_kernel_dir/reference.json" \
  --output-csv "$hbm_csv" --metadata-json "$run_dir/hbm_metadata.json" \
  --expected-gpu-count "$hbm_expected_gpu_count" \
  --bin-seconds "${RIVF26_PLOT_BIN_SECONDS:-1}" \
  --experiment-start-epoch-s "$started_epoch_s"

kv_capacity=$(
  "$RIVF26_VENV_BIN/python" -c \
    'import re,sys; t=open(sys.argv[1], errors="replace").read(); m=re.search(r"GPU KV cache size:\s*([0-9,]+) tokens",t,re.I); print(m.group(1).replace(",", "") if m else "")' \
    "$server_log"
)
plot_cmd=(
  "$RIVF26_VENV_BIN/python" "$rivf26_root/analysis/build_plot_data.py"
  --per-request "$bulk_run_dir/raw/bench_results.per_request.csv"
  --prometheus "$bulk_run_dir/raw/bench_results.prometheus_samples.jsonl"
  --hbm "$hbm_csv"
  --output "$run_dir/plot_data.json"
  --run-id "$run_id"
  --precision "$precision"
  --mode accuracy
  --max-num-seqs "$max_num_seqs"
  --max-num-batched-tokens "$RIVF26_MAX_NUM_BATCHED_TOKENS"
  --bin-seconds "${RIVF26_PLOT_BIN_SECONDS:-1}"
  --experiment-start-epoch-s "$started_epoch_s"
)
if [[ -n "$kv_capacity" ]]; then
  plot_cmd+=(--kv-capacity-tokens "$kv_capacity")
fi
"${plot_cmd[@]}"

"$RIVF26_VENV_BIN/python" "$script_dir/finalize_gpqa_run.py" \
  --bench-results "$bench_results" \
  --generations "$generations" \
  --preflight "$preflight" \
  --post-server "$post_server" \
  --server-command "$run_dir/logs/server_command.txt" \
  --client-command "$run_dir/client_command.txt" \
  --server-log "$server_log" \
  --hbm-csv "$hbm_csv" \
  --bulk-run-dir "$bulk_run_dir" \
  --output-summary "$run_dir/summary.json" \
  --output-manifest "$run_dir/manifest.json" \
  --run-id "$run_id" \
  --precision "$precision" \
  --run-kind "$run_kind" \
  --num-samples "$num_samples" \
  --smoke-matrix "$smoke_gate" \
  --max-num-seqs "$max_num_seqs" \
  --max-num-batched-tokens "$RIVF26_MAX_NUM_BATCHED_TOKENS" \
  --started-epoch-s "$started_epoch_s" \
  --ended-epoch-s "$ended_epoch_s"

run_completed=1
echo "RIVF26 GPQA run completed: $run_dir"
