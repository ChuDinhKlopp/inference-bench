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
source "$rivf26_root/scripts/common/venv.sh"
source "$rivf26_root/scripts/common/paths.sh"
source "$rivf26_root/scripts/common/scheduler_env.sh"
source "$rivf26_root/scripts/common/workload_env.sh"

export RIVF26_MAX_MODEL_LEN=${RIVF26_MAX_MODEL_LEN:-65536}
rivf26_set_scheduler_env
rivf26_set_workload_env performance

max_num_seqs=${RIVF26_MAX_NUM_SEQS:-256}

port=${RIVF26_PORT:-8000}
request_count=${RIVF26_PERFORMANCE_REQUESTS:-1000}
if [[ "$request_count" != 1000 && "$request_count" != 2000 ]]; then
  echo "RIVF26_PERFORMANCE_REQUESTS must be 1000 or 2000; got $request_count" >&2
  exit 2
fi
workload_suffix=
if [[ "$request_count" == 2000 ]]; then
  workload_suffix=_longest
fi
run_id=${RIVF26_RUN_ID:-$(date -u +%Y%m%d_%H%M%S)_performance_pubmed_${request_count}_${precision}_mns${max_num_seqs}}
workload=${RIVF26_PUBMED_WORKLOAD:-$RIVF26_BULK_ROOT/datasets/processed/pubmed_azure_bursty_${request_count}${workload_suffix}.jsonl}
trace_csv=${RIVF26_AZURE_TRACE_CSV:-$rivf26_root/traces/processed/azure_multimodal_bursty_${request_count}.csv}
smoke_gate=${RIVF26_SMOKE_MATRIX:-$rivf26_root/manifests/smoke_matrix_mbt16384_20260816.json}

run_dir=${RIVF26_RUN_DIR:-$rivf26_root/results/part1/performance/$run_id}
bulk_run_dir=${RIVF26_BULK_RUN_DIR:-$RIVF26_BULK_ROOT/results/part1/performance/$run_id}
manifest_dir=$rivf26_root/manifests/$run_id
preflight=$manifest_dir/preflight.json
post_server=$manifest_dir/post_server.json
server_log=$bulk_run_dir/logs/server.log
client_log=$bulk_run_dir/logs/client.log
resource_guard_log=$bulk_run_dir/logs/resource_guard.log
resource_guard_csv=$bulk_run_dir/raw/resource_guard.csv
hbm_prefix=$bulk_run_dir/raw/hbm
hbm_csv=$bulk_run_dir/raw/hbm.csv
server_launcher=$rivf26_root/scripts/servers/run_server_Qwen3.6-35B-A3B_${precision}.sh

validate_cmd=(
  "$RIVF26_VENV_BIN/python" "$script_dir/run_pubmed_trace.py"
  --workload "$workload"
  --trace-csv "$trace_csv"
  --output-dir "$run_dir"
  --max-num-batched-tokens "$RIVF26_MAX_NUM_BATCHED_TOKENS"
  --thinking-token-budget "$RIVF26_THINKING_TOKEN_BUDGET"
  --expected-requests "$request_count"
  --validate-only
)
preflight_cmd=(
  "$RIVF26_VENV_BIN/python" "$rivf26_root/scripts/utilities/preflight.py"
  --run-id "$run_id"
  --mode performance
  --precision "$precision"
  --max-num-seqs "$max_num_seqs"
  --max-num-batched-tokens "$RIVF26_MAX_NUM_BATCHED_TOKENS"
  --estimated-output-gib "${RIVF26_ESTIMATED_OUTPUT_GIB:-80}"
  --safety-reserve-gib "${RIVF26_SAFETY_RESERVE_GIB:-50}"
  --port "$port"
)
client_cmd=(
  "$RIVF26_VENV_BIN/python" "$script_dir/run_pubmed_trace.py"
  --workload "$workload"
  --trace-csv "$trace_csv"
  --output-dir "$run_dir"
  --base-url "http://127.0.0.1:$port"
  --model Qwen3.6-35B-A3B
  --server-log-file "$server_log"
  --server-metrics-poll-interval "${RIVF26_SERVER_METRICS_POLL_INTERVAL:-0.2}"
  --max-num-batched-tokens "$RIVF26_MAX_NUM_BATCHED_TOKENS"
  --thinking-token-budget "$RIVF26_THINKING_TOKEN_BUDGET"
  --expected-requests "$request_count"
  --timeout "${RIVF26_REQUEST_TIMEOUT:-86400}"
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
profiler_enabled=${RIVF26_ENABLE_TORCH_PROFILER:-${ENABLE_TORCH_PROFILER:-0}}
profiler_start_delay=${RIVF26_PROFILER_START_DELAY_SECONDS:-0}
if [[ ! "$profiler_start_delay" =~ ^[0-9]+$ ]]; then
  echo "RIVF26_PROFILER_START_DELAY_SECONDS must be a non-negative integer" >&2
  exit 2
fi

print_command() {
  local label=$1
  shift
  printf '%s:' "$label"
  printf ' %q' "$@"
  printf '\n'
}

if [[ ${RIVF26_DRY_RUN:-0} == 1 ]]; then
  print_command "PubMed/Azure workload validation" "${validate_cmd[@]}"
  print_command "Stage A preflight" "${preflight_cmd[@]}"
  RIVF26_DRY_RUN=1 RIVF26_RUN_ID="$run_id" RIVF26_PORT="$port" \
    RIVF26_MAX_NUM_SEQS="$max_num_seqs" RIVF26_MODE=performance \
    "$server_launcher"
  print_command "Stage B validation" "${post_cmd[@]}"
  print_command "PubMed client" "${client_cmd[@]}"
  print_command "HBM-wrapped client" "$rivf26_root/scripts/monitoring/capture_hbm.sh" "$hbm_prefix" "${client_cmd[@]}"
  printf 'MAX_GEN_TOKS=%s THINKING_TOKEN_BUDGET=%s MAX_NUM_BATCHED_TOKENS=%s BENCH_ARRIVAL_RATE=%s reasoning_effort=%s total_requests=%s\n' \
    "$MAX_GEN_TOKS" "$RIVF26_THINKING_TOKEN_BUDGET" "$RIVF26_MAX_NUM_BATCHED_TOKENS" "$BENCH_ARRIVAL_RATE" "$RIVF26_REASONING_EFFORT" "$request_count"
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
    kill -TERM -- "-$server_pid" 2>/dev/null || true
    for _ in {1..120}; do
      kill -0 -- "-$server_pid" 2>/dev/null || break
      sleep 0.5
    done
    if kill -0 -- "-$server_pid" 2>/dev/null; then
      kill -KILL -- "-$server_pid" 2>/dev/null || true
    fi
    wait "$server_pid" 2>/dev/null || true
  fi
  if (( run_completed == 0 )); then
    printf '{"status":"FAIL","exit_code":%d,"run_id":"%s"}\n' "$rc" "$run_id" > "$run_dir/failure.json"
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# The Azure window can release nearly 1,000 concurrent HTTP connections.
# The inherited soft limit was 1,024 and caused request loss in the first run.
ulimit -n 65536
if (( $(ulimit -n) < 65536 )); then
  echo "unable to raise open-file limit to 65536" >&2
  exit 2
fi
"${validate_cmd[@]}"
"${preflight_cmd[@]}"

export RIVF26_RUN_ID=$run_id
export RIVF26_PORT=$port
export RIVF26_MAX_NUM_SEQS=$max_num_seqs
export RIVF26_MODE=performance
export RIVF26_RUN_DIR=$run_dir
export RIVF26_BULK_RUN_DIR=$bulk_run_dir
export RIVF26_PREFLIGHT_JSON=$preflight

setsid "$server_launcher" > "$server_log" 2>&1 &
server_pid=$!
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting server"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server log: $server_log"
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
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server is ready"

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

"$RIVF26_VENV_BIN/python" "$script_dir/probe_thinking_budget.py" \
  --workload "$workload" \
  --base-url "http://127.0.0.1:$port" \
  --model Qwen3.6-35B-A3B \
  --model-path "${RIVF26_BF16_MODEL_PATH:-/dev/shm/Qwen3.6-35B-A3B}" \
  --thinking-token-budget "$RIVF26_THINKING_TOKEN_BUDGET" \
  --output "$run_dir/thinking_budget_probe.json"

{
  printf 'BENCH_ARRIVAL_RATE=azure '
  printf '%q ' "${client_cmd[@]}"
  printf '\n'
} > "$run_dir/client_command.txt"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting benchmark"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Benchmark log: $client_log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Benchmark output directory: $run_dir"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Benchmark dataset: ccdv/pubmed-summarization"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Benchmark arrival mode: azure"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Benchmark requests: $request_count"
started_epoch_s=$("$RIVF26_VENV_BIN/python" -c 'import time; print(time.time())')
set +e
if [[ "$profiler_enabled" == "1" || "$profiler_enabled" == "2" ]]; then
  # Start the workload first. The optional delay is a warm-up interval under
  # real load, not an idle wait after server readiness.
  BENCH_ARRIVAL_RATE=azure "$rivf26_root/scripts/monitoring/capture_hbm.sh" "$hbm_prefix" "${client_cmd[@]}" \
    > "$client_log" 2>&1 &
  client_pid=$!
  if (( profiler_start_delay > 0 )); then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Workload started (pid=$client_pid); waiting ${profiler_start_delay}s before starting Torch Profiler"
    sleep "$profiler_start_delay"
  fi
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Torch Profiler via /start_profile"
  curl -fsS -X POST "http://127.0.0.1:$port/start_profile" >> "$client_log" 2>&1 || true
  wait "$client_pid"
  client_rc=$?
else
  BENCH_ARRIVAL_RATE=azure "$rivf26_root/scripts/monitoring/capture_hbm.sh" "$hbm_prefix" "${client_cmd[@]}" \
    2>&1 | tee "$client_log"
  client_rc=${PIPESTATUS[0]}
fi
set -e
ended_epoch_s=$("$RIVF26_VENV_BIN/python" -c 'import time; print(time.time())')
if (( client_rc != 0 )); then
  echo "PubMed client/HBM capture failed with exit code $client_rc" >&2
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

"$RIVF26_VENV_BIN/python" "$rivf26_root/scripts/monitoring/parse_nsys_hbm.py" \
  "$hbm_prefix.nsys-rep" "$hbm_csv" \
  --metadata-json "$run_dir/hbm_metadata.json" \
  --experiment-start-epoch-s "$started_epoch_s"

kv_capacity=$("$RIVF26_VENV_BIN/python" -c \
  'import re,sys; t=open(sys.argv[1], errors="replace").read(); m=re.search(r"GPU KV cache size:\s*([0-9,]+) tokens",t,re.I); print(m.group(1).replace(",", "") if m else "")' \
  "$server_log")
plot_cmd=(
  "$RIVF26_VENV_BIN/python" "$rivf26_root/analysis/build_plot_data.py"
  --per-request "$bulk_run_dir/raw/per_request.csv"
  --prometheus "$bulk_run_dir/raw/iteration_metrics.prometheus_samples.jsonl"
  --hbm "$hbm_csv"
  --output "$run_dir/plot_data.json"
  --run-id "$run_id"
  --precision "$precision"
  --mode performance
  --max-num-seqs "$max_num_seqs"
  --max-num-batched-tokens "$RIVF26_MAX_NUM_BATCHED_TOKENS"
  --bin-seconds "${RIVF26_PLOT_BIN_SECONDS:-1}"
  --experiment-start-epoch-s "$started_epoch_s"
)
if [[ -n "$kv_capacity" ]]; then
  plot_cmd+=(--kv-capacity-tokens "$kv_capacity")
fi
"${plot_cmd[@]}"

"$RIVF26_VENV_BIN/python" "$script_dir/finalize_pubmed_run.py" \
  --workload "$workload" \
  --trace-csv "$trace_csv" \
  --client-summary "$run_dir/summary.json" \
  --responses "$bulk_run_dir/raw/responses.jsonl" \
  --preflight "$preflight" \
  --post-server "$post_server" \
  --thinking-budget-probe "$run_dir/thinking_budget_probe.json" \
  --server-command "$run_dir/logs/server_command.txt" \
  --client-command "$run_dir/client_command.txt" \
  --bulk-run-dir "$bulk_run_dir" \
  --output-manifest "$run_dir/manifest.json" \
  --run-id "$run_id" \
  --precision "$precision" \
  --max-num-seqs "$max_num_seqs" \
  --max-num-batched-tokens "$RIVF26_MAX_NUM_BATCHED_TOKENS" \
  --started-epoch-s "$started_epoch_s" \
  --ended-epoch-s "$ended_epoch_s"

run_completed=1
echo "RIVF26 PubMed/Azure run completed: $run_dir"
