#!/usr/bin/env bash
set -euo pipefail

precision=${1:?usage: run_smoke.sh PRECISION}
case "$precision" in
  w16kv16|w8kv16|w8kv8|w16kv8) ;;
  *) echo "unsupported precision: $precision" >&2; exit 2 ;;
esac

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
rivf26_root=${RIVF26_ROOT:-$(cd -- "$script_dir/../.." && pwd)}
source "$rivf26_root/scripts/common/venv.sh"
source "$rivf26_root/scripts/common/paths.sh"
source "$rivf26_root/scripts/common/scheduler_env.sh"
rivf26_set_scheduler_env

run_id=${RIVF26_RUN_ID:-$(date -u +%Y%m%d_%H%M%S)_smoke_mbt16384_${precision}}
port=${RIVF26_PORT:-8000}
max_num_seqs=2
run_dir=$rivf26_root/results/part1/smoke/$run_id
bulk_run_dir=$RIVF26_BULK_ROOT/results/part1/smoke/$run_id
manifest_dir=$rivf26_root/manifests/$run_id
preflight=$manifest_dir/preflight.json
post_server=$manifest_dir/post_server.json
server_log=$bulk_run_dir/logs/server.log
hbm_prefix=$bulk_run_dir/raw/hbm
hbm_csv=$bulk_run_dir/raw/hbm.csv
server_launcher=$rivf26_root/scripts/servers/run_server_Qwen3.6-35B-A3B_${precision}.sh
case "$precision" in
  w8*) model_path=${RIVF26_FP8_MODEL_PATH:-/dev/shm/Qwen3.6-35B-A3B-FP8} ;;
  *) model_path=${RIVF26_BF16_MODEL_PATH:-/dev/shm/Qwen3.6-35B-A3B} ;;
esac

if [[ -e "$run_dir" || -e "$bulk_run_dir" || -e "$manifest_dir" ]]; then
  echo "refusing to overwrite smoke run: $run_id" >&2
  exit 2
fi
mkdir -p "$run_dir" "$bulk_run_dir/logs" "$bulk_run_dir/raw" "$manifest_dir"
ln -s "$bulk_run_dir/logs" "$run_dir/logs"
ln -s "$bulk_run_dir/raw" "$run_dir/raw"

server_pid=
run_complete=0
cleanup() {
  local rc=$?
  trap - EXIT
  if [[ -n ${server_pid:-} ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM -- "-$server_pid" 2>/dev/null || true
    for _ in {1..120}; do
      kill -0 "$server_pid" 2>/dev/null || break
      sleep 0.5
    done
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -KILL -- "-$server_pid" 2>/dev/null || true
    fi
    wait "$server_pid" 2>/dev/null || true
  fi
  if (( run_complete == 0 )); then
    printf '{"status":"FAIL","exit_code":%d,"run_id":"%s"}\n' "$rc" "$run_id" > "$run_dir/failure.json"
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"$RIVF26_VENV_BIN/python" "$rivf26_root/scripts/utilities/preflight.py" \
  --run-id "$run_id" --mode smoke --precision "$precision" \
  --max-num-seqs "$max_num_seqs" --max-num-batched-tokens 16384 \
  --estimated-output-gib 2 --safety-reserve-gib 20 --port "$port"

export RIVF26_RUN_ID=$run_id
export RIVF26_PORT=$port
export RIVF26_MAX_NUM_SEQS=$max_num_seqs
export RIVF26_MODE=smoke
export RIVF26_RUN_DIR=$run_dir
export RIVF26_BULK_RUN_DIR=$bulk_run_dir
export RIVF26_PREFLIGHT_JSON=$preflight
export RIVF26_MAX_MODEL_LEN=${RIVF26_MAX_MODEL_LEN:-65536}

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
    'import sys,urllib.request; sys.exit(0 if urllib.request.urlopen(sys.argv[1],timeout=2).status==200 else 1)' \
    "http://127.0.0.1:$port/health" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 1
done
if (( ready == 0 )); then
  echo "server readiness timeout" >&2
  exit 2
fi

post_cmd=(
  "$RIVF26_VENV_BIN/python" "$rivf26_root/scripts/utilities/post_server_validate.py"
  --run-id "$run_id" --precision "$precision" --server-log "$server_log"
  --preflight "$preflight" --output "$post_server" --port "$port"
  --max-num-batched-tokens 16384
)
if [[ "$precision" == *kv8 ]]; then
  post_cmd+=(--accept-fp8-kv-scale-one)
fi
"${post_cmd[@]}"
"$RIVF26_VENV_BIN/python" -c \
  'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d["status"]=="PASS" and d.get("long_run_eligible") else 2)' \
  "$post_server"

started_epoch_s=$("$RIVF26_VENV_BIN/python" -c 'import time; print(time.time())')
"$rivf26_root/scripts/monitoring/capture_hbm.sh" "$hbm_prefix" \
  "$RIVF26_VENV_BIN/python" "$script_dir/smoke_client.py" \
  --output-dir "$bulk_run_dir/raw" --base-url "http://127.0.0.1:$port" \
  --model Qwen3.6-35B-A3B --model-path "$model_path" --server-log-file "$server_log" \
  --max-gen-toks 128 --max-concurrency "$max_num_seqs" --metrics-poll-interval 0.2

"$RIVF26_VENV_BIN/python" "$rivf26_root/scripts/monitoring/parse_nsys_hbm.py" \
  "$hbm_prefix.nsys-rep" "$hbm_csv" --metadata-json "$run_dir/hbm_metadata.json" \
  --experiment-start-epoch-s "$started_epoch_s"
kv_capacity=$("$RIVF26_VENV_BIN/python" -c \
  'import re,sys; t=open(sys.argv[1],errors="replace").read(); m=re.search(r"GPU KV cache size:\s*([0-9,]+) tokens",t,re.I); print(m.group(1).replace(",","") if m else "")' \
  "$server_log")
plot_cmd=(
  "$RIVF26_VENV_BIN/python" "$rivf26_root/analysis/build_plot_data.py"
  --per-request "$bulk_run_dir/raw/per_request.csv"
  --prometheus "$bulk_run_dir/raw/iteration_metrics.prometheus_samples.jsonl"
  --hbm "$hbm_csv" --output "$run_dir/plot_data.json" --run-id "$run_id"
  --precision "$precision" --mode smoke --max-num-seqs "$max_num_seqs"
  --max-num-batched-tokens 16384 --experiment-start-epoch-s "$started_epoch_s"
)
if [[ -n "$kv_capacity" ]]; then
  plot_cmd+=(--kv-capacity-tokens "$kv_capacity")
fi
"${plot_cmd[@]}"

"$RIVF26_VENV_BIN/python" "$script_dir/finalize_smoke.py" \
  --run-id "$run_id" --precision "$precision" --raw-dir "$bulk_run_dir/raw" \
  --bulk-run-dir "$bulk_run_dir" --preflight "$preflight" --post-server "$post_server" \
  --hbm-metadata "$run_dir/hbm_metadata.json" --plot-data "$run_dir/plot_data.json" \
  --output "$run_dir/summary.json"

run_complete=1
echo "smoke PASS: $run_id"
