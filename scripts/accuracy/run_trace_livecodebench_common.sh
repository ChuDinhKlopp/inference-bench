#!/usr/bin/env bash
set -euo pipefail
precision=${1:?precision is required}
case "$precision" in w16kv16|w8kv16|w8kv8|w16kv8) ;; *) exit 2 ;; esac
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
rivf26_root=${RIVF26_ROOT:-$(cd "$script_dir/../.." && pwd)}
parent_root=$(cd "$rivf26_root/.." && pwd)
source "$rivf26_root/scripts/common/venv.sh"
source "$rivf26_root/scripts/common/paths.sh"
source "$rivf26_root/scripts/common/scheduler_env.sh"
source "$rivf26_root/scripts/common/workload_env.sh"
rivf26_set_scheduler_env
rivf26_set_workload_env accuracy
num_requests=${RIVF26_LCB_NUM_PROMPTS:-1055}
max_num_seqs=${RIVF26_MAX_NUM_SEQS:-256}
max_gen_toks=${RIVF26_LCB_MAX_GEN_TOKS:-253952}
case "$precision" in
  w8kv16|w8kv8) local_tokenizer=${RIVF26_FP8_MODEL_PATH:-/dev/shm/Qwen3.6-35B-A3B-FP8} ;;
  *) local_tokenizer=${RIVF26_BF16_MODEL_PATH:-/dev/shm/Qwen3.6-35B-A3B} ;;
esac
port=${RIVF26_PORT:-8000}
run_id=${RIVF26_RUN_ID:-$(date -u +%Y%m%d_%H%M%S)_accuracy_livecodebench_v6_${precision}_mns${max_num_seqs}}
run_dir=${RIVF26_RUN_DIR:-$rivf26_root/results/part1/accuracy/$run_id}
bulk=${RIVF26_BULK_RUN_DIR:-$RIVF26_BULK_ROOT/results/part1/accuracy/$run_id}
manifest=$rivf26_root/manifests/$run_id
server_log=$bulk/logs/server.log
client_log=$bulk/logs/client.log
hbm_prefix=$bulk/raw/hbm
if [[ ${RIVF26_DRY_RUN:-0} == 1 ]]; then
  echo "precision=$precision requests=$num_requests max_num_seqs=$max_num_seqs max_gen_toks=$max_gen_toks thinking_token_budget=$RIVF26_THINKING_TOKEN_BUDGET max_num_batched_tokens=$RIVF26_MAX_NUM_BATCHED_TOKENS BENCH_ARRIVAL_RATE=none"
  exit 0
fi
if [[ -e "$run_dir" || -e "$bulk" || -e "$manifest" ]]; then echo "refusing to overwrite $run_id" >&2; exit 2; fi
mkdir -p "$run_dir" "$bulk/logs" "$bulk/raw" "$manifest"
ln -s "$bulk/logs" "$run_dir/logs"; ln -s "$bulk/raw" "$run_dir/raw"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 BENCH_ARRIVAL_RATE=none
"$RIVF26_VENV_BIN/python" -c 'import lcb_runner' || { echo 'LiveCodeBench v6 requires lcb_runner in the vLLM environment' >&2; exit 2; }
"$RIVF26_VENV_BIN/python" "$rivf26_root/scripts/utilities/preflight.py" --run-id "$run_id" --mode accuracy --precision "$precision" --max-num-seqs "$max_num_seqs" --max-num-batched-tokens "$RIVF26_MAX_NUM_BATCHED_TOKENS" --estimated-output-gib "${RIVF26_ESTIMATED_OUTPUT_GIB:-80}" --safety-reserve-gib "${RIVF26_SAFETY_RESERVE_GIB:-50}" --port "$port" > "$manifest/preflight.txt"
server_launcher=$rivf26_root/scripts/servers/run_server_Qwen3.6-35B-A3B_${precision}.sh
export RIVF26_RUN_ID=$run_id RIVF26_RUN_DIR=$run_dir RIVF26_BULK_RUN_DIR=$bulk RIVF26_PORT=$port RIVF26_MODE=accuracy RIVF26_MAX_NUM_SEQS=$max_num_seqs
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting server"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server log: $server_log"
setsid "$server_launcher" > "$server_log" 2>&1 & server_pid=$!
cleanup() { rc=$?; kill -TERM -- "-$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; exit "$rc"; }
trap cleanup EXIT INT TERM
for _ in $(seq 1 "${RIVF26_SERVER_READY_ATTEMPTS:-1800}"); do
  curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break
  kill -0 "$server_pid" 2>/dev/null || exit 2
  sleep 1
done
curl -fsS "http://127.0.0.1:$port/health" >/dev/null || exit 2
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server is ready"
client=("$RIVF26_VENV_BIN/python" "$parent_root/bench.py" --dataset livecodebench --lcb-release-version release_v6 --num-prompts "$num_requests" --lcb-max-gen-toks "$max_gen_toks" --model Qwen3.6-35B-A3B --tokenizer "$local_tokenizer" --base-url "http://127.0.0.1:$port" --max-concurrency "$max_num_seqs" --temperature 0 --seed 42 --timeout "${RIVF26_REQUEST_TIMEOUT:-86400}" --enable-thinking --server-log-file "$server_log" --server-metrics-poll-interval "${RIVF26_SERVER_METRICS_POLL_INTERVAL:-0.2}" --iteration-metrics-prefix "$bulk/raw/bench_results" --save-generations "$bulk/raw/generations.json" --output "$bulk/raw/bench_results.json")
printf '%q ' "${client[@]}" > "$run_dir/client_command.txt"
printf '\n' >> "$run_dir/client_command.txt"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting benchmark"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Benchmark log: $client_log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Benchmark output: $bulk/raw/bench_results.json"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Benchmark dataset: livecodebench release_v6"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Benchmark arrival mode: none"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Benchmark max concurrency: $max_num_seqs"
started=$($RIVF26_VENV_BIN/python -c 'import time; print(time.time())')
"$rivf26_root/scripts/monitoring/capture_hbm.sh" "$hbm_prefix" "${client[@]}" 2>&1 | tee "$client_log"
ended=$($RIVF26_VENV_BIN/python -c 'import time; print(time.time())')
"$RIVF26_VENV_BIN/python" "$rivf26_root/scripts/monitoring/parse_nsys_hbm.py" "$hbm_prefix.nsys-rep" "$bulk/raw/hbm.csv" --metadata-json "$run_dir/hbm_metadata.json" --experiment-start-epoch-s "$started" || true
kv=$($RIVF26_VENV_BIN/python -c 'import re,sys; t=open(sys.argv[1],errors="replace").read(); m=re.search(r"GPU KV cache size:\s*([0-9,]+) tokens",t,re.I); print(m.group(1).replace(",","") if m else "")' "$server_log")
plot=("$RIVF26_VENV_BIN/python" "$rivf26_root/analysis/build_plot_data.py" --per-request "$bulk/raw/bench_results.per_request.csv" --prometheus "$bulk/raw/bench_results.prometheus_samples.jsonl" --hbm "$bulk/raw/hbm.csv" --output "$run_dir/plot_data.json" --run-id "$run_id" --precision "$precision" --mode accuracy --max-num-seqs "$max_num_seqs" --max-num-batched-tokens "$RIVF26_MAX_NUM_BATCHED_TOKENS" --experiment-start-epoch-s "$started")
[[ -n "$kv" ]] && plot+=(--kv-capacity-tokens "$kv")
"${plot[@]}"
record_script=$parent_root/record_e2e_metrics.py
e2e_csv=$rivf26_root/e2e_metrics_record.csv
[[ -f "$record_script" ]] || { echo "missing legacy recorder: $record_script" >&2; exit 2; }
"$RIVF26_VENV_BIN/python" "$record_script" "$bulk/raw/bench_results.json" \
  --csv "$e2e_csv" \
  --server-log "$server_log" \
  --model-suffix "$precision" \
  --attn-backend FLASHINFER
"$RIVF26_VENV_BIN/python" -c 'import json,sys; json.dump({"schema_version":"rivf26.livecodebench_run.v1","status":"PASS","run_id":sys.argv[1],"precision":sys.argv[2],"dataset":"livecodebench","release":"v6","requests":int(sys.argv[3]),"max_num_seqs":int(sys.argv[4]),"max_gen_toks":int(sys.argv[5]),"arrival_mode":"none"},open(sys.argv[6],"w"),indent=2)' "$run_id" "$precision" "$num_requests" "$max_num_seqs" "$max_gen_toks" "$run_dir/manifest.json"
echo "RIVF26 LiveCodeBench v6 run completed: $run_dir"
