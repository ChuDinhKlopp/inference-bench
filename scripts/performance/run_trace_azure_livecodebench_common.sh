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
source "$rivf26_root/scripts/common/hbm_env.sh"
rivf26_set_scheduler_env

# LCB performance has its own output/reasoning budget.  Clear the generic
# workload guard's inherited values while it establishes arrival-mode and
# reasoning defaults, then restore the audited LCB-specific overrides.
lcb_thinking_budget=${RIVF26_LCB_PERFORMANCE_THINKING_TOKEN_BUDGET:-${RIVF26_THINKING_TOKEN_BUDGET:-6144}}
lcb_max_gen_toks=${RIVF26_LCB_PERFORMANCE_MAX_GEN_TOKS:-${RIVF26_MAX_GEN_TOKS:-10240}}
unset MAX_GEN_TOKS RIVF26_MAX_GEN_TOKS RIVF26_THINKING_TOKEN_BUDGET THINKING_TOKEN_BUDGET
BENCH_ARRIVAL_RATE=azure rivf26_set_workload_env performance
export RIVF26_THINKING_TOKEN_BUDGET=$lcb_thinking_budget THINKING_TOKEN_BUDGET=$lcb_thinking_budget

num_requests=${RIVF26_LCB_PERFORMANCE_REQUESTS:-1055}
[[ "$num_requests" == 1055 ]] || { echo "LiveCodeBench performance requires 1055 requests" >&2; exit 2; }
max_num_seqs=${RIVF26_MAX_NUM_SEQS:-256}
max_gen_toks=$lcb_max_gen_toks
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$HIP_VISIBLE_DEVICES}
trace_csv=${RIVF26_AZURE_LCB_TRACE_CSV:-$rivf26_root/traces/processed/azure_multimodal_bursty_1055.csv}
case "$precision" in
  w8kv16|w8kv8) local_tokenizer=${RIVF26_FP8_MODEL_PATH:-$HOME/models/Qwen3.6-35B-A3B-FP8} ;;
  *) local_tokenizer=${RIVF26_BF16_MODEL_PATH:-/dev/shm/Qwen3.6-35B-A3B} ;;
esac
port=${RIVF26_PORT:-8000}
run_id=${RIVF26_RUN_ID:-$(date -u +%Y%m%d_%H%M%S)_performance_livecodebench_v6_${precision}_mns${max_num_seqs}}
run_dir=${RIVF26_RUN_DIR:-$rivf26_root/results/part1/performance/$run_id}
bulk=${RIVF26_BULK_RUN_DIR:-$RIVF26_BULK_ROOT/results/part1/performance/$run_id}
manifest=$rivf26_root/manifests/$run_id
server_log=$bulk/logs/server.log
client_log=$bulk/logs/client.log

[[ -f "$trace_csv" ]] || { echo "missing 1055-request Azure trace: $trace_csv" >&2; exit 2; }
if [[ ${RIVF26_DRY_RUN:-0} == 1 ]]; then
  echo "precision=$precision requests=$num_requests trace=$trace_csv max_num_seqs=$max_num_seqs max_gen_toks=$max_gen_toks thinking_token_budget=$RIVF26_THINKING_TOKEN_BUDGET max_num_batched_tokens=$RIVF26_MAX_NUM_BATCHED_TOKENS BENCH_ARRIVAL_RATE=azure"
  exit 0
fi
if [[ -e "$run_dir" || -e "$bulk" || -e "$manifest" ]]; then echo "refusing to overwrite $run_id" >&2; exit 2; fi
mkdir -p "$run_dir" "$bulk/logs" "$bulk/raw" "$manifest"
ln -s "$bulk/logs" "$run_dir/logs"
ln -s "$bulk/raw" "$run_dir/raw"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
"$RIVF26_VENV_BIN/python" -c 'import lcb_runner' || { echo 'LiveCodeBench v6 requires lcb_runner' >&2; exit 2; }
"$RIVF26_VENV_BIN/python" "$rivf26_root/scripts/utilities/preflight.py" --run-id "$run_id" --mode performance --precision "$precision" --max-num-seqs "$max_num_seqs" --max-num-batched-tokens "$RIVF26_MAX_NUM_BATCHED_TOKENS" --estimated-output-gib "${RIVF26_ESTIMATED_OUTPUT_GIB:-80}" --safety-reserve-gib "${RIVF26_SAFETY_RESERVE_GIB:-50}" --min-host-available-gib "${RIVF26_RUNTIME_MIN_HOST_AVAILABLE_GIB:-256}" --port "$port" > "$manifest/preflight.txt"

server_launcher=$rivf26_root/scripts/servers/run_server_Qwen3.6-35B-A3B_${precision}.sh
export RIVF26_RUN_ID=$run_id RIVF26_RUN_DIR=$run_dir RIVF26_BULK_RUN_DIR=$bulk RIVF26_PORT=$port RIVF26_MODE=performance RIVF26_MAX_NUM_SEQS=$max_num_seqs
setsid "$server_launcher" > "$server_log" 2>&1 & server_pid=$!
cleanup() { rc=$?; if [[ -n ${server_pid:-} ]]; then rivf26_stop_vllm_server "$server_pid" "$port"; fi; rivf26_stop_hbm_sampler "$bulk"; exit "$rc"; }
trap cleanup EXIT INT TERM
ready=0
for _ in $(seq 1 "${RIVF26_SERVER_READY_ATTEMPTS:-1800}"); do
  if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then ready=1; break; fi
  kill -0 "$server_pid" 2>/dev/null || exit 2
  sleep 1
done
(( ready == 1 )) || exit 2

client=("$RIVF26_VENV_BIN/python" "$parent_root/bench.py" --dataset livecodebench --lcb-release-version release_v6 --num-prompts "$num_requests" --lcb-max-gen-toks "$max_gen_toks" --model Qwen3.6-35B-A3B --tokenizer "$local_tokenizer" --base-url "http://127.0.0.1:$port" --max-concurrency "$max_num_seqs" --temperature 0 --seed 42 --timeout "${RIVF26_REQUEST_TIMEOUT:-86400}" --enable-thinking --azure-trace-csv "$trace_csv" --arrival-scale 1 --skip-evaluation --server-log-file "$server_log" --server-metrics-poll-interval "${RIVF26_SERVER_METRICS_POLL_INTERVAL:-0.2}" --iteration-metrics-prefix "$bulk/raw/bench_results" --save-generations "$bulk/raw/generations.json" --output "$bulk/raw/bench_results.json")
printf '%q ' "${client[@]}" > "$run_dir/client_command.txt"
printf '\n' >> "$run_dir/client_command.txt"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server ready; starting LiveCodeBench v6 Azure performance benchmark"
started=$($RIVF26_VENV_BIN/python -c 'import time; print(time.time())')
BENCH_ARRIVAL_RATE=azure "${client[@]}" 2>&1 | tee "$client_log"
ended=$($RIVF26_VENV_BIN/python -c 'import time; print(time.time())')
# Stop the server now the client is done, and stop the sampler so it finalizes
# hbm_metadata.json with the real sample_rows/gpu_count before anything below
# reads it.
rivf26_stop_vllm_server "$server_pid" "$port"
server_pid=
rivf26_stop_hbm_sampler "$bulk"
kv=$($RIVF26_VENV_BIN/python -c 'import re,sys; t=open(sys.argv[1],errors="replace").read(); m=re.search(r"GPU KV cache size:\s*([0-9,]+) tokens",t,re.I); print(m.group(1).replace(",","") if m else "")' "$server_log")
plot=("$RIVF26_VENV_BIN/python" "$rivf26_root/analysis/build_plot_data.py" --per-request "$bulk/raw/bench_results.per_request.csv" --prometheus "$bulk/raw/bench_results.prometheus_samples.jsonl" --hbm "$bulk/raw/hbm.csv" --output "$run_dir/plot_data.json" --run-id "$run_id" --precision "$precision" --mode performance --max-num-seqs "$max_num_seqs" --max-num-batched-tokens "$RIVF26_MAX_NUM_BATCHED_TOKENS" --bin-seconds "${RIVF26_PLOT_BIN_SECONDS:-1}" --experiment-start-epoch-s "$started")
[[ -n "$kv" ]] && plot+=(--kv-capacity-tokens "$kv")
"${plot[@]}"
"$RIVF26_VENV_BIN/python" "$parent_root/record_e2e_metrics.py" "$bulk/raw/bench_results.json" --csv "$rivf26_root/e2e_metrics_record.csv" --server-log "$server_log" --model-suffix "$precision" --attn-backend TRITON_ATTN
"$RIVF26_VENV_BIN/python" -c 'import json,sys; json.dump({"schema_version":"rivf26.livecodebench_performance.v1","status":"PASS","run_id":sys.argv[1],"precision":sys.argv[2],"dataset":"livecodebench","release":"v6","requests":int(sys.argv[3]),"max_num_seqs":int(sys.argv[4]),"max_gen_toks":int(sys.argv[5]),"arrival_mode":"azure","trace":sys.argv[6]},open(sys.argv[7],"w"),indent=2)' "$run_id" "$precision" "$num_requests" "$max_num_seqs" "$max_gen_toks" "$trace_csv" "$run_dir/manifest.json"
echo "RIVF26 LiveCodeBench v6 Azure performance run completed: $run_dir"
