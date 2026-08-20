#!/usr/bin/env bash
# LiveCodeBench v6 under the Azure bursty trace, for the Part 3 KV-quantization study.
#
#   scripts/kvquant/run_azure_livecodebench_<model_key>_<kv_dtype_key>.sh
#   e.g. scripts/kvquant/run_azure_livecodebench_qwen3_30b_kv4.sh
#
# Mirrors scripts/performance/run_trace_azure_livecodebench_common.sh, with three
# deliberate differences:
#
#   1. The arm is (model_key, kv_dtype_key), not a Part 1 precision. Weight precision is
#      never swept -- each checkpoint runs in its native format and only --kv-cache-dtype
#      changes, which is what makes this a KV study rather than a precision study.
#   2. The attention backend is pinned to whatever the spec says (TRITON_ATTN for every
#      primary arm) and the runtime is validated against it before any measurement.
#   3. It does NOT call scripts/utilities/preflight.py: that resolves model paths from
#      configs/precision_configs.json, which only knows the four Qwen3.6 Part 1 arms.
#      The essential resource guards are inlined below instead.
#
# LiveCodeBench IS the heavy-decode regime: measured ISL ~0.6-1k against OSL ~22-32k.
set -euo pipefail

model_key=${1:?model key is required}
kv_key=${2:?kv dtype key is required}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
rivf26_root=${RIVF26_ROOT:-$(cd "$script_dir/../.." && pwd)}
parent_root=$(cd "$rivf26_root/.." && pwd)
spec="$rivf26_root/configs/part3_kvquant.json"

source "$rivf26_root/scripts/common/venv.sh"
source "$rivf26_root/scripts/common/paths.sh"
source "$rivf26_root/scripts/common/scheduler_env.sh"
source "$rivf26_root/scripts/common/workload_env.sh"
rivf26_set_scheduler_env

# LCB decode-heavy has its own output/reasoning budget. Clear the generic guard's
# inherited values while it establishes arrival mode, then restore ours.
lcb_thinking_budget=${RIVF26_KVQ_LCB_THINKING_TOKEN_BUDGET:-${RIVF26_THINKING_TOKEN_BUDGET:-6144}}
lcb_max_gen_toks=${RIVF26_KVQ_LCB_MAX_GEN_TOKS:-32768}
unset MAX_GEN_TOKS RIVF26_MAX_GEN_TOKS RIVF26_THINKING_TOKEN_BUDGET THINKING_TOKEN_BUDGET
BENCH_ARRIVAL_RATE=azure rivf26_set_workload_env performance
export RIVF26_THINKING_TOKEN_BUDGET=$lcb_thinking_budget THINKING_TOKEN_BUDGET=$lcb_thinking_budget

# Resolve the arm from the study spec. Capture explicitly: process substitution does not
# propagate the child's exit status, and this enforces structural exclusions such as
# "TurboQuant cannot serve GPT-OSS" (no sink support), which must fail closed.
if ! resolved=$("$RIVF26_VENV_BIN/python" - "$spec" "$model_key" "$kv_key" <<'PY'
import json, sys
spec = json.load(open(sys.argv[1]))
mk, kk = sys.argv[2], sys.argv[3]
if mk not in spec["models"]:
    sys.exit(f"unknown model {mk!r}; have {list(spec['models'])}")
if kk not in spec["kv_dtypes"]:
    sys.exit(f"unknown kv dtype {kk!r}; have {list(spec['kv_dtypes'])}")
m, d = spec["models"][mk], spec["kv_dtypes"][kk]
allowed = d.get("models")
if allowed is not None and mk not in allowed:
    sys.exit(f"{kk} is not supported for {mk} (see spec notes); refusing to run")
print(m["path_default"], d["backend"], d["cli"], spec["common"]["max_model_len"])
PY
); then
  exit 1
fi
read -r model_path backend kv_cli spec_mml <<<"$resolved"

num_requests=${RIVF26_LCB_PERFORMANCE_REQUESTS:-1055}
[[ "$num_requests" == 1055 ]] || { echo "LiveCodeBench performance requires 1055 requests" >&2; exit 2; }
max_num_seqs=${RIVF26_MAX_NUM_SEQS:-128}
max_model_len=${RIVF26_MAX_MODEL_LEN:-$spec_mml}
trace_csv=${RIVF26_AZURE_LCB_TRACE_CSV:-$rivf26_root/traces/processed/azure_multimodal_bursty_1055.csv}
port=${RIVF26_PORT:-8000}
arm="${model_key}_${kv_key}"

run_id=${RIVF26_RUN_ID:-$(date -u +%Y%m%d_%H%M%S)_part3_livecodebench_v6_${arm}_mns${max_num_seqs}}
run_dir=${RIVF26_RUN_DIR:-$rivf26_root/results/part3/$run_id}
bulk=${RIVF26_BULK_RUN_DIR:-$RIVF26_BULK_ROOT/results/part3/$run_id}
manifest=$rivf26_root/manifests/$run_id
server_log=$bulk/logs/server.log
client_log=$bulk/logs/client.log
hbm_prefix=$bulk/raw/hbm

[[ -f "$trace_csv" ]] || { echo "missing 1055-request Azure trace: $trace_csv" >&2; exit 2; }
[[ -d "$model_path" ]] || { echo "model directory not found: $model_path" >&2; exit 2; }

if [[ ${RIVF26_DRY_RUN:-0} == 1 ]]; then
  echo "arm=$arm model_path=$model_path backend=$backend kv_cache_dtype=$kv_cli requests=$num_requests trace=$trace_csv max_num_seqs=$max_num_seqs max_model_len=$max_model_len max_gen_toks=$lcb_max_gen_toks thinking_token_budget=$RIVF26_THINKING_TOKEN_BUDGET max_num_batched_tokens=$RIVF26_MAX_NUM_BATCHED_TOKENS BENCH_ARRIVAL_RATE=azure"
  exit 0
fi

if [[ -e "$run_dir" || -e "$bulk" || -e "$manifest" ]]; then
  echo "refusing to overwrite $run_id" >&2; exit 2
fi
mkdir -p "$run_dir" "$bulk/logs" "$bulk/raw" "$manifest"
ln -s "$bulk/logs" "$run_dir/logs"
ln -s "$bulk/raw" "$run_dir/raw"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

"$RIVF26_VENV_BIN/python" -c 'import lcb_runner' || {
  echo 'LiveCodeBench v6 requires lcb_runner in the vLLM environment' >&2; exit 2; }

# Inlined resource guard (see header note 3).
free_gib=$(df -BG --output=avail "$RIVF26_BULK_ROOT" | tail -1 | tr -dc '0-9')
(( free_gib >= ${RIVF26_KVQ_MIN_FREE_GIB:-120} )) || {
  echo "only ${free_gib}GiB free under $RIVF26_BULK_ROOT; need ${RIVF26_KVQ_MIN_FREE_GIB:-120}GiB" >&2; exit 2; }
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$manifest/gpu_before.txt"
fi
ss -ltn 2>/dev/null | grep -q ":$port " && { echo "port $port already in use" >&2; exit 2; }

export RIVF26_RUN_ID=$run_id RIVF26_RUN_DIR=$run_dir RIVF26_BULK_RUN_DIR=$bulk \
       RIVF26_PORT=$port RIVF26_MODE=performance RIVF26_MAX_NUM_SEQS=$max_num_seqs \
       RIVF26_MAX_MODEL_LEN=$max_model_len

echo "[$(date '+%Y-%m-%d %H:%M:%S')] arm=$arm backend=$backend kv=$kv_cli mns=$max_num_seqs"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server log: $server_log"
setsid "$script_dir/run_server.sh" "$model_key" "$kv_key" > "$server_log" 2>&1 & server_pid=$!
cleanup() { rc=$?; kill -TERM -- "-$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; exit "$rc"; }
trap cleanup EXIT INT TERM

ready=0
for _ in $(seq 1 "${RIVF26_SERVER_READY_ATTEMPTS:-1800}"); do
  if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then ready=1; break; fi
  kill -0 "$server_pid" 2>/dev/null || exit 2
  sleep 1
done
(( ready == 1 )) || exit 2

# A dry run proves the CLI string; it does not prove the engine used that backend. vLLM
# falls back silently, and a silent fallback would turn the controlled variable back into
# a confound. Fail closed before spending hours of GPU time.
"$RIVF26_VENV_BIN/python" "$script_dir/validate_runtime.py" \
  --server-log "$server_log" --expect-backend "$backend" --expect-kv-dtype "$kv_cli" \
  --out "$run_dir/runtime_validation.json" || {
  echo "runtime validation FAILED -- refusing to measure $arm" >&2; exit 2; }

client=("$RIVF26_VENV_BIN/python" "$parent_root/bench.py" --dataset livecodebench
  --lcb-release-version release_v6 --num-prompts "$num_requests"
  --lcb-max-gen-toks "$lcb_max_gen_toks" --model "$model_key" --tokenizer "$model_path"
  --base-url "http://127.0.0.1:$port" --max-concurrency "$max_num_seqs"
  --temperature 0 --seed 42 --timeout "${RIVF26_REQUEST_TIMEOUT:-86400}" --enable-thinking
  --azure-trace-csv "$trace_csv" --arrival-scale 1 --skip-evaluation
  --server-log-file "$server_log"
  --server-metrics-poll-interval "${RIVF26_SERVER_METRICS_POLL_INTERVAL:-0.2}"
  --iteration-metrics-prefix "$bulk/raw/bench_results"
  --save-generations "$bulk/raw/generations.json" --output "$bulk/raw/bench_results.json")
printf '%q ' "${client[@]}" > "$run_dir/client_command.txt"
printf '\n' >> "$run_dir/client_command.txt"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server ready; starting LiveCodeBench v6 Azure run"
started=$($RIVF26_VENV_BIN/python -c 'import time; print(time.time())')
BENCH_ARRIVAL_RATE=azure "$rivf26_root/scripts/monitoring/capture_hbm.sh" "$hbm_prefix" "${client[@]}" 2>&1 | tee "$client_log"
ended=$($RIVF26_VENV_BIN/python -c 'import time; print(time.time())')

"$RIVF26_VENV_BIN/python" "$rivf26_root/scripts/monitoring/parse_nsys_hbm.py" \
  "$hbm_prefix.nsys-rep" "$bulk/raw/hbm.csv" --metadata-json "$run_dir/hbm_metadata.json" \
  --experiment-start-epoch-s "$started" || true

kv=$($RIVF26_VENV_BIN/python -c 'import re,sys; t=open(sys.argv[1],errors="replace").read(); m=re.search(r"GPU KV cache size:\s*([0-9,]+) tokens",t,re.I); print(m.group(1).replace(",","") if m else "")' "$server_log")
plot=("$RIVF26_VENV_BIN/python" "$rivf26_root/analysis/build_plot_data.py"
  --per-request "$bulk/raw/bench_results.per_request.csv"
  --prometheus "$bulk/raw/bench_results.prometheus_samples.jsonl"
  --hbm "$bulk/raw/hbm.csv" --output "$run_dir/plot_data.json" --run-id "$run_id"
  --precision "$arm" --mode performance --max-num-seqs "$max_num_seqs"
  --max-num-batched-tokens "$RIVF26_MAX_NUM_BATCHED_TOKENS"
  --bin-seconds "${RIVF26_PLOT_BIN_SECONDS:-1}" --experiment-start-epoch-s "$started")
[[ -n "$kv" ]] && plot+=(--kv-capacity-tokens "$kv")
"${plot[@]}"

"$RIVF26_VENV_BIN/python" "$parent_root/record_e2e_metrics.py" "$bulk/raw/bench_results.json" \
  --csv "$rivf26_root/e2e_metrics_record.csv" --server-log "$server_log" \
  --model-suffix "$arm" --attn-backend "$backend"

"$RIVF26_VENV_BIN/python" -c 'import json,sys; json.dump({"schema_version":"rivf26.kvquant_livecodebench.v1","status":"PASS","run_id":sys.argv[1],"arm":sys.argv[2],"model_key":sys.argv[3],"kv_dtype_key":sys.argv[4],"kv_cache_dtype":sys.argv[5],"attention_backend":sys.argv[6],"dataset":"livecodebench","release":"v6","regime":"decode_heavy","requests":int(sys.argv[7]),"max_num_seqs":int(sys.argv[8]),"max_gen_toks":int(sys.argv[9]),"arrival_mode":"azure","trace":sys.argv[10],"kv_capacity_tokens":(int(sys.argv[11]) if sys.argv[11] else None)},open(sys.argv[12],"w"),indent=2)' \
  "$run_id" "$arm" "$model_key" "$kv_key" "$kv_cli" "$backend" "$num_requests" \
  "$max_num_seqs" "$lcb_max_gen_toks" "$trace_csv" "$kv" "$run_dir/manifest.json"

echo "RIVF26 Part 3 LiveCodeBench v6 Azure run completed: $run_dir"
