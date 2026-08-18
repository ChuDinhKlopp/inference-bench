#!/usr/bin/env bash
set -euo pipefail

# Short Part 2 capture.  The server-side profiler configuration is intentionally
# passed through run_server_common.sh so it remains identical to the legacy
# run_server_Qwen3-4B_fp16.sh implementation.
precision=${1:?usage: $0 <w16kv16|w8kv16|w8kv8|w16kv8>}
case "$precision" in w16kv16|w8kv16|w8kv8|w16kv8) ;; *) echo "unsupported precision: $precision" >&2; exit 2 ;; esac

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
rivf26_root=${RIVF26_ROOT:-$(cd -- "$script_dir/../.." && pwd)}
source "$rivf26_root/scripts/common/venv.sh"
source "$rivf26_root/scripts/common/paths.sh"
source "$rivf26_root/scripts/common/scheduler_env.sh"
rivf26_set_scheduler_env

run_id=${RIVF26_RUN_ID:-$(date -u +%Y%m%d_%H%M%S)_part2_torch_${precision}}
port=${RIVF26_PORT:-8000}
run_dir=${RIVF26_RUN_DIR:-$rivf26_root/results/part2/$precision/$run_id}
bulk_run_dir=${RIVF26_BULK_RUN_DIR:-$RIVF26_BULK_ROOT/results/part2/$precision/$run_id}
manifest_dir=$rivf26_root/manifests/$run_id
server_log=$bulk_run_dir/logs/server.log

mkdir -p "$run_dir" "$bulk_run_dir/logs" "$bulk_run_dir/raw" "$manifest_dir"
for kind in logs raw; do
  ln -s "$bulk_run_dir/$kind" "$run_dir/$kind" 2>/dev/null || true
done

"$RIVF26_VENV_BIN/python" "$rivf26_root/scripts/utilities/preflight.py" \
  --run-id "$run_id" --mode smoke --precision "$precision" \
  --max-num-seqs "${RIVF26_MAX_NUM_SEQS:-1}" \
  --max-num-batched-tokens "$RIVF26_MAX_NUM_BATCHED_TOKENS" \
  --estimated-output-gib "${RIVF26_ESTIMATED_OUTPUT_GIB:-5}" \
  --safety-reserve-gib "${RIVF26_SAFETY_RESERVE_GIB:-5}" --port "$port" \
  > "$manifest_dir/preflight.log"

export RIVF26_RUN_ID="$run_id" RIVF26_MODE=part2 RIVF26_RUN_DIR="$run_dir"
export RIVF26_BULK_RUN_DIR="$bulk_run_dir" RIVF26_PREFLIGHT_JSON="$manifest_dir/preflight.json"
export RIVF26_PORT="$port" RIVF26_MAX_NUM_SEQS="${RIVF26_MAX_NUM_SEQS:-1}"
export RIVF26_ENABLE_TORCH_PROFILER=1
export RIVF26_PROFILER_AUTO_SHUTDOWN="${RIVF26_PROFILER_AUTO_SHUTDOWN:-1}"
export RIVF26_TORCH_PROFILER_DELAY_ITERS="${RIVF26_TORCH_PROFILER_DELAY_ITERS:-20}"
export RIVF26_TORCH_PROFILER_WARMUP_ITERS="${RIVF26_TORCH_PROFILER_WARMUP_ITERS:-10}"
export RIVF26_TORCH_PROFILER_MAX_ITERS="${RIVF26_TORCH_PROFILER_MAX_ITERS:-250}"
export RIVF26_TORCH_PROFILER_WITH_STACK="${RIVF26_TORCH_PROFILER_WITH_STACK:-0}"
export RIVF26_TORCH_PROFILER_DIR="${RIVF26_TORCH_PROFILER_DIR:-$bulk_run_dir/raw/torch_profiler}"

server_launcher="$rivf26_root/scripts/servers/run_server_Qwen3.6-35B-A3B_${precision}.sh"
tokenizer_path=${RIVF26_BF16_MODEL_PATH:-/dev/shm/Qwen3.6-35B-A3B}
if [[ "$precision" == w8* ]]; then
  tokenizer_path=${RIVF26_FP8_MODEL_PATH:-/dev/shm/Qwen3.6-35B-A3B-FP8}
fi
setsid "$server_launcher" > "$server_log" 2>&1 &
server_pid=$!
cleanup() { kill -TERM -- "-$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

for _ in {1..900}; do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -100 "$server_log" >&2 || true
    exit 2
  fi
  if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then break; fi
  sleep 1
done

echo "Starting Torch Profiler via /start_profile"
curl -fsS -X POST "http://127.0.0.1:$port/start_profile" >/dev/null

# Four short requests provide enough decode iterations for the 250-iteration
# capture while retaining the existing timing/metrics client infrastructure.
set +e
"$RIVF26_VENV_BIN/python" "$rivf26_root/scripts/utilities/smoke_client.py" \
  --output-dir "$run_dir" --base-url "http://127.0.0.1:$port" \
  --model-path "$tokenizer_path" \
  --server-log-file "$server_log" --max-gen-toks "${RIVF26_PROFILE_MAX_GEN_TOKS:-512}" \
  --max-concurrency "${RIVF26_PROFILE_MAX_CONCURRENCY:-1}" \
  --metrics-poll-interval "${RIVF26_SERVER_METRICS_POLL_INTERVAL:-0.2}"
client_rc=$?
set -e

profiler_sentinel="$RIVF26_TORCH_PROFILER_DIR/profiler_out_0.txt"
if (( client_rc != 0 )) && [[ ! -s "$profiler_sentinel" ]]; then
  echo "profile client failed before profiler completion (exit=$client_rc)" >&2
  exit "$client_rc"
fi

printf '{"status":"PASS","run_id":"%s","precision":"%s","profiler_dir":"%s","active_iterations":%s}\n' \
  "$run_id" "$precision" "$RIVF26_TORCH_PROFILER_DIR" "$RIVF26_TORCH_PROFILER_MAX_ITERS" \
  > "$run_dir/part2_summary.json"
echo "Part 2 Torch Profiler capture complete: $RIVF26_TORCH_PROFILER_DIR"
