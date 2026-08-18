#!/usr/bin/env bash
set -euo pipefail

precision=${1:?precision is required}
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
run_id=${RIVF26_RUN_ID:-}
port=${RIVF26_PORT:-8000}
max_num_seqs=${RIVF26_MAX_NUM_SEQS:-24}
max_model_len=${RIVF26_MAX_MODEL_LEN:-262144}
gpu_memory_utilization=${RIVF26_GPU_MEMORY_UTILIZATION:-0.90}
vllm_python=$RIVF26_VENV_BIN/python
attention_backend=${RIVF26_ATTENTION_BACKEND:-TRITON_ATTN}

if [[ "$attention_backend" != "TRITON_ATTN" ]]; then
  echo "Part 1 requires one MI250-compatible attention backend across all arms; expected TRITON_ATTN" >&2
  exit 2
fi

quant_args=()
case "$precision" in
  w16kv16)
    model_path=${RIVF26_BF16_MODEL_PATH:-/dev/shm/Qwen3.6-35B-A3B}
    kv_dtype=bfloat16
    ;;
  w8kv16)
    model_path=${RIVF26_FP8_MODEL_PATH:-$HOME/models/Qwen3.6-35B-A3B-FP8}
    kv_dtype=bfloat16
    quant_args=(--quantization fp8)
    ;;
  w8kv8)
    model_path=${RIVF26_FP8_MODEL_PATH:-$HOME/models/Qwen3.6-35B-A3B-FP8}
    kv_dtype=fp8
    quant_args=(--quantization fp8)
    ;;
  w16kv8)
    model_path=${RIVF26_BF16_MODEL_PATH:-/dev/shm/Qwen3.6-35B-A3B}
    kv_dtype=fp8
    ;;
esac

calculate_kv_scales=${RIVF26_CALCULATE_KV_SCALES:-0}
if [[ "$calculate_kv_scales" != 0 && "$calculate_kv_scales" != 1 ]]; then
  echo "RIVF26_CALCULATE_KV_SCALES must be 0 or 1" >&2
  exit 2
fi
if [[ "$calculate_kv_scales" == 1 && "$kv_dtype" != fp8 ]]; then
  echo "runtime KV-scale calculation is only valid for a KV8 arm" >&2
  exit 2
fi

cmd=(
  "$vllm_python" -m vllm.entrypoints.cli.main serve "$model_path"
  --tokenizer "$model_path"
  --served-model-name Qwen3.6-35B-A3B
  --host 127.0.0.1
  --port "$port"
  --tensor-parallel-size 2
  --max-model-len "$max_model_len"
  --max-num-seqs "$max_num_seqs"
  --gpu-memory-utilization "$gpu_memory_utilization"
  --kv-cache-dtype "$kv_dtype"
  --attention-backend "$attention_backend"
  --reasoning-parser qwen3
  --reasoning-config '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}'
  --enable-logging-iteration-details
  "${quant_args[@]}"
)

if [[ "$calculate_kv_scales" == 1 ]]; then
  cmd+=(--calculate-kv-scales)
fi

cmd+=(--max-num-batched-tokens "$RIVF26_MAX_NUM_BATCHED_TOKENS")
if [[ ${RIVF26_ENFORCE_EAGER:-0} == 1 ]]; then
  # Diagnostic-only: disables CUDA graph capture/replay to isolate whether the
  # repeated "Worker died unexpectedly (exit code: None)" crash is graph-related.
  cmd+=(--enforce-eager)
fi
if [[ -n ${RIVF26_EXTRA_SERVER_ARGS:-} ]]; then
  echo "RIVF26_EXTRA_SERVER_ARGS is intentionally unsupported; add audited arguments to this launcher" >&2
  exit 2
fi

printf 'RIVF26 server command:'
printf ' %q' "${cmd[@]}"
printf '\n'
if [[ ${RIVF26_DRY_RUN:-0} == 1 ]]; then
  exit 0
fi

if [[ -z "$run_id" ]]; then
  echo "RIVF26_RUN_ID is required" >&2
  exit 2
fi
preflight=${RIVF26_PREFLIGHT_JSON:-$rivf26_root/manifests/$run_id/preflight.json}
if [[ ! -f "$preflight" ]]; then
  echo "missing Stage A preflight: $preflight" >&2
  exit 2
fi
if ! "$RIVF26_VENV_BIN/python" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["status"] == "PASS" else 2)' "$preflight"; then
  echo "Stage A preflight did not PASS: $preflight" >&2
  exit 2
fi
if [[ ! -d "$model_path" ]]; then
  echo "local model directory is missing: $model_path" >&2
  exit 2
fi
if [[ ! -x "$vllm_python" ]]; then
  echo "vLLM Python is not executable: $vllm_python" >&2
  exit 2
fi

mode=${RIVF26_MODE:-accuracy}
run_dir=${RIVF26_RUN_DIR:-$rivf26_root/results/part1/$mode/$run_id}
bulk_run_dir=${RIVF26_BULK_RUN_DIR:-$RIVF26_BULK_ROOT/results/part1/$mode/$run_id}
mkdir -p "$run_dir" "$bulk_run_dir/logs" "$bulk_run_dir/raw"
for artifact_kind in logs raw; do
  link_path=$run_dir/$artifact_kind
  target_path=$bulk_run_dir/$artifact_kind
  if [[ -L "$link_path" ]]; then
    if [[ $(readlink -f "$link_path") != $(readlink -f "$target_path") ]]; then
      echo "existing $link_path points to the wrong bulk artifact directory" >&2
      exit 2
    fi
  elif [[ -e "$link_path" ]]; then
    echo "refusing to replace existing non-symlink artifact path: $link_path" >&2
    exit 2
  else
    ln -s "$target_path" "$link_path"
  fi
done
{
  printf '%q ' "${cmd[@]}"
  printf '\n'
} > "$run_dir/logs/server_command.txt"

# Prevent model/tokenizer resolution from creating a hidden remote duplicate.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$HIP_VISIBLE_DEVICES}
# vLLM 0.27's thinking-token budget is implemented by the V1 model runner.
export VLLM_USE_V2_MODEL_RUNNER=0
if [[ "$calculate_kv_scales" == 1 ]]; then
  scale_audit_site=$rivf26_root/scripts/utilities/kv_scale_audit_site
  export RIVF26_KV_SCALE_AUDIT_DIR=${RIVF26_KV_SCALE_AUDIT_DIR:-$bulk_run_dir/raw/kv_scale_audit}
  mkdir -p "$RIVF26_KV_SCALE_AUDIT_DIR"
  export PYTHONPATH=$scale_audit_site${PYTHONPATH:+:$PYTHONPATH}
fi

if [[ ${RIVF26_DISABLE_ROCPROF:-0} == 1 ]]; then
  # Diagnostic-only: skip rocprof instrumentation entirely (no HBM capture) to
  # test whether rocprof's per-kernel hooking is implicated in the recurring
  # "Worker died unexpectedly" crash. Not for real runs -- no HBM data.
  echo "WARNING: RIVF26_DISABLE_ROCPROF=1 -- running without rocprof; no HBM data will be captured" >&2
  exec "${cmd[@]}"
fi

# HBM read/write bandwidth is captured by wrapping the server itself with classic
# rocprof (v1), not rocprofv3/amdsmi: rocprofv3's hardware-counter injection SIGABRTs
# (launch mode) or silently no-ops (attach mode) on any process that imports this
# environment's PyTorch, and amdsmi/amd-smi/rocm-smi's UMC activity telemetry is
# frozen on this host under any workload or privilege level. Classic rocprof avoids
# the collision and gives real per-kernel FETCH_SIZE/WRITE_SIZE counters (see
# scripts/monitoring/parse_rocprof_hbm.py). rocprof only writes its output CSV if its
# own wrapper process is left running long enough to see the wrapped vLLM process
# exit on its own -- callers MUST NOT process-group-kill this server; they must send
# SIGTERM to the vLLM child PID specifically and let this rocprof wrapper finish its
# post-processing and exit by itself.
hbm_dir=$bulk_run_dir/raw/hbm_kernel
mkdir -p "$hbm_dir"
cp "$rivf26_root/scripts/monitoring/rocm_hbm_rpl_rc.xml" "$hbm_dir/rpl_rc.xml"
"$RIVF26_VENV_BIN/python" -c \
  'import time,json,sys; json.dump({"reference_epoch_s": time.time(), "reference_monotonic_ns": time.monotonic_ns()}, open(sys.argv[1], "w"))' \
  "$hbm_dir/reference.json"
cd "$hbm_dir"

# Classic rocprof re-evaluates its wrapped command with `eval "$APP_CMD"` after a
# naive per-argument \"$arg\" quote-wrap. That wrapping breaks (and rocprof
# misparses the vLLM command entirely, e.g. treating the --reasoning-config JSON's
# "<think>" as shell input redirection) for any argument that itself contains
# double quotes, which --reasoning-config's JSON value always does. Writing the
# real invocation to a script file with bash's own %q-quoting and having rocprof
# wrap that (a single argument-free command) avoids rocprof's eval reconstruction
# ever seeing the problematic characters.
launch_script=$hbm_dir/launch_vllm.sh
{
  printf '#!/usr/bin/env bash\n'
  printf 'exec'
  printf ' %q' "${cmd[@]}"
  printf '\n'
} > "$launch_script"
chmod +x "$launch_script"

exec rocprof -i "$rivf26_root/scripts/monitoring/rocm_hbm_counters.txt" \
  -o "$hbm_dir/kernel_dispatches.csv" \
  -d "$hbm_dir/rocprof_data" \
  "$launch_script"
