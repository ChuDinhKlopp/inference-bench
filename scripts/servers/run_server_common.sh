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
attention_backend=${RIVF26_ATTENTION_BACKEND:-FLASHINFER}

# Part 2 uses the same vLLM profiler configuration as the legacy server
# harness.  Keep the legacy variable names as aliases so existing operator
# commands remain valid, while making the output location portable.
enable_torch_profiler=${RIVF26_ENABLE_TORCH_PROFILER:-${ENABLE_TORCH_PROFILER:-0}}
torch_profiler_delay_iters=${RIVF26_TORCH_PROFILER_DELAY_ITERS:-${TORCH_PROFILER_DELAY_ITERS:-20}}
torch_profiler_warmup_iters=${RIVF26_TORCH_PROFILER_WARMUP_ITERS:-${TORCH_PROFILER_WARMUP_ITERS:-10}}
torch_profiler_max_iters=${RIVF26_TORCH_PROFILER_MAX_ITERS:-${TORCH_PROFILER_MAX_ITERS:-250}}
torch_profiler_with_stack=${RIVF26_TORCH_PROFILER_WITH_STACK:-${TORCH_PROFILER_WITH_STACK:-0}}

if [[ "$attention_backend" != "FLASHINFER" ]]; then
  echo "Part 1 requires one A100-compatible attention backend across all arms; expected FLASHINFER" >&2
  exit 2
fi

quant_args=()
case "$precision" in
  w16kv16)
    model_path=${RIVF26_BF16_MODEL_PATH:-/dev/shm/Qwen3.6-35B-A3B}
    kv_dtype=bfloat16
    ;;
  w8kv16)
    model_path=${RIVF26_FP8_MODEL_PATH:-/dev/shm/Qwen3.6-35B-A3B-FP8}
    kv_dtype=bfloat16
    quant_args=(--quantization fp8)
    ;;
  w8kv8)
    model_path=${RIVF26_FP8_MODEL_PATH:-/dev/shm/Qwen3.6-35B-A3B-FP8}
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
  --tensor-parallel-size 4
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

if [[ "$enable_torch_profiler" == "1" || "$enable_torch_profiler" == "2" ]]; then
  if [[ -z "$run_id" ]]; then
    echo "RIVF26_RUN_ID is required when Torch Profiler is enabled" >&2
    exit 2
  fi
  profiler_dir=${RIVF26_TORCH_PROFILER_DIR:-${TORCH_PROFILER_DIR:-$RIVF26_BULK_ROOT/results/part2/$precision/$run_id/raw/torch_profiler}}
  mkdir -p "$profiler_dir"
  echo "Torch Profiler enabled: delay=$torch_profiler_delay_iters warmup=$torch_profiler_warmup_iters active=$torch_profiler_max_iters dir=$profiler_dir"
  cmd+=(
    --enforce-eager
    --profiler-config.profiler=torch
    "--profiler-config.torch_profiler_dir=$profiler_dir"
    "--profiler-config.delay_iterations=$torch_profiler_delay_iters"
    "--profiler-config.max_iterations=$torch_profiler_max_iters"
    "--profiler-config.warmup_iterations=$torch_profiler_warmup_iters"
    "--profiler-config.active_iterations=$torch_profiler_max_iters"
    "--profiler-config.torch_profiler_with_stack=$torch_profiler_with_stack"
    --profiler-config.ignore_frontend=true
  )
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

# Event-only scheduler telemetry.  vLLM writes one JSONL record per
# preemption, immediately before it resets the request's computed-token count.
# Keep this opt-out configurable so legacy/non-RIVF26 launches are unaffected.
if [[ ${RIVF26_ENABLE_PREEMPTION_EVENTS:-1} == 1 ]]; then
  export VLLM_PREEMPTION_EVENTS_PATH=${RIVF26_PREEMPTION_EVENTS_PATH:-$bulk_run_dir/raw/preemption_events.jsonl}
fi
{
  printf '%q ' "${cmd[@]}"
  printf '\n'
} > "$run_dir/logs/server_command.txt"

# Prevent model/tokenizer resolution from creating a hidden remote duplicate.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
# vLLM 0.27's thinking-token budget is implemented by the V1 model runner.
export VLLM_USE_V2_MODEL_RUNNER=0
if [[ "$calculate_kv_scales" == 1 ]]; then
  scale_audit_site=$rivf26_root/scripts/utilities/kv_scale_audit_site
  export RIVF26_KV_SCALE_AUDIT_DIR=${RIVF26_KV_SCALE_AUDIT_DIR:-$bulk_run_dir/raw/kv_scale_audit}
  mkdir -p "$RIVF26_KV_SCALE_AUDIT_DIR"
  export PYTHONPATH=$scale_audit_site${PYTHONPATH:+:$PYTHONPATH}
fi

if [[ -n ${VLLM_PREEMPTION_EVENTS_PATH:-} ]]; then
  echo "Preemption event log: $VLLM_PREEMPTION_EVENTS_PATH"
fi

if [[ "$enable_torch_profiler" == "1" || "$enable_torch_profiler" == "2" ]] && [[ "${RIVF26_PROFILER_AUTO_SHUTDOWN:-0}" == "1" ]]; then
  # vLLM writes profiler_out_0.txt from rank 0 after TorchProfilerWrapper._stop
  # has flushed the trace.  Treat that file as the completion sentinel, then
  # terminate only this server process.  This is opt-in because a profiling
  # capture intentionally ends before a normal benchmark workload completes.
  profiler_sentinel="$profiler_dir/profiler_out_0.txt"
  "${cmd[@]}" &
  server_child=$!
  while kill -0 "$server_child" 2>/dev/null; do
    if [[ -s "$profiler_sentinel" ]]; then
      echo "Torch Profiler completed; shutting down server (sentinel: $profiler_sentinel)"
      kill -TERM "$server_child" 2>/dev/null || true
      break
    fi
    sleep "${RIVF26_PROFILER_POLL_SECONDS:-1}"
  done
  wait "$server_child" 2>/dev/null || true
  exit 0
fi

exec "${cmd[@]}"
