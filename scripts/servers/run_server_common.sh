#!/usr/bin/env bash
set -euo pipefail

model=${1:?model is required}
precision=${2:?precision is required}
case "$model" in
  Qwen3.6-35B-A3B|gpt-oss-120b) ;;
  *) echo "unsupported model: $model" >&2; exit 2 ;;
esac
case "$precision" in
  w16kv16|w8kv16|w8kv8|w16kv8) ;;
  *) echo "unsupported precision: $precision" >&2; exit 2 ;;
esac
if [[ "$model" == "gpt-oss-120b" && "$precision" != "w16kv16" && "$precision" != "w16kv8" ]]; then
  echo "gpt-oss-120b only supports w16kv16/w16kv8 -- it ships a single native mxfp4 weight checkpoint, so there is no separate w8 weight-quant axis to select" >&2
  exit 2
fi
# preflight.py/post_server_validate.py key their precision-config lookups by
# a single string; namespace gpt-oss-120b's so it never collides with Qwen's
# own w16kv16/w16kv8 entries, and so profiler-output paths stay distinct.
if [[ "$model" == "gpt-oss-120b" ]]; then
  full_precision="gpt-oss-120b_${precision}"
else
  full_precision=$precision
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
rivf26_root=${RIVF26_ROOT:-$(cd -- "$script_dir/../.." && pwd)}
source "$rivf26_root/scripts/common/venv.sh"
source "$rivf26_root/scripts/common/paths.sh"
source "$rivf26_root/scripts/common/scheduler_env.sh"
rivf26_set_scheduler_env
run_id=${RIVF26_RUN_ID:-}
port=${RIVF26_PORT:-8000}
max_num_seqs=${RIVF26_MAX_NUM_SEQS:-24}
gpu_memory_utilization=${RIVF26_GPU_MEMORY_UTILIZATION:-0.90}
vllm_python=$RIVF26_VENV_BIN/python
attention_backend=${RIVF26_ATTENTION_BACKEND:-TRITON_ATTN}

# Part 2 Torch Profiler configuration. Legacy variable names (without the
# RIVF26_ prefix) are honored as a fallback so operator commands ported from
# the pre-RIVF26 harness keep working.
enable_torch_profiler=${RIVF26_ENABLE_TORCH_PROFILER:-${ENABLE_TORCH_PROFILER:-0}}
torch_profiler_delay_iters=${RIVF26_TORCH_PROFILER_DELAY_ITERS:-${TORCH_PROFILER_DELAY_ITERS:-20}}
torch_profiler_warmup_iters=${RIVF26_TORCH_PROFILER_WARMUP_ITERS:-${TORCH_PROFILER_WARMUP_ITERS:-10}}
torch_profiler_max_iters=${RIVF26_TORCH_PROFILER_MAX_ITERS:-${TORCH_PROFILER_MAX_ITERS:-250}}
torch_profiler_with_stack=${RIVF26_TORCH_PROFILER_WITH_STACK:-${TORCH_PROFILER_WITH_STACK:-0}}

if [[ "$attention_backend" != "TRITON_ATTN" ]]; then
  echo "Part 1 requires one MI250-compatible attention backend across all arms; expected TRITON_ATTN" >&2
  exit 2
fi

# The KV-cache axis (kv_dtype) is the same vocabulary for every model:
# w16kv16/w16kv8 never touch weights, w8kv16/w8kv8 additionally select the
# FP8 weight checkpoint + --quantization fp8 (Qwen only -- rejected above for
# gpt-oss-120b, which has no separate weight-quant checkpoint to select).
quant_args=()
case "$precision" in
  w16kv16) kv_dtype=bfloat16 ;;
  w8kv16)  kv_dtype=bfloat16; quant_args=(--quantization fp8) ;;
  w8kv8)   kv_dtype=fp8;      quant_args=(--quantization fp8) ;;
  w16kv8)  kv_dtype=fp8 ;;
esac

# Model identity: path, served name, reasoning parser, and context ceiling.
case "$model" in
  Qwen3.6-35B-A3B)
    served_model_name=Qwen3.6-35B-A3B
    max_model_len_default=262144
    reasoning_args=(--reasoning-parser qwen3 --reasoning-config '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}')
    if [[ "$precision" == w8* ]]; then
      model_path=${RIVF26_FP8_MODEL_PATH:-$HOME/models/Qwen3.6-35B-A3B-FP8}
    else
      model_path=${RIVF26_BF16_MODEL_PATH:-/dev/shm/Qwen3.6-35B-A3B}
    fi
    ;;
  gpt-oss-120b)
    # Single native mxfp4 MoE checkpoint (auto-detected from the checkpoint's
    # own quantization_config, no --quantization flag needed).
    model_path=${RIVF26_GPTOSS120B_MODEL_PATH:-$HOME/models/gpt-oss-120b/gpt-oss-120b}
    served_model_name=gpt-oss-120b
    # gpt-oss-120b's own max_position_embeddings is 131072, well under the
    # Qwen arms' 262144 -- do not reuse their default here.
    max_model_len_default=131072
    reasoning_args=(--reasoning-parser openai_gptoss)
    ;;
esac
max_model_len=${RIVF26_MAX_MODEL_LEN:-$max_model_len_default}

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
  --served-model-name "$served_model_name"
  --host 127.0.0.1
  --port "$port"
  --tensor-parallel-size "${RIVF26_TENSOR_PARALLEL_SIZE:-2}"
  --max-model-len "$max_model_len"
  --max-num-seqs "$max_num_seqs"
  --gpu-memory-utilization "$gpu_memory_utilization"
  --kv-cache-dtype "$kv_dtype"
  --attention-backend "$attention_backend"
  "${reasoning_args[@]}"
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

if [[ "$enable_torch_profiler" == "1" || "$enable_torch_profiler" == "2" ]]; then
  if [[ -z "$run_id" ]]; then
    echo "RIVF26_RUN_ID is required when Torch Profiler is enabled" >&2
    exit 2
  fi
  profiler_dir=${RIVF26_TORCH_PROFILER_DIR:-${TORCH_PROFILER_DIR:-$RIVF26_BULK_ROOT/results/part2/$full_precision/$run_id/raw/torch_profiler}}
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

# HBM read/write bandwidth is captured by a fully separate amdsmi-based sampler
# process (scripts/monitoring/rocm_hbm_sampler.py), not by wrapping this server
# with a profiler. Classic rocprof does capture real per-kernel data, but
# wrapping the server with it for its whole lifetime was confirmed 2026-08-18
# to cause the recurring "Worker died unexpectedly" crash (decisive A/B test:
# identical config, only variable is rocprof wrapping on/off -- on crashes
# every time, off completed 1000/1000 requests with zero crashes). rocprofv3
# is unusable here for an unrelated reason (its hardware-counter injection
# SIGABRTs on any process that imports this environment's PyTorch). The
# sampler polls amdsmi externally and never touches this process, so it
# cannot reproduce that crash class -- and its UMC activity signal, earlier
# believed frozen, was confirmed working 2026-08-18 (that was a monitoring
# bug: watching the wrong amd-smi Device index for the GPU actually in use).
if [[ ${RIVF26_DISABLE_HBM_CAPTURE:-0} != 1 ]]; then
  mkdir -p "$bulk_run_dir/raw"
  setsid "$RIVF26_VENV_BIN/python" "$rivf26_root/scripts/monitoring/rocm_hbm_sampler.py" \
    --output-csv "$bulk_run_dir/raw/hbm.csv" \
    --metadata-json "$run_dir/hbm_metadata.json" \
    --visible-devices "$HIP_VISIBLE_DEVICES" \
    --frequency-hz "${RIVF26_HBM_FREQUENCY_HZ:-10}" \
    > "$bulk_run_dir/logs/hbm_sampler.log" 2>&1 &
  echo $! > "$bulk_run_dir/raw/hbm_sampler.pid"
  disown
fi

if [[ "$enable_torch_profiler" == "1" || "$enable_torch_profiler" == "2" ]] && [[ "${RIVF26_PROFILER_AUTO_SHUTDOWN:-0}" == "1" ]]; then
  # vLLM writes profiler_out_0.txt from rank 0 after TorchProfilerWrapper._stop
  # has flushed the trace. Treat that file as the completion sentinel, then
  # terminate only this server process (not the sampler, not the caller). This
  # is opt-in because a profiling capture intentionally ends before a normal
  # benchmark workload completes -- the caller's client will then see the
  # connection drop and should treat that as expected, not a crash.
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
