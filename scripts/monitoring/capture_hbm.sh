#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
rivf26_root=${RIVF26_ROOT:-$(cd -- "$script_dir/../.." && pwd)}
source "$rivf26_root/scripts/common/venv.sh"
source "$rivf26_root/scripts/common/paths.sh"

if (( $# < 2 )); then
  echo "usage: $0 OUTPUT_PREFIX COMMAND [ARG ...]" >&2
  exit 2
fi

output_prefix=$1
shift
frequency_hz=${RIVF26_HBM_FREQUENCY_HZ:-10}
# nsys' --gpu-metrics-set is chip-specific (ga100 for A100, gh100 for H100/Hopper --
# see `nsys profile --gpu-metrics-set=help`). Default stays ga100 for the A100 box;
# override per machine rather than hardcoding one architecture here.
gpu_metrics_set=${RIVF26_HBM_GPU_METRICS_SET:-ga100}
# Some hosts restrict GPU performance counters to admin users
# (RmProfilingAdminOnly/NVreg_RestrictProfilingToAdminUsers) with no per-user fix
# available -- e.g. inside a container, where the module parameter can't be safely
# reloaded without touching the host. -E preserves the caller's environment (venv,
# RIVF26_* vars) into the wrapped command, which also then runs as root; the caller is
# responsible for reclaiming ownership of anything this writes.
sudo_cmd=()
if [[ ${RIVF26_HBM_SUDO:-0} == 1 ]]; then
  sudo_cmd=(sudo -E)
fi
# --gpu-metrics-devices=all is a system-wide exclusive lock across every installed GPU,
# not just the ones this process uses: two concurrent nsys sessions both asking for
# 'all' collide with "Already under profiling" even when their actual workloads are on
# disjoint GPUs (e.g. parallel run_kvquant.sh model streams). Scope to the caller's
# CUDA_VISIBLE_DEVICES via nsys's own 'cuda-visible' keyword instead of passing the
# CUDA_VISIBLE_DEVICES values through as literal --gpu-metrics-devices IDs: nsys numbers
# devices by its own enumeration (verified NOT the same as CUDA/nvidia-smi index -- e.g.
# nsys id 0 was nvidia-smi index 2 on this host), so raw numeric passthrough would silently
# capture the wrong physical GPUs. Falls back to 'all' when CUDA_VISIBLE_DEVICES is unset,
# matching prior single-stream-at-a-time behavior.
gpu_metrics_devices=${RIVF26_HBM_GPU_METRICS_DEVICES:-$([[ -n ${CUDA_VISIBLE_DEVICES:-} ]] && echo cuda-visible || echo all)}

if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys is required for HBM telemetry (--gpu-metrics-set=$gpu_metrics_set)" >&2
  exit 2
fi
if [[ -e "${output_prefix}.nsys-rep" || -e "${output_prefix}.qdstrm" ]]; then
  echo "refusing to overwrite existing HBM capture: ${output_prefix}.*" >&2
  exit 2
fi

mkdir -p "$(dirname "$output_prefix")"

# GPU Metrics is device-level and samples independently of CUDA tracing. The
# wrapped process defines the exact workload interval; no CPU sampling, kernel
# trace, or backtrace collection is enabled.
exec "${sudo_cmd[@]}" nsys profile \
  --trace=none \
  --sample=none \
  --gpu-metrics-devices="$gpu_metrics_devices" \
  --gpu-metrics-set="$gpu_metrics_set" \
  --gpu-metrics-frequency="$frequency_hz" \
  --force-overwrite=false \
  --output "$output_prefix" \
  "$@"
