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

if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys is required for GA100 HBM telemetry" >&2
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
exec nsys profile \
  --trace=none \
  --sample=none \
  --gpu-metrics-devices=all \
  --gpu-metrics-set=ga100 \
  --gpu-metrics-frequency="$frequency_hz" \
  --force-overwrite=false \
  --output "$output_prefix" \
  "$@"
