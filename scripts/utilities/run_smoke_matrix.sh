#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
rivf26_root=${RIVF26_ROOT:-$(cd -- "$script_dir/../.." && pwd)}
source "$rivf26_root/scripts/common/venv.sh"
source "$rivf26_root/scripts/common/paths.sh"

matrix_id=${RIVF26_SMOKE_MATRIX_ID:-$(date -u +%Y%m%d_%H%M%S)_smoke_matrix_mbt16384}
gate=${RIVF26_SMOKE_MATRIX_OUTPUT:-$rivf26_root/manifests/$matrix_id.json}
precisions=(w16kv16 w8kv16 w8kv8 w16kv8)
summaries=()

if [[ -e "$gate" ]]; then
  echo "refusing to overwrite smoke matrix gate: $gate" >&2
  exit 2
fi

for precision in "${precisions[@]}"; do
  run_id=$(date -u +%Y%m%d_%H%M%S)_smoke_mbt16384_${precision}
  RIVF26_RUN_ID="$run_id" "$script_dir/run_smoke.sh" "$precision"
  summaries+=(--summary "$rivf26_root/results/part1/smoke/$run_id/summary.json")
done

"$RIVF26_VENV_BIN/python" "$script_dir/build_smoke_matrix.py" \
  --root "$rivf26_root" --output "$gate" "${summaries[@]}"

printf 'four-precision smoke matrix PASS: %s\n' "$gate"
printf 'export RIVF26_SMOKE_MATRIX=%q\n' "$gate"
