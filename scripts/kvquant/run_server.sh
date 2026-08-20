#!/usr/bin/env bash
# Launch a vLLM server for one KV-quantization study cell.
#
#   scripts/kvquant/run_server.sh <model_key> <kv_dtype_key>
#   e.g. scripts/kvquant/run_server.sh qwen3_30b kv4
#
# Reads configs/part3_kvquant.json so model paths, backend and limits come from the
# study spec rather than being duplicated here. Unlike the Part 1 launcher this one
# never varies WEIGHT precision -- each checkpoint runs in its native format and only
# --kv-cache-dtype changes, which is the whole point of the study.
#
# Env overrides: RIVF26_MAX_NUM_SEQS, RIVF26_MAX_MODEL_LEN, RIVF26_PORT,
#                RIVF26_RUN_ID, RIVF26_DRY_RUN
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
rivf26_root=${RIVF26_ROOT:-$(cd -- "$script_dir/../.." && pwd)}
spec="$rivf26_root/configs/part3_kvquant.json"
py=${RIVF26_VENV_BIN:-$HOME/repos/vllm/.venv/bin}/python

model_key=${1:?usage: run_server.sh <model_key> <kv_dtype_key>}
kv_key=${2:?usage: run_server.sh <model_key> <kv_dtype_key>}

# Resolve via the spec. Capture explicitly rather than `read < <(...)`: process
# substitution does not propagate the child's exit status, and these lookups enforce
# structural exclusions (e.g. TurboQuant cannot run GPT-OSS) that must fail closed.
if ! resolved=$("$py" - "$spec" "$model_key" "$kv_key" <<'PY'
import json, sys
spec = json.load(open(sys.argv[1]))
mk, kk = sys.argv[2], sys.argv[3]
if mk not in spec["models"]:
    sys.exit(f"unknown model {mk!r}; have {list(spec['models'])}")
if kk not in spec["kv_dtypes"]:
    sys.exit(f"unknown kv dtype {kk!r}; have {list(spec['kv_dtypes'])}")
m, d, c = spec["models"][mk], spec["kv_dtypes"][kk], spec["common"]
allowed = d.get("models")
if allowed is not None and mk not in allowed:
    sys.exit(f"{kk} is not supported for {mk} (see spec notes); refusing to launch")
print(m["path_default"], m["quantization"] or "-", d["backend"], d["cli"],
      c["tensor_parallel_size"], c["max_num_batched_tokens"],
      c["gpu_memory_utilization"], c["max_model_len"])
PY
); then
  exit 1
fi
read -r model_path quantization backend kv_cli tp mbt gmu mml <<<"$resolved"

max_num_seqs=${RIVF26_MAX_NUM_SEQS:-128}
max_model_len=${RIVF26_MAX_MODEL_LEN:-$mml}
port=${RIVF26_PORT:-8000}

if [[ ! -d $model_path ]]; then
  echo "model directory not found: $model_path" >&2
  echo "  the harness never downloads weights; place them locally or edit the spec" >&2
  exit 2
fi

cmd=(
  "${RIVF26_VENV_BIN:-$HOME/repos/vllm/.venv/bin}/python" -m vllm.entrypoints.cli.main serve
  "$model_path"
  --tokenizer "$model_path"
  --served-model-name "$model_key"
  --host 127.0.0.1 --port "$port"
  --tensor-parallel-size "$tp"
  --max-model-len "$max_model_len"
  --max-num-seqs "$max_num_seqs"
  --max-num-batched-tokens "$mbt"
  --gpu-memory-utilization "$gmu"
  --kv-cache-dtype "$kv_cli"
  --attention-backend "$backend"
  --enable-logging-iteration-details
)
[[ $quantization != "-" ]] && cmd+=(--quantization "$quantization")

printf '%q ' "${cmd[@]}"; echo
if [[ ${RIVF26_DRY_RUN:-0} == 1 ]]; then
  echo "# dry run: model=$model_key kv=$kv_key backend=$backend dtype=$kv_cli mns=$max_num_seqs"
  exit 0
fi

exec "${cmd[@]}"
