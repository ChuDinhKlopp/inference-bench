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
source "$rivf26_root/scripts/common/workload_env.sh"

export RIVF26_MAX_MODEL_LEN=${RIVF26_MAX_MODEL_LEN:-65536}
rivf26_set_scheduler_env
rivf26_set_workload_env performance

run_id=${RIVF26_RUN_ID:-}
port=${RIVF26_PORT:-8000}
workload=${RIVF26_PUBMED_WORKLOAD:-$RIVF26_BULK_ROOT/datasets/processed/pubmed_azure_bursty_1000.jsonl}
if [[ -z "$run_id" && ${RIVF26_DRY_RUN:-0} != 1 ]]; then
  echo "RIVF26_RUN_ID is required" >&2
  exit 2
fi

run_dir=${RIVF26_RUN_DIR:-$rivf26_root/results/part1/performance/$run_id}
bulk_run_dir=${RIVF26_BULK_RUN_DIR:-$RIVF26_BULK_ROOT/results/part1/performance/$run_id}
server_log=${RIVF26_SERVER_LOG:-$run_dir/logs/server.log}
post_server=${RIVF26_POST_SERVER_JSON:-$rivf26_root/manifests/$run_id/post_server.json}

cmd=(
  "$RIVF26_VENV_BIN/python" "$script_dir/run_pubmed_trace.py"
  --workload "$workload"
  --output-dir "$run_dir"
  --base-url "http://127.0.0.1:$port"
  --model Qwen3.6-35B-A3B
  --server-log-file "$server_log"
  --server-metrics-poll-interval "${RIVF26_SERVER_METRICS_POLL_INTERVAL:-0.2}"
  --max-num-batched-tokens "$RIVF26_MAX_NUM_BATCHED_TOKENS"
  --timeout "${RIVF26_REQUEST_TIMEOUT:-43200}"
)

printf 'RIVF26 PubMed client command:'
printf ' %q' "${cmd[@]}"
printf '\nMAX_GEN_TOKS=%s MAX_NUM_BATCHED_TOKENS=%s BENCH_ARRIVAL_RATE=%s reasoning_effort=%s precision=%s\n' \
  "$MAX_GEN_TOKS" "$RIVF26_MAX_NUM_BATCHED_TOKENS" "$BENCH_ARRIVAL_RATE" "$RIVF26_REASONING_EFFORT" "$precision"

if [[ ${RIVF26_DRY_RUN:-0} == 1 ]]; then
  exec "${cmd[@]}" --validate-only
fi
if [[ ! -f "$workload" ]]; then
  echo "missing frozen PubMed/Azure workload: $workload" >&2
  exit 2
fi
if [[ ! -f "$post_server" ]]; then
  echo "missing Stage B validation: $post_server" >&2
  exit 2
fi
if ! "$RIVF26_VENV_BIN/python" -c \
  'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d["status"] == "PASS" and d.get("long_run_eligible", False) else 2)' \
  "$post_server"; then
  echo "Stage B did not PASS all scientific long-run gates: $post_server" >&2
  exit 2
fi

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

exec "${cmd[@]}"
