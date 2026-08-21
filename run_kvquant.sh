#!/usr/bin/env bash
# Part 3 KV-quantization sweep -- LiveCodeBench v6 Azure, three attention architectures
# x three KV dtypes x a max_num_seqs ladder, all on a pinned TRITON_ATTN.
#
#   ./run_kvquant.sh                        # 9 primary arms x MNS ladder
#   ./run_kvquant.sh --dry-run              # print every cell's config, launch nothing
#   ./run_kvquant.sh qwen3_30b              # substring filter: that model, all dtypes
#   ./run_kvquant.sh kv4                    # all models at int4
#   ./run_kvquant.sh qwen3_30b_kv4          # one arm
#   ./run_kvquant.sh --mns 128              # override the ladder
#   ./run_kvquant.sh --include-secondary    # add the 2 TurboQuant arms (Qwen only)
#
# Unlike run_lcb.sh (Part 1) this never varies WEIGHT precision: each checkpoint runs in
# its native format and only --kv-cache-dtype changes. That is what makes it a KV study.
#
# Parallel across MODELS, sequential within each model. In one invocation with multiple
# models selected -- the default, or e.g. `./run_kvquant.sh qwen3_30b gptoss_120b` --
# qwen36_35b, gptoss_120b and qwen3_30b each get their own vLLM server: a disjoint
# tensor_parallel_size-wide slice of CUDA_VISIBLE_DEVICES and a distinct port
# (RIVF26_PORT_BASE, default 8000, +0/+1/+2 in that model order), so they run genuinely
# concurrently instead of sharing GPUs.
#
# Running three SEPARATE invocations (e.g. one per terminal) instead does NOT get you
# this automatically -- each process only sees the one model it was given, so all three
# would independently compute index 0 and collide on port 8000 / GPUs 0,1. Pin each
# terminal explicitly instead:
#
#   ./run_kvquant.sh qwen36_35b  --port 8000 --cuda 0,1
#   ./run_kvquant.sh gptoss_120b --port 8001 --cuda 2,3
#   ./run_kvquant.sh qwen3_30b   --port 8002 --cuda 4,5
#
# --port/--cuda require exactly one model selected in that invocation (there is nowhere
# for a second model's server to bind on a port/GPU set you already handed to the first).
#
# Within one model's stream, kv-dtype arms and the max_num_seqs ladder still run one cell
# at a time and still cooldown between them -- the HBM-drain reasoning only applies to the
# arm that just freed that memory, and each stream owns its own GPUs regardless of what
# the other two streams are doing. Fail-fast is per-stream: one model's failure stops that
# model's remaining cells (KVQ_CONTINUE_ON_FAIL=1 to keep going past failures) but does not
# stop the other two models already running in parallel on their own GPUs.
# Each stream's orchestration log lands at $RIVF26_BULK_ROOT/sweep_logs/<model>.log; this
# script's own stdout only gets the per-model launch/completion lines.
set -uo pipefail

cd -- "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# ---- H100 machine overrides --------------------------------------------------
# scripts/common/{venv,paths}.sh ship defaults baked in for other machines (a100's
# ducct/UID 1009 venv and tmpfs bulk root). Overriding here, rather than editing those
# shared files, keeps them identical across branches so main/a100/h100 cherry-picks
# don't conflict.
export RIVF26_VENV_BIN="${RIVF26_VENV_BIN:-$HOME/projects/h100_vllm/vllm/.venv/bin}"
export RIVF26_BULK_ROOT="${RIVF26_BULK_ROOT:-$HOME/projects/h100_vllm/rivf26-bulk}"
# scripts/monitoring/capture_hbm.sh defaults nsys's --gpu-metrics-set to ga100 (A100);
# this box is Hopper/GH100, so it needs the matching set name (`nsys profile
# --gpu-metrics-set=help` lists the valid names per chip).
export RIVF26_HBM_GPU_METRICS_SET="${RIVF26_HBM_GPU_METRICS_SET:-gh100}"
# This host runs the driver with RmProfilingAdminOnly=1 (GPU perf counters restricted to
# root) and is a privileged container on a shared host, so reloading the nvidia kernel
# module to lift that per-user is out of scope here. capture_hbm.sh runs nsys (and the
# client it wraps) under `sudo -E` instead; run_azure_livecodebench_common.sh chowns the
# run's output back to this user afterward.
export RIVF26_HBM_SUDO="${RIVF26_HBM_SUDO:-1}"

# ---- shared configuration for every cell ------------------------------------
export RIVF26_LCB_PERFORMANCE_REQUESTS=1055
export RIVF26_AZURE_LCB_TRACE_CSV="$PWD/traces/processed/azure_multimodal_bursty_1055.csv"
export RIVF26_MAX_NUM_BATCHED_TOKENS=8192
export RIVF26_KVQ_LCB_THINKING_TOKEN_BUDGET=24576
export RIVF26_KVQ_LCB_MAX_GEN_TOKS=32768
export RIVF26_PLOT_BIN_SECONDS=0.5
# max_model_len comes from configs/part3_kvquant.json (49152) unless overridden here;
# it must stay identical across arms or the capacity comparison is not like-for-like.
: "${RIVF26_MAX_MODEL_LEN:=}"
[[ -n $RIVF26_MAX_MODEL_LEN ]] && export RIVF26_MAX_MODEL_LEN

# Canonical model order -- also the CUDA_VISIBLE_DEVICES / port assignment order below.
MODELS=(qwen36_35b gptoss_120b qwen3_30b)
PRIMARY_ARMS=(
  qwen36_35b_kv16 qwen36_35b_kv8 qwen36_35b_kv4
  gptoss_120b_kv16 gptoss_120b_kv8 gptoss_120b_kv4
  qwen3_30b_kv16  qwen3_30b_kv8  qwen3_30b_kv4
)
SECONDARY_ARMS=(qwen36_35b_kvtq4 qwen3_30b_kvtq4)   # TurboQuant: no sink support -> Qwen only
COOLDOWN_S=${KVQ_COOLDOWN_S:-30}
PORT_BASE=${RIVF26_PORT_BASE:-8000}

dry_run=0; secondary=0; filters=(); mns_override=(); port_override=""; cuda_override=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)           dry_run=1; shift ;;
    --include-secondary) secondary=1; shift ;;
    --mns)               mns_override+=("${2:?--mns needs a value}"); shift 2 ;;
    --port)               port_override=${2:?--port needs a value}; shift 2 ;;
    --cuda)               cuda_override=${2:?--cuda needs a value, e.g. 0,1}; shift 2 ;;
    -h|--help)            sed -n '2,42p' "$0"; exit 0 ;;
    -*)                   echo "unknown option: $1" >&2; exit 2 ;;
    *)                    filters+=("$1"); shift ;;
  esac
done

arms=("${PRIMARY_ARMS[@]}")
(( secondary )) && arms+=("${SECONDARY_ARMS[@]}")

if (( ${#filters[@]} )); then
  selected=()
  for a in "${arms[@]}"; do
    for f in "${filters[@]}"; do
      [[ $a == *"$f"* ]] && { selected+=("$a"); break; }
    done
  done
  (( ${#selected[@]} )) || { echo "no arm matches: ${filters[*]}" >&2
                             echo "available: ${arms[*]}" >&2; exit 2; }
  arms=("${selected[@]}")
fi

if (( ${#mns_override[@]} )); then
  mns_ladder=("${mns_override[@]}")
else
  mapfile -t mns_ladder < <(python3 -c '
import json;print("\n".join(str(x) for x in json.load(open("configs/part3_kvquant.json"))["sweep"]["max_num_seqs"]))')
fi

tp=$(python3 -c '
import json;print(json.load(open("configs/part3_kvquant.json"))["common"]["tensor_parallel_size"])')

[[ -f $RIVF26_AZURE_LCB_TRACE_CSV ]] || {
  echo "missing Azure trace: $RIVF26_AZURE_LCB_TRACE_CSV" >&2; exit 2; }

# ---- group the selected arms by model, in MODELS order -----------------------
declare -A model_arms
models_to_run=()
for m in "${MODELS[@]}"; do
  bucket=()
  for a in "${arms[@]}"; do
    [[ $a == "${m}_"* ]] && bucket+=("$a")
  done
  if (( ${#bucket[@]} )); then
    model_arms[$m]="${bucket[*]}"
    models_to_run+=("$m")
  fi
done

if [[ -n $port_override || -n $cuda_override ]] && (( ${#models_to_run[@]} != 1 )); then
  echo "--port/--cuda require exactly one model selected, got: ${models_to_run[*]:-none}" >&2
  exit 2
fi
if [[ -n $cuda_override ]]; then
  n_cuda=$(( $(grep -o , <<<"$cuda_override" | wc -l) + 1 ))
  (( n_cuda == tp )) || {
    echo "--cuda $cuda_override has $n_cuda device(s), tensor_parallel_size is $tp" >&2; exit 2; }
fi

# ---- assign each model a disjoint GPU slice and a distinct port --------------
# --port/--cuda (single-model invocations only, e.g. one per terminal) override the
# automatic index-based assignment below, which is only collision-free *within* one
# invocation that sees multiple models at once.
declare -A model_port model_devices
i=0
for m in "${models_to_run[@]}"; do
  model_port[$m]=${port_override:-$(( PORT_BASE + i ))}
  if [[ -n $cuda_override ]]; then
    model_devices[$m]=$cuda_override
  else
    lo=$(( i * tp )); hi=$(( lo + tp - 1 ))
    model_devices[$m]=$(seq -s, "$lo" "$hi")
  fi
  i=$((i+1))
done

if (( ! dry_run )) && command -v nvidia-smi >/dev/null 2>&1; then
  n_gpus=$(nvidia-smi -L | wc -l)
  needed=$(( ${#models_to_run[@]} * tp ))
  (( needed <= n_gpus )) || {
    echo "need $needed GPUs for ${#models_to_run[@]} parallel model(s) at tp=$tp, only $n_gpus visible" >&2
    exit 2; }
fi

sweep_log_dir="$RIVF26_BULK_ROOT/sweep_logs"
mkdir -p "$sweep_log_dir"

total=$(( ${#arms[@]} * ${#mns_ladder[@]} ))
echo "=============================================================="
echo "Part 3 KV-quantization sweep (LiveCodeBench v6 Azure)"
echo "  arms            : ${#arms[@]}  (${arms[*]})"
echo "  max_num_seqs    : ${mns_ladder[*]}"
echo "  cells           : $total"
echo "  requests        : $RIVF26_LCB_PERFORMANCE_REQUESTS"
echo "  max_batched_tok : $RIVF26_MAX_NUM_BATCHED_TOKENS"
echo "  thinking budget : $RIVF26_KVQ_LCB_THINKING_TOKEN_BUDGET"
echo "  max_gen_toks    : $RIVF26_KVQ_LCB_MAX_GEN_TOKS"
echo "  trace           : $RIVF26_AZURE_LCB_TRACE_CSV"
echo "  tensor_parallel : $tp"
for m in "${models_to_run[@]}"; do
  echo "  $m  ->  port=${model_port[$m]}  cuda=${model_devices[$m]}  arms=${model_arms[$m]}"
done
echo "=============================================================="

# One model's full sweep (its arms x the mns ladder), pinned to one port/GPU slice.
# Runs as a background job of the caller; all of its output is redirected to its own
# log file so three concurrent streams don't interleave mid-line on the terminal.
run_model_stream() {
  local model=$1 port=$2 devices=$3; shift 3
  local -a my_arms=("$@")
  (
    export RIVF26_PORT="$port" CUDA_VISIBLE_DEVICES="$devices"
    local failed=0 cell=0
    echo "[$model] port=$port cuda=$devices arms=${my_arms[*]}"
    for arm in "${my_arms[@]}"; do
      runner="./scripts/kvquant/run_azure_livecodebench_${arm}.sh"
      if [[ ! -x $runner ]]; then
        echo "[$model] missing runner: $runner" >&2
        failed=$((failed+1)); continue
      fi

      for mns in "${mns_ladder[@]}"; do
        cell=$((cell+1))
        # Timestamp at launch, not up front: later cells start hours after this stream does.
        run_id="$(date -u +%Y%m%d_%H%M%S)_part3_livecodebench_v6_${arm}_mns${mns}"

        echo
        echo "--------------------------------------------------------------"
        echo "[$model] [$(date '+%F %T')] cell $cell arm=$arm mns=$mns run_id=$run_id"

        if (( dry_run )); then
          RIVF26_DRY_RUN=1 RIVF26_MAX_NUM_SEQS="$mns" RIVF26_RUN_ID="$run_id" "$runner" | sed 's/^/  /'
          continue
        fi

        start=$SECONDS
        if RIVF26_MAX_NUM_SEQS="$mns" RIVF26_RUN_ID="$run_id" "$runner"; then
          el=$((SECONDS - start))
          printf '[%s] [%s] arm=%s mns=%s OK in %02d:%02d:%02d\n' "$(date '+%F %T')" "$model" "$arm" "$mns" \
            $((el/3600)) $(((el%3600)/60)) $((el%60))
        else
          rc=$?
          echo "[$model] [$(date '+%F %T')] arm=$arm mns=$mns FAILED (exit $rc)" >&2
          failed=$((failed+1))
          if [[ ${KVQ_CONTINUE_ON_FAIL:-0} != 1 ]]; then
            echo "[$model] fail-fast: stopping this stream. Set KVQ_CONTINUE_ON_FAIL=1 to continue past failures." >&2
            break 2
          fi
        fi

        # Let this stream's own HBM actually free before its next cell sizes its KV
        # cache -- unrelated to the other two streams, which own different GPUs.
        (( dry_run )) || { echo "[$model] cooldown ${COOLDOWN_S}s"; sleep "$COOLDOWN_S"; }
      done
    done
    exit "$failed"
  ) > "$sweep_log_dir/${model}.log" 2>&1
}

declare -A stream_pid
for m in "${models_to_run[@]}"; do
  echo "[$(date '+%F %T')] launching $m  port=${model_port[$m]}  cuda=${model_devices[$m]}  log=$sweep_log_dir/${m}.log"
  run_model_stream "$m" "${model_port[$m]}" "${model_devices[$m]}" ${model_arms[$m]} &
  stream_pid[$m]=$!
done

overall_failed=0
for m in "${models_to_run[@]}"; do
  if wait "${stream_pid[$m]}"; then
    echo "[$(date '+%F %T')] $m stream completed OK"
  else
    rc=$?
    echo "[$(date '+%F %T')] $m stream FAILED (exit $rc) -- see $sweep_log_dir/${m}.log" >&2
    overall_failed=$((overall_failed+1))
  fi
  (( dry_run )) && cat "$sweep_log_dir/${m}.log"
done

echo
echo "=============================================================="
echo "summary"
for m in "${models_to_run[@]}"; do
  echo "  --- $m ---"
  grep -E 'OK in|FAILED \(exit|missing runner|model_path=' "$sweep_log_dir/${m}.log" | sed 's/^/    /'
done
echo "=============================================================="
if (( overall_failed )); then
  echo "$overall_failed model stream(s) failed"
  exit 1
fi
echo "all cells completed"
