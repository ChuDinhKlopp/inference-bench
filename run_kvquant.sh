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
# Sequential by design -- the arms share four GPUs. Fail-fast unless
# KVQ_CONTINUE_ON_FAIL=1.
set -uo pipefail

cd -- "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

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

PRIMARY_ARMS=(
  qwen36_35b_kv16 qwen36_35b_kv8 qwen36_35b_kv4
  gptoss_120b_kv16 gptoss_120b_kv8 gptoss_120b_kv4
  qwen3_30b_kv16  qwen3_30b_kv8  qwen3_30b_kv4
)
SECONDARY_ARMS=(qwen36_35b_kvtq4 qwen3_30b_kvtq4)   # TurboQuant: no sink support -> Qwen only
COOLDOWN_S=${KVQ_COOLDOWN_S:-30}

dry_run=0; secondary=0; filters=(); mns_override=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)           dry_run=1; shift ;;
    --include-secondary) secondary=1; shift ;;
    --mns)               mns_override+=("${2:?--mns needs a value}"); shift 2 ;;
    -h|--help)           sed -n '2,18p' "$0"; exit 0 ;;
    -*)                  echo "unknown option: $1" >&2; exit 2 ;;
    *)                   filters+=("$1"); shift ;;
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

[[ -f $RIVF26_AZURE_LCB_TRACE_CSV ]] || {
  echo "missing Azure trace: $RIVF26_AZURE_LCB_TRACE_CSV" >&2; exit 2; }

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
echo "=============================================================="

declare -a summary=()
failed=0; done_cells=0

for arm in "${arms[@]}"; do
  runner="./scripts/kvquant/run_azure_livecodebench_${arm}.sh"
  if [[ ! -x $runner ]]; then
    echo "missing runner: $runner" >&2
    summary+=("$arm  --  MISSING-RUNNER"); failed=$((failed+1)); continue
  fi

  for mns in "${mns_ladder[@]}"; do
    done_cells=$((done_cells+1))
    # Timestamp at launch, not up front: later cells start hours after this script does.
    run_id="$(date -u +%Y%m%d_%H%M%S)_part3_livecodebench_v6_${arm}_mns${mns}"

    echo
    echo "--------------------------------------------------------------"
    echo "[$(date '+%F %T')] cell $done_cells/$total  arm=$arm mns=$mns"
    echo "  run_id: $run_id"

    if (( dry_run )); then
      RIVF26_DRY_RUN=1 RIVF26_MAX_NUM_SEQS="$mns" RIVF26_RUN_ID="$run_id" "$runner" | sed 's/^/  /'
      summary+=("$arm  mns=$mns  DRY-RUN")
      continue
    fi

    start=$SECONDS
    if RIVF26_MAX_NUM_SEQS="$mns" RIVF26_RUN_ID="$run_id" "$runner"; then
      el=$((SECONDS - start))
      printf '[%s] arm=%s mns=%s OK in %02d:%02d:%02d\n' "$(date '+%F %T')" "$arm" "$mns" \
        $((el/3600)) $(((el%3600)/60)) $((el%60))
      summary+=("$arm  mns=$mns  OK  $(printf '%02d:%02d:%02d' $((el/3600)) $(((el%3600)/60)) $((el%60)))  $run_id")
    else
      rc=$?
      echo "[$(date '+%F %T')] arm=$arm mns=$mns FAILED (exit $rc)" >&2
      summary+=("$arm  mns=$mns  FAILED(exit $rc)  $run_id")
      failed=$((failed+1))
      if [[ ${KVQ_CONTINUE_ON_FAIL:-0} != 1 ]]; then
        echo "fail-fast: stopping. Set KVQ_CONTINUE_ON_FAIL=1 to continue past failures." >&2
        break 2
      fi
    fi

    # Let HBM actually free before the next cell sizes its KV cache -- the capacity
    # numbers are the measurement, so a stale allocation would corrupt them.
    (( done_cells < total )) && { echo "  cooldown ${COOLDOWN_S}s"; sleep "$COOLDOWN_S"; }
  done
done

echo
echo "=============================================================="
echo "summary"
for line in "${summary[@]}"; do echo "  $line"; done
echo "=============================================================="
if (( failed )); then
  echo "$failed cell(s) failed"
  exit 1
fi
echo "all cells completed"
