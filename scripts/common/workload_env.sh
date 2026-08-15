#!/usr/bin/env bash

# This file is sourced by Part 1 workload launchers. MAX_GEN_TOKS is a client
# request limit; RIVF26_MAX_MODEL_LEN is the independent server context limit.
rivf26_set_workload_env() {
  local mode=${1:?workload mode is required}
  local expected_max_gen_toks expected_arrival_mode expected_reasoning_effort

  case "$mode" in
    accuracy)
      expected_max_gen_toks=32768
      expected_arrival_mode=none
      expected_reasoning_effort=high
      ;;
    performance)
      expected_max_gen_toks=10240
      expected_arrival_mode=azure
      expected_reasoning_effort=low
      ;;
    *)
      echo "unsupported RIVF26 workload mode: $mode" >&2
      return 2
      ;;
  esac

  # Fail closed if a caller inherited a cap from another workload. This avoids
  # silently running a matrix cell with the wrong scientific configuration.
  if [[ -n ${MAX_GEN_TOKS+x} && "$MAX_GEN_TOKS" != "$expected_max_gen_toks" ]]; then
    echo "$mode requires MAX_GEN_TOKS=$expected_max_gen_toks; got $MAX_GEN_TOKS" >&2
    return 2
  fi
  if [[ -n ${RIVF26_MAX_GEN_TOKS+x} && "$RIVF26_MAX_GEN_TOKS" != "$expected_max_gen_toks" ]]; then
    echo "$mode requires RIVF26_MAX_GEN_TOKS=$expected_max_gen_toks; got $RIVF26_MAX_GEN_TOKS" >&2
    return 2
  fi
  if [[ "$mode" == accuracy && -n ${GPQA_MAX_GEN_TOKS+x} && "$GPQA_MAX_GEN_TOKS" != "$expected_max_gen_toks" ]]; then
    echo "accuracy requires GPQA_MAX_GEN_TOKS=$expected_max_gen_toks; got $GPQA_MAX_GEN_TOKS" >&2
    return 2
  fi
  if [[ "$mode" == performance && -n ${PUBMED_MAX_GEN_TOKS+x} && "$PUBMED_MAX_GEN_TOKS" != "$expected_max_gen_toks" ]]; then
    echo "performance requires PUBMED_MAX_GEN_TOKS=$expected_max_gen_toks; got $PUBMED_MAX_GEN_TOKS" >&2
    return 2
  fi
  if [[ -n ${BENCH_ARRIVAL_RATE+x} && "$BENCH_ARRIVAL_RATE" != "$expected_arrival_mode" ]]; then
    echo "$mode requires BENCH_ARRIVAL_RATE=$expected_arrival_mode; got $BENCH_ARRIVAL_RATE" >&2
    return 2
  fi

  export RIVF26_MODE=$mode
  export RIVF26_MAX_GEN_TOKS=$expected_max_gen_toks
  export MAX_GEN_TOKS=$expected_max_gen_toks
  export BENCH_ARRIVAL_RATE=$expected_arrival_mode
  export RIVF26_REASONING_EFFORT=$expected_reasoning_effort

  if [[ "$mode" == accuracy ]]; then
    export GPQA_MAX_GEN_TOKS=$expected_max_gen_toks
  else
    export PUBMED_MAX_GEN_TOKS=$expected_max_gen_toks
  fi

  if [[ -n ${RIVF26_MAX_MODEL_LEN:-} ]] && (( RIVF26_MAX_MODEL_LEN <= expected_max_gen_toks )); then
    echo "RIVF26_MAX_MODEL_LEN=$RIVF26_MAX_MODEL_LEN must exceed MAX_GEN_TOKS=$expected_max_gen_toks to leave room for the prompt" >&2
    return 2
  fi
}
