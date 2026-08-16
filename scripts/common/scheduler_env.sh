#!/usr/bin/env bash

# Part 1 keeps the scheduler token budget identical across every precision,
# workload, and max-num-seqs matrix cell.
rivf26_set_scheduler_env() {
  local expected_max_num_batched_tokens=16384
  if [[ -n ${RIVF26_MAX_NUM_BATCHED_TOKENS+x} \
        && "$RIVF26_MAX_NUM_BATCHED_TOKENS" != "$expected_max_num_batched_tokens" ]]; then
    echo "RIVF26 requires RIVF26_MAX_NUM_BATCHED_TOKENS=$expected_max_num_batched_tokens; got $RIVF26_MAX_NUM_BATCHED_TOKENS" >&2
    return 2
  fi
  export RIVF26_MAX_NUM_BATCHED_TOKENS=$expected_max_num_batched_tokens
}
