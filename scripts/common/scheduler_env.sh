#!/usr/bin/env bash

# Keep the scheduler token budget explicit and consistent within each run.
rivf26_set_scheduler_env() {
  local value=${RIVF26_MAX_NUM_BATCHED_TOKENS:-8192}
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "RIVF26_MAX_NUM_BATCHED_TOKENS must be a positive integer; got $value" >&2
    return 2
  fi
  export RIVF26_MAX_NUM_BATCHED_TOKENS=$value
}
