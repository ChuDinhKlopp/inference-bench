#!/usr/bin/env bash

# RIVF26 uses the vLLM checkout's virtual environment for every script. Keep
# the directory configurable for machine portability, but never silently fall
# back to an unrelated Python environment.
RIVF26_VENV_BIN=${RIVF26_VENV_BIN:-$HOME/repos/vllm/.venv/bin}
if [[ ! -x "$RIVF26_VENV_BIN/python" ]]; then
  echo "RIVF26 vLLM virtual environment is missing: $RIVF26_VENV_BIN/python" >&2
  return 2 2>/dev/null || exit 2
fi
export RIVF26_VENV_BIN
export PATH="$RIVF26_VENV_BIN:$PATH"
