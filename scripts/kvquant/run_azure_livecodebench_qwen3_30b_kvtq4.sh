#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/run_azure_livecodebench_common.sh" qwen3_30b kvtq4
