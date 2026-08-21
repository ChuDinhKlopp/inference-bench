#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/run_azure_livecodebench_common.sh" gptoss_120b kv8
