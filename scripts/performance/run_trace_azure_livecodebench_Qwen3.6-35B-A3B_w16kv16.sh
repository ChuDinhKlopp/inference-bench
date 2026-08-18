#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/run_trace_azure_livecodebench_common.sh" w16kv16
