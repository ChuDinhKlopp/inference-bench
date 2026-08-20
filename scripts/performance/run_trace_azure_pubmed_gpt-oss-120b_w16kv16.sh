#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "$script_dir/run_trace_azure_pubmed_common.sh" gpt-oss-120b w16kv16 "$@"
