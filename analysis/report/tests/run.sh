#!/usr/bin/env bash
# Check a built report. Extracts its <script> block and runs the render,
# control-path, Pareto-claim, and tooltip-host checks against it.
#
#   analysis/report/tests/run.sh [path/to/report.html]
#
# Requires node. Exits non-zero if any check fails.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
report="${1:-$root/results/part1/performance/precision_performance_report.html}"

if [[ ! -f "$report" ]]; then
  echo "no such report: $report" >&2
  echo "build one first: python analysis/report/build_report.py" >&2
  exit 2
fi
if ! command -v node >/dev/null 2>&1; then
  echo "node not found; these checks need it" >&2
  exit 2
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cp "$here"/*.js "$work"/

# the harnesses require ./script.js -- the report's inlined chart code
python3 - "$report" "$work/script.js" <<'PY'
import re, sys
html = open(sys.argv[1]).read()
m = re.search(r'<script>\n(.*)\n</script>', html, re.S)
if not m:
    sys.exit("could not find the <script> block in the report")
open(sys.argv[2], 'w').write(m.group(1))
PY
[[ -f "$work/script.js" ]] || exit 2

export REPORT_HTML="$report"
fail=0
for check in harness.js harness2.js pareto_check.js tooltip_check.js; do
  echo "=== $check"
  ( cd "$work" && node "$check" ) || fail=1
  echo
done

if [[ $fail -ne 0 ]]; then
  echo "CHECKS FAILED"
  exit 1
fi
echo "ALL CHECKS PASSED"
