#!/usr/bin/env python
"""Assert a running server actually honoured the pin, before any measurement.

    python scripts/kvquant/validate_runtime.py --server-log <log> \
        --expect-backend TRITON_ATTN --expect-kv-dtype int4_per_token_head

A dry run proves the CLI string; it does not prove the engine used that backend. vLLM
falls back when a backend rejects a configuration, and a silent fallback would turn this
study's controlled variable back into a confound -- the exact failure the pin exists to
prevent. So parse the server log and fail closed.
"""

import argparse
import json
import pathlib
import re
import sys

# vLLM logs e.g. "[cuda.py:422] Using AttentionBackendEnum.TRITON_ATTN backend."
# The enum prefix is optional so older/ROCm builds that log the bare name still match.
# Requiring " backend" immediately after the name keeps the MoE line
# ("Using TRITON Unquantized MoE backend") and the all-reduce line from matching.
BACKEND_RE = re.compile(r"Using (?:AttentionBackendEnum\.)?([A-Z][A-Z0-9_]*) backend\b")

# From the engine config line: "kv_cache_dtype=bfloat16," -- stop at the punctuation.
KV_RE = re.compile(r"kv[_-]cache[_-]dtype[:=]\s*'?([A-Za-z0-9_]+)", re.I)

CAP_RE = re.compile(r"GPU KV cache size:\s*([\d,]+)", re.I)

# From the engine config line: "reasoning_parser='qwen3'". An empty value here means
# bench.py's thinking_token_budget cannot be honoured: the server answers 200 with empty
# content and never decodes a token, which looks like a fast successful run.
REASONING_RE = re.compile(r"reasoning_parser='([^']*)'")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-log", required=True, type=pathlib.Path)
    ap.add_argument("--expect-backend", required=True)
    ap.add_argument("--expect-kv-dtype", required=True)
    ap.add_argument("--expect-workers", type=int, default=0,
                    help="if set, require this many worker backend lines (TP size)")
    ap.add_argument("--expect-reasoning-parser", default="",
                    help="if set, require this parser to be active (needed for thinking)")
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()

    if not a.server_log.exists():
        sys.exit(f"server log not found: {a.server_log}")
    text = a.server_log.read_text(errors="replace")

    backends = BACKEND_RE.findall(text)
    # The CLI echo on line 1 also contains --kv-cache-dtype; that is the request, not the
    # observation. Keep every hit but report the distinct set.
    kv = KV_RE.findall(text)
    cap = CAP_RE.findall(text)
    reasoning = REASONING_RE.findall(text)

    problems = []
    if not backends:
        problems.append("no attention backend line found in the server log")
    else:
        wrong = sorted(set(backends) - {a.expect_backend})
        if wrong:
            problems.append(
                f"backend mismatch: expected {a.expect_backend}, log also shows {wrong}")
        if a.expect_workers and len(backends) < a.expect_workers:
            problems.append(
                f"only {len(backends)} worker(s) reported a backend, expected {a.expect_workers}")

    if not kv:
        problems.append("no kv_cache_dtype found in the server log")
    elif a.expect_kv_dtype not in set(kv):
        problems.append(
            f"kv dtype mismatch: expected {a.expect_kv_dtype}, log shows {sorted(set(kv))}")

    if a.expect_reasoning_parser:
        active = {r for r in reasoning if r}
        if not active:
            problems.append(
                f"no reasoning parser active (expected {a.expect_reasoning_parser}); "
                "thinking_token_budget cannot be honoured and generation will be empty")
        elif a.expect_reasoning_parser not in active:
            problems.append(
                f"reasoning parser mismatch: expected {a.expect_reasoning_parser}, "
                f"log shows {sorted(active)}")

    result = {
        "status": "FAIL" if problems else "PASS",
        "expected_backend": a.expect_backend,
        "expected_kv_dtype": a.expect_kv_dtype,
        "observed_backends": sorted(set(backends)),
        "backend_lines": len(backends),
        "observed_kv_dtypes": sorted(set(kv)),
        "observed_reasoning_parsers": sorted({r for r in reasoning if r}),
        "kv_cache_tokens": int(cap[-1].replace(",", "")) if cap else None,
        "problems": problems,
    }
    print(json.dumps(result, indent=2))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=2) + "\n")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
