#!/usr/bin/env python
"""Assert a running server actually honoured the pin, before any measurement.

    python scripts/kvquant/validate_runtime.py --server-log <log> --expect-backend TRITON_ATTN \
        --expect-kv-dtype int4_per_token_head

A dry run proves the CLI string; it does not prove the engine used that backend. vLLM
will fall back when a backend rejects a configuration, and a silent fallback would turn
this study's controlled variable back into a confound -- the exact failure the pin exists
to prevent. So parse the server log and fail closed.
"""
import argparse, json, pathlib, re, sys

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-log", required=True, type=pathlib.Path)
    ap.add_argument("--expect-backend", required=True)
    ap.add_argument("--expect-kv-dtype", required=True)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()

    if not a.server_log.exists():
        sys.exit(f"server log not found: {a.server_log}")
    text = a.server_log.read_text(errors="replace")

    backends = re.findall(r"Using (\w+) backend", text) + \
               re.findall(r"attention backend[:= ]+(\w+)", text, re.I)
    kv = re.findall(r"kv[_-]cache[_-]dtype[:= ]+(\S+)", text, re.I)
    cap = re.findall(r"GPU KV cache size[:= ]+([\d,]+)", text, re.I)

    problems = []
    if not backends:
        problems.append("no attention backend line found in the server log")
    elif a.expect_backend not in backends:
        problems.append(f"backend mismatch: expected {a.expect_backend}, log shows {sorted(set(backends))}")
    if kv and not any(a.expect_kv_dtype in k for k in kv):
        problems.append(f"kv dtype mismatch: expected {a.expect_kv_dtype}, log shows {sorted(set(kv))}")

    result = {
        "status": "FAIL" if problems else "PASS",
        "expected_backend": a.expect_backend,
        "expected_kv_dtype": a.expect_kv_dtype,
        "observed_backends": sorted(set(backends)),
        "observed_kv_dtypes": sorted(set(kv)),
        "kv_cache_tokens": cap[-1].replace(",", "") if cap else None,
        "problems": problems,
    }
    print(json.dumps(result, indent=2))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=2) + "\n")
    return 1 if problems else 0

if __name__ == "__main__":
    raise SystemExit(main())
