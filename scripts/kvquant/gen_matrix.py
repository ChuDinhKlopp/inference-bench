#!/usr/bin/env python
"""Expand configs/part3_kvquant.json into the run matrix the driver executes.

    python scripts/kvquant/gen_matrix.py                  # write configs/part3_kvquant_matrix.csv
    python scripts/kvquant/gen_matrix.py --plan           # human-readable summary, no write
    python scripts/kvquant/gen_matrix.py --include-secondary

Primary arms are model x {kv16,kv8,kv4} x {prefill_heavy,decode_heavy} x max_num_seqs,
all on TRITON_ATTN so the kernel is constant and kv-cache-dtype is the only variable.
The TurboQuant arm is a different backend and is therefore opt-in.
"""

import argparse
import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = ROOT / "configs/part3_kvquant.json"
OUT = ROOT / "configs/part3_kvquant_matrix.csv"

FIELDS = [
    "run_key", "model", "model_path", "attention_architecture", "growing_kv_layers",
    "kv_dtype", "kv_cache_dtype_cli", "attention_backend", "workload", "isl", "osl",
    "num_requests", "max_num_seqs", "max_model_len", "max_num_batched_tokens",
    "tensor_parallel_size", "gpu_memory_utilization", "quantization",
    "predicted_kv_bytes_per_token", "role",
]


def build(spec, include_secondary):
    common = spec["common"]
    rows = []
    for mkey, m in spec["models"].items():
        for dkey, d in spec["kv_dtypes"].items():
            if d["role"] == "secondary":
                if not include_secondary:
                    continue
                if mkey not in d.get("models", []):
                    continue  # structurally unsupported for this architecture
            for wkey, w in spec["workloads"].items():
                if w.get("status") != "implemented":
                    continue   # e.g. prefill_heavy has no runner yet
                for mns in spec["sweep"]["max_num_seqs"]:
                    rows.append({
                        "run_key": f"{mkey}_{dkey}_{wkey}_mns{mns}",
                        "model": mkey,
                        "model_path": m["path_default"],
                        "attention_architecture": m["attention_architecture"],
                        "growing_kv_layers": f"{m['growing_kv_layers']}/{m['total_layers']}",
                        "kv_dtype": dkey,
                        "kv_cache_dtype_cli": d["cli"],
                        "attention_backend": m.get("kv_backends", {}).get(dkey, d["backend"]),
                        "workload": wkey,
                        "isl": w.get("measured_isl_avg", ""),
                        "osl": w.get("measured_osl_avg", ""),
                        "num_requests": w["num_requests"],
                        "max_num_seqs": mns,
                        "max_model_len": common["max_model_len"],
                        "max_num_batched_tokens": common["max_num_batched_tokens"],
                        "tensor_parallel_size": common["tensor_parallel_size"],
                        "gpu_memory_utilization": common["gpu_memory_utilization"],
                        "quantization": m["quantization"] or "",
                        "predicted_kv_bytes_per_token":
                            round(m["kv_bytes_per_token_bf16"] / d["compression"]),
                        "role": d["role"],
                    })
    return rows


def plan(spec, rows):
    """Print the matrix shape and the capacity arithmetic that sets the MNS ladder."""
    print(f"{len(rows)} cells\n")
    by_role = {}
    for r in rows:
        by_role.setdefault(r["role"], []).append(r)
    for role, rs in by_role.items():
        print(f"  {role}: {len(rs)} cells")
    print("\nKV footprint per token (bf16 -> fp8 -> int4), and what it implies:\n")
    print(f"  {'model':<28}{'growing KV':>12}{'bf16':>10}{'fp8':>9}{'int4':>9}")
    for mkey, m in spec["models"].items():
        b = m["kv_bytes_per_token_bf16"]
        print(f"  {m['label']:<28}{m['growing_kv_layers']}/{m['total_layers']:<9}"
              f"{b/1024:>9.1f}K{b/2048:>8.1f}K{b/4096:>8.1f}K")
    print("\n  Leverage ordering (potential capacity benefit):")
    order = sorted(spec["models"].items(), key=lambda kv: -kv[1]["kv_bytes_per_token_bf16"])
    print("    " + " > ".join(f"{m['label']} ({m['growing_kv_layers']}/{m['total_layers']})"
                              for _, m in order))
    print("\n  Tokens resident per request (ISL+OSL), used to predict the knee:")
    for wkey, w in spec["workloads"].items():
        if w.get("status") != "implemented":
            print(f"    {wkey:<16} (not yet specified - no runner)")
            continue
        i, o = w["measured_isl_avg"], w["measured_osl_avg"]
        print(f"    {wkey:<16} {i:>6} + {o:>6} = {i+o:>6} tok/req  (measured, {w['dataset']})")
    print("\n  effective_concurrency = kv_capacity_tokens / (ISL+OSL); the knee is where")
    print("  that crosses max_num_seqs. Measure capacity first (stage 0), then site the ladder.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", type=pathlib.Path, default=SPEC)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    ap.add_argument("--include-secondary", action="store_true",
                    help="also emit the TurboQuant arm (different backend; Qwen models only)")
    ap.add_argument("--plan", action="store_true", help="summarize, do not write")
    a = ap.parse_args()

    spec = json.load(open(a.spec))
    rows = build(spec, a.include_secondary)
    if not rows:
        sys.exit("no rows generated")
    if a.plan:
        plan(spec, rows)
        return 0
    with open(a.out, "w", newline="") as f:
        # explicit LF: csv defaults to \r\n, which lands CRLF in a Unix repo
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out}  ({len(rows)} cells)")
    plan(spec, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
