#!/home/ducct/repos/vllm/.venv/bin/python
"""Validate and summarize smoke-only per-layer runtime KV-scale evidence."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    records = []
    for path in sorted(args.audit_dir.glob("worker_*.jsonl")):
        with path.open() as stream:
            records.extend(json.loads(line) for line in stream if line.strip())

    scale_names = ("q_scale", "k_scale", "v_scale", "prob_scale")
    finite_positive = bool(records) and all(
        math.isfinite(float(row[name])) and float(row[name]) > 0.0
        for row in records
        for name in scale_names
    )
    summaries = {}
    for name in scale_names:
        values = [float(row[name]) for row in records]
        summaries[name] = {
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "unique_count": len(set(values)),
            "all_one": bool(values) and all(value == 1.0 for value in values),
        }
    result = {
        "schema_version": "rivf26.kv_scale_summary.v1",
        "status": "PASS" if finite_positive else "FAIL",
        "record_count": len(records),
        "worker_pids": sorted({int(row["pid"]) for row in records}),
        "layer_record_counts": dict(sorted(Counter(row["layer_name"] for row in records).items())),
        "kv_cache_dtypes": sorted({row["kv_cache_dtype"] for row in records}),
        "query_quant_enabled": sorted({bool(row["query_quant_enabled"]) for row in records}),
        "scales": summaries,
        "scientific_warning": (
            "prob_scale remained 1.0 while FP8 query quantization was enabled"
            if records
            and any(bool(row["query_quant_enabled"]) for row in records)
            and summaries["prob_scale"]["all_one"]
            else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
