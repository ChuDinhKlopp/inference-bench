#!/usr/bin/env python3
"""Extract compact, auditable Part 2 kernel summaries from profiler_out_0.txt."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROW_RE = re.compile(
    r"(?P<self_cuda>[0-9.]+)(?P<self_unit>s|ms)\s+"
    r"(?P<self_pct>[0-9.]+)%\s+"
    r"(?P<cuda_total>[0-9.]+)(?P<total_unit>s|ms)\s+"
    r"(?P<cuda_avg>[0-9.]+)(?P<avg_unit>ms|us)\s+(?P<calls>[0-9]+)\s*$"
)


def read_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        match = ROW_RE.search(line)
        if not match:
            continue
        name = line[: match.start()].strip()
        values = match.groupdict()
        self_cuda = float(values["self_cuda"])
        if values["self_unit"] == "ms":
            self_cuda /= 1000
        cuda_total = float(values["cuda_total"])
        if values["total_unit"] == "ms":
            cuda_total /= 1000
        cuda_avg = float(values["cuda_avg"])
        if values["avg_unit"] == "us":
            cuda_avg /= 1000
        rows.append({
            "name": name,
            "self_cuda_s": self_cuda,
            "self_cuda_pct": float(values["self_pct"]),
            "cuda_total_s": cuda_total,
            "cuda_avg_ms": cuda_avg,
            "calls": int(values["calls"]),
        })
    return rows


def first(rows: list[dict], needle: str) -> dict | None:
    return next((r for r in rows if needle in r["name"]), None)


def summarize(path: Path, display_path: str) -> dict:
    rows = read_rows(path)
    needles = {
        "nccl_allreduce_kernel": "ncclDevKernel_AllReduce",
        "vllm_all_reduce": "vllm::all_reduce",
        "weight_marlin_gemm": "_C::marlin_gemm",
        "moe_marlin_gemm": "_moe_C::moe_wna16_marlin_gemm",
        "attention": "vllm::unified_attention_with_output",
        "kv_cache_update": "_C_cache_ops::reshape_and_cache_flash",
    }
    return {
        "profiler_output": display_path,
        "rows": {key: first(rows, needle) for key, needle in needles.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True,
                    help="directory containing <precision>/torch_profiler/profiler_out_0.txt")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = {"schema_version": "rivf26.part2_profiler_summary.v1", "arms": {}}
    for precision in ("w8kv8", "w8kv16", "w16kv8"):
        path = args.root / precision / "torch_profiler" / "profiler_out_0.txt"
        if not path.is_file():
            raise SystemExit(f"missing profiler summary: {path}")
        result["arms"][precision] = summarize(
            path, f"{precision}/torch_profiler/profiler_out_0.txt"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
