#!/home/ducct/repos/vllm/.venv/bin/python
"""Validate and summarize one scheduler-budget smoke run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--bulk-run-dir", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--post-server", type=Path, required=True)
    parser.add_argument("--hbm-metadata", type=Path, required=True)
    parser.add_argument("--plot-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    client = json.loads((args.raw_dir / "smoke_summary.json").read_text())
    preflight = json.loads(args.preflight.read_text())
    post_server = json.loads(args.post_server.read_text())
    hbm = json.loads(args.hbm_metadata.read_text())
    plot = json.loads(args.plot_data.read_text())
    errors = []
    if preflight.get("status") != "PASS":
        errors.append("Stage A did not pass")
    if post_server.get("status") != "PASS" or not post_server.get("long_run_eligible"):
        errors.append("Stage B did not establish long-run eligibility")
    if client.get("request_count") != 4 or client.get("successful_requests") != 4:
        errors.append("four smoke requests did not succeed")
    if hbm.get("gpu_count") != 4 or hbm.get("sample_rows", 0) <= 0:
        errors.append("four-GPU HBM telemetry is incomplete")
    series = plot.get("TS", {}).get(f"Qwen3.6-35B-A3B|{args.precision}", {})
    runs = series.get("runs", {})
    plotted = runs.get(args.run_id)
    required_series = ("ttft", "tpot", "run", "wait", "pre", "kv", "hbm", "hbm_read", "hbm_write")
    if not plotted or any(not isinstance(plotted.get(name), list) for name in required_series):
        errors.append("plot-ready latency/scheduler/KV/HBM data is incomplete")

    duration = float(client["duration_s"])
    bytes_written = tree_bytes(args.bulk_run_dir)
    growth = {
        "bytes_written": bytes_written,
        "elapsed_s": duration,
        "requests": 4,
        "generated_tokens": sum(
            int(json.loads(line).get("output_tokens") or 0)
            for line in (args.raw_dir / "responses.jsonl").read_text().splitlines()
            if line.strip()
        ),
    }
    growth["gib_per_hour"] = bytes_written / 1024**3 / (duration / 3600) if duration > 0 else None
    growth["mib_per_request"] = bytes_written / 1024**2 / 4
    growth["bytes_per_generated_token"] = (
        bytes_written / growth["generated_tokens"] if growth["generated_tokens"] else None
    )
    output = {
        "schema_version": "rivf26.smoke_result.v2",
        "status": "PASS" if not errors else "FAIL",
        "run_id": args.run_id,
        "precision": args.precision,
        "max_num_batched_tokens": 16384,
        "requests_successful": client.get("successful_requests"),
        "hbm_samples": hbm.get("sample_rows"),
        "plot_bins": len(plotted["hbm"]) if plotted else 0,
        "runtime_checks": post_server.get("checks"),
        "fp8_kv_scale_policy": post_server.get("fp8_kv_scale_policy"),
        "log_growth": growth,
        "errors": errors,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
