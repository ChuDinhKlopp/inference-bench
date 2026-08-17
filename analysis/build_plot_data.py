#!/home/ducct/repos/vllm/.venv/bin/python
"""Deterministically convert Part 1 raw telemetry into latency_plots data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def stats_ms(values_s: list[float], prefix: str) -> dict[str, float | None]:
    values = [value * 1000.0 for value in values_s if math.isfinite(value)]
    return {
        f"{prefix}_mean": statistics.fmean(values) if values else None,
        f"{prefix}_med": percentile(values, 0.50),
        f"{prefix}_p90": percentile(values, 0.90),
        f"{prefix}_p95": percentile(values, 0.95),
        f"{prefix}_p99": percentile(values, 0.99),
    }


def read_requests(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as stream:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def read_prometheus(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def metric(row: dict[str, Any], name: str, reduction: str = "sum") -> float | None:
    value = row.get("metrics", {}).get(name)
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    selected = value.get(reduction)
    return float(selected) if selected is not None else None


def binned_average(rows: list[tuple[int, float]], count: int, carry: bool = False) -> list[float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for index, value in rows:
        if 0 <= index < count and math.isfinite(value):
            grouped[index].append(value)
    output: list[float] = []
    previous = 0.0
    for index in range(count):
        if grouped[index]:
            previous = statistics.fmean(grouped[index])
        elif not carry:
            previous = 0.0
        output.append(round(previous, 6))
    return output


def binned_last(rows: list[tuple[int, float]], count: int) -> list[float]:
    grouped: dict[int, float] = {}
    for index, value in rows:
        if 0 <= index < count and math.isfinite(value):
            grouped[index] = value
    output, previous = [], 0.0
    for index in range(count):
        previous = grouped.get(index, previous)
        output.append(round(previous, 6))
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-request", type=Path, required=True)
    parser.add_argument("--prometheus", type=Path, required=True)
    parser.add_argument("--hbm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        choices=(8192,),
        default=8192,
        help="Must match the RIVF26 scheduler token budget",
    )
    parser.add_argument("--kv-capacity-tokens", type=int)
    parser.add_argument(
        "--bin-seconds",
        type=float,
        default=1.0,
        help="Plot timestep width; 1 s retains 10 Hz HBM and 5 Hz scheduler dynamics",
    )
    parser.add_argument("--experiment-start-epoch-s", type=float)
    args = parser.parse_args()
    if not math.isfinite(args.bin_seconds) or args.bin_seconds <= 0:
        parser.error("--bin-seconds must be a finite positive value")

    requests = read_requests(args.per_request)
    prometheus = read_prometheus(args.prometheus)
    with args.hbm.open(newline="") as stream:
        hbm = list(csv.DictReader(stream))
    if not requests or not prometheus or not hbm:
        raise ValueError("per-request, Prometheus, and HBM inputs must all be non-empty")

    epochs = [row["send_epoch_s"] for row in requests]
    epochs.extend(float(row["timestamp_epoch_s"]) for row in prometheus)
    epochs.extend(float(row["timestamp_epoch_s"]) for row in hbm)
    start = args.experiment_start_epoch_s if args.experiment_start_epoch_s is not None else min(epochs)
    end = max(
        max(row["send_epoch_s"] + row["latency_s"] for row in requests),
        max(float(row["timestamp_epoch_s"]) for row in prometheus),
        max(float(row["timestamp_epoch_s"]) for row in hbm),
    )
    count = max(1, math.floor((end - start) / args.bin_seconds) + 1)

    def index(epoch: float) -> int:
        return math.floor((epoch - start) / args.bin_seconds)

    prom_pairs: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in prometheus:
        idx = index(float(row["timestamp_epoch_s"]))
        for name, reduction in (
            ("requests_running", "sum"),
            ("requests_waiting", "sum"),
            ("preemptions_cum", "sum"),
            ("kv_cache_usage", "avg"),
            ("num_batched_tokens", "sum"),
            ("requests_processed_cum", "sum"),
        ):
            value = metric(row, name, reduction)
            if value is not None:
                prom_pairs[name].append((idx, value))

    hbm_pairs: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in hbm:
        idx = index(float(row["timestamp_epoch_s"]))
        for name in ("hbm_utilization_percent", "hbm_read_gbps", "hbm_write_gbps", "hbm_aggregate_gbps"):
            hbm_pairs[name].append((idx, float(row[name])))

    request_pairs: dict[str, list[tuple[int, float]]] = defaultdict(list)
    completion_tokens: dict[int, float] = defaultdict(float)
    arrivals = [0.0] * count
    completions = [0.0] * count
    for row in requests:
        send_idx = index(row["send_epoch_s"])
        completion_idx = index(row["send_epoch_s"] + row["latency_s"])
        if 0 <= send_idx < count:
            arrivals[send_idx] += 1
            request_pairs["ttft"].append((send_idx, row["ttft_s"] * 1000.0))
            request_pairs["tpot"].append((send_idx, row["tpot_s"] * 1000.0))
        if 0 <= completion_idx < count:
            completions[completion_idx] += 1
            completion_tokens[completion_idx] += row["output_tokens"]

    throughput = [round(completion_tokens.get(i, 0.0) / args.bin_seconds, 6) for i in range(count)]
    total_duration = max(row["send_epoch_s"] + row["latency_s"] for row in requests) - min(row["send_epoch_s"] for row in requests)
    output_tokens = sum(row["output_tokens"] for row in requests)
    prompt_tokens = sum(row["prompt_len"] for row in requests)
    ttft_s = [row["ttft_s"] for row in requests]
    tpot_s = [row["tpot_s"] for row in requests]
    aggregate = {
        "seqs": args.max_num_seqs,
        "tok": args.max_num_batched_tokens,
        "thr": output_tokens / total_duration if total_duration > 0 else None,
        "req": len(requests) / total_duration if total_duration > 0 else None,
        "tot": (output_tokens + prompt_tokens) / total_duration if total_duration > 0 else None,
        "preempt": int(max((value for _, value in prom_pairs["preemptions_cum"]), default=0)),
        **stats_ms(ttft_s, "ttft"),
        **stats_ms(tpot_s, "tpot"),
        **stats_ms(tpot_s, "itl"),
    }

    series_key = f"Qwen3.6-35B-A3B|{args.precision}"
    ts_run = {
        "seqs": args.max_num_seqs,
        "tok": args.max_num_batched_tokens,
        "cap": args.kv_capacity_tokens,
        "thr": throughput,
        "kv": binned_average(prom_pairs["kv_cache_usage"], count, carry=True),
        "run": binned_average(prom_pairs["requests_running"], count),
        "wait": binned_average(prom_pairs["requests_waiting"], count),
        "pre": binned_last(prom_pairs["preemptions_cum"], count),
        "scheduled_tokens": binned_average(prom_pairs["num_batched_tokens"], count),
        "completed_cumulative": binned_last(prom_pairs["requests_processed_cum"], count),
        "arrivals": arrivals,
        "completions": completions,
        "ttft": binned_average(request_pairs["ttft"], count),
        "tpot": binned_average(request_pairs["tpot"], count),
        "hbm": binned_average(hbm_pairs["hbm_utilization_percent"], count),
        "hbm_read": binned_average(hbm_pairs["hbm_read_gbps"], count),
        "hbm_write": binned_average(hbm_pairs["hbm_write_gbps"], count),
        "hbm_aggregate": binned_average(hbm_pairs["hbm_aggregate_gbps"], count),
    }
    output = {
        "schema_version": "rivf26.plot_data.v1",
        "run": {
            "run_id": args.run_id,
            "mode": args.mode,
            "precision": args.precision,
            "experiment_start_epoch_s": start,
            "experiment_end_epoch_s": end,
            "bin_seconds": args.bin_seconds,
        },
        "DATA": {series_key: [aggregate]},
        "TS": {series_key: {"bin_s": args.bin_seconds, "runs": {args.run_id: ts_run}, "golden": None}},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, separators=(",", ":")) + "\n")
    print(f"wrote {args.output} with {count} deterministic bins")
    return 0


if __name__ == "__main__":
    sys.exit(main())
