#!/usr/bin/env python3
"""Convert classic-rocprof per-kernel FETCH_SIZE/WRITE_SIZE dispatches into hbm.csv.

Classic `rocprof` (v1) is used instead of rocprofv3/amdsmi: rocprofv3's hardware-counter
injection collides with PyTorch's bundled rocprofiler-sdk (SIGABRT on launch, silent
no-op on attach); amdsmi/amd-smi/rocm-smi's UMC (memory-controller) activity telemetry
is frozen on this host regardless of privilege or workload (confirmed under both compute-
and memory-bound synthetic loads, as regular user and root). Classic rocprof uses an
older, separate instrumentation library that avoids the collision and gives real
per-kernel-dispatch FETCH_SIZE (KB read) / WRITE_SIZE (KB written), which this script
bins into a time series and converts into GB/s using each GPU's theoretical peak HBM
bandwidth (queried once via `amdsmi`, which is reliable for static device info -- only
the dynamic activity gauges are broken on this host).

rocprof's BeginNs/EndNs are on the same clock domain as Python's time.monotonic_ns()
(validated empirically: a 2s python-side sleep matched a 2.000s BeginNs delta between
the surrounding kernels), so a single (epoch_s, monotonic_ns) reference pair captured
right before the server launches is enough to anchor every row to wall-clock time.

rocprof's own `gpu-id` column does not correlate simply with amdsmi's processor-handle
order or with HIP_VISIBLE_DEVICES restriction (empirically: gpu-id stayed the same
regardless of HIP_VISIBLE_DEVICES) -- different ROCm tools use different internal GPU
numbering on this topology. Since every MI250 GCD on this host has identical HBM specs
(same bus width, same max memory clock), the peak-bandwidth constant used for the
percent columns does not depend on resolving that correlation, so this script reports
rocprof's raw gpu-id as gpu_index and does not claim a specific bus_id/uuid for it.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


FIELDS = [
    "timestamp_epoch_s", "timestamp_iso", "elapsed_trace_s", "elapsed_experiment_s",
    "gpu_index", "gpu_name", "gpu_bus_id", "gpu_uuid", "peak_hbm_gbps",
    "hbm_read_percent", "hbm_write_percent", "hbm_aggregate_percent_raw",
    "hbm_utilization_percent", "hbm_read_gbps", "hbm_write_gbps", "hbm_aggregate_gbps",
]


def _numeric(value, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def shared_gpu_spec() -> dict:
    """Every MI250 GCD on this host has identical HBM specs, so one query suffices."""
    import amdsmi

    amdsmi.amdsmi_init()
    try:
        handle = amdsmi.amdsmi_get_processor_handles()[0]
        asic = amdsmi.amdsmi_get_gpu_asic_info(handle)
        vram = amdsmi.amdsmi_get_gpu_vram_info(handle)
        mem_clock = amdsmi.amdsmi_get_clock_info(handle, amdsmi.AmdSmiClkType.MEM)
        bit_width = _numeric(vram.get("vram_bit_width"))
        max_mem_clock_mhz = _numeric(mem_clock.get("max_clk"))
        peak_gbps = max_mem_clock_mhz * 1e6 * 2 * (bit_width / 8) / 1e9
        return {"name": asic.get("market_name", "unknown"), "peak_hbm_gbps": peak_gbps}
    finally:
        amdsmi.amdsmi_shut_down()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-csv", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument(
        "--expected-gpu-count", type=int, required=True,
        help="number of distinct rocprof gpu-id values expected (the replica's tensor-parallel size)",
    )
    parser.add_argument("--bin-seconds", type=float, default=1.0)
    parser.add_argument("--experiment-start-epoch-s", type=float)
    args = parser.parse_args()

    if args.output_csv.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_csv}")
    if args.metadata_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.metadata_json}")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    reference = json.loads(args.reference_json.read_text())
    reference_epoch_s = float(reference["reference_epoch_s"])
    reference_monotonic_ns = int(reference["reference_monotonic_ns"])

    spec = shared_gpu_spec()

    bins: dict[int, dict[int, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {"read_bytes": 0.0, "write_bytes": 0.0})
    )
    seen_gpu_ids: set[int] = set()
    rows_read = 0
    with args.kernel_csv.open(newline="") as stream:
        for row in csv.DictReader(stream):
            rows_read += 1
            try:
                gpu_id = int(row["gpu-id"])
                begin_ns = int(row["BeginNs"])
                fetch_kb = float(row["FETCH_SIZE"])
                write_kb = float(row["WRITE_SIZE"])
            except (KeyError, ValueError):
                continue
            seen_gpu_ids.add(gpu_id)
            epoch_s = reference_epoch_s + (begin_ns - reference_monotonic_ns) / 1e9
            bin_index = int(epoch_s // args.bin_seconds)
            cell = bins[bin_index][gpu_id]
            cell["read_bytes"] += fetch_kb * 1024.0
            cell["write_bytes"] += write_kb * 1024.0

    peak_gbps = spec["peak_hbm_gbps"]
    written = 0
    with args.output_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for bin_index in sorted(bins):
            bin_mid_epoch_s = bin_index * args.bin_seconds + args.bin_seconds / 2.0
            for gpu_id, cell in bins[bin_index].items():
                read_gbps = cell["read_bytes"] / args.bin_seconds / 1e9
                write_gbps = cell["write_bytes"] / args.bin_seconds / 1e9
                aggregate_gbps = read_gbps + write_gbps
                utilization = min(100.0, max(0.0, aggregate_gbps / peak_gbps * 100.0)) if peak_gbps else 0.0
                writer.writerow({
                    "timestamp_epoch_s": f"{bin_mid_epoch_s:.9f}",
                    "timestamp_iso": datetime.fromtimestamp(bin_mid_epoch_s, timezone.utc).isoformat(),
                    "elapsed_trace_s": f"{bin_mid_epoch_s - reference_epoch_s:.9f}",
                    "elapsed_experiment_s": (
                        "" if args.experiment_start_epoch_s is None
                        else f"{bin_mid_epoch_s - args.experiment_start_epoch_s:.9f}"
                    ),
                    "gpu_index": gpu_id,
                    "gpu_name": spec["name"],
                    "gpu_bus_id": "unresolved (rocprof gpu-id numbering not correlated to amdsmi bus order)",
                    "gpu_uuid": "unresolved (rocprof gpu-id numbering not correlated to amdsmi uuid order)",
                    "peak_hbm_gbps": f"{peak_gbps:.3f}",
                    "hbm_read_percent": f"{min(100.0, read_gbps / peak_gbps * 100.0):.3f}" if peak_gbps else "0.000",
                    "hbm_write_percent": f"{min(100.0, write_gbps / peak_gbps * 100.0):.3f}" if peak_gbps else "0.000",
                    "hbm_aggregate_percent_raw": f"{aggregate_gbps / peak_gbps * 100.0:.3f}" if peak_gbps else "0.000",
                    "hbm_utilization_percent": f"{utilization:.3f}",
                    "hbm_read_gbps": f"{read_gbps:.3f}",
                    "hbm_write_gbps": f"{write_gbps:.3f}",
                    "hbm_aggregate_gbps": f"{aggregate_gbps:.3f}",
                })
                written += 1

    metadata = {
        "schema_version": "rivf26.hbm.v3",
        "collector": "classic rocprof (v1) per-kernel FETCH_SIZE/WRITE_SIZE, binned",
        "reference_epoch_s": reference_epoch_s,
        "reference_monotonic_ns": reference_monotonic_ns,
        "kernel_rows_read": rows_read,
        "sample_rows": written,
        "gpu_count": len(seen_gpu_ids),
        "expected_gpu_count": args.expected_gpu_count,
        "gpu_ids_seen": sorted(seen_gpu_ids),
        "bin_seconds": args.bin_seconds,
        "peak_hbm_gbps": peak_gbps,
        "gpu_name": spec["name"],
        "normalization": (
            "hbm_read_gbps/hbm_write_gbps come from real, independently-measured "
            "rocprof FETCH_SIZE/WRITE_SIZE hardware counters (bytes read/written) per "
            "kernel dispatch, summed per bin and divided by bin duration, then divided "
            "by the (shared, since every MI250 GCD on this host has identical HBM "
            "specs) theoretical peak HBM bandwidth (max mem clock * 2 * bus width) for "
            "the percent columns. Each kernel's full byte count is attributed to the "
            "bin containing its dispatch start time (BeginNs); kernels that straddle a "
            "bin boundary are not split. gpu_index is rocprof's own gpu-id, which does "
            "not correlate to amdsmi's processor-handle order on this host -- "
            "gpu_bus_id/gpu_uuid are intentionally left unresolved rather than guessed."
        ),
    }
    args.metadata_json.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {written} HBM bins from {rows_read} kernel dispatches to {args.output_csv}")
    return 0 if written else 2


if __name__ == "__main__":
    raise SystemExit(main())
