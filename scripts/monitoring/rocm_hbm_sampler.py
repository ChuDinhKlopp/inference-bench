#!/usr/bin/env python3
"""Sample AMD MI250 HBM/UMC activity at a fixed frequency and write hbm.csv.

Out-of-process telemetry via `amdsmi` (not rocprof/rocprofv3): classic rocprof
does capture real per-kernel data, but wrapping the vLLM server with it for the
server's whole lifetime is what was causing the recurring "Worker died
unexpectedly" crash (confirmed 2026-08-18 by a decisive A/B test: identical
config, only variable is rocprof wrapping on/off -- on crashes every time,
off completed 1000/1000 requests with zero crashes). rocprofv3 is unusable
here for an unrelated reason (its hardware-counter injection collides with
PyTorch's bundled rocprofiler-sdk and SIGABRTs). This sampler polls `amdsmi`
from a fully separate process -- it never touches or instruments the vLLM
process at all, so it cannot reproduce that crash class.

amdsmi's `umc_activity` (memory-controller busy %) was earlier believed to be
frozen/non-functional on this host; that was a monitoring bug, not a real
limitation (every earlier synthetic test watched amd-smi Device index 0 while
running the workload on unrestricted `cuda:0`, which this host topology maps
to Device index 2 -- see HIP_TO_AMDSMI_DEVICE below). Confirmed working
2026-08-18 against a live vLLM server. It remains a single combined
read+write percentage -- no separate split is available on this hardware
regardless of tooling -- so hbm_read_gbps/hbm_write_gbps both report the same
combined value, clearly labeled as such in the metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


FIELDS = [
    "timestamp_epoch_s", "timestamp_iso", "elapsed_trace_s", "elapsed_experiment_s",
    "gpu_index", "gpu_name", "gpu_bus_id", "gpu_uuid", "peak_hbm_gbps",
    "hbm_read_percent", "hbm_write_percent", "hbm_aggregate_percent_raw",
    "hbm_utilization_percent", "hbm_read_gbps", "hbm_write_gbps", "hbm_aggregate_gbps",
]

# HIP_VISIBLE_DEVICES value -> amd-smi/rocm-smi/amdsmi Device index, on this
# specific host (mv-mi250-06). User-confirmed 2026-08-18 via the host's KFD
# node-ID topology: a clean pairwise swap between adjacent GCD pairs. Not a
# general ROCm fact -- re-derive on any other host.
HIP_TO_AMDSMI_DEVICE = {0: 2, 1: 3, 2: 0, 3: 1, 4: 6, 5: 7, 6: 4, 7: 5}


def _numeric(value, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def resolve_devices(amdsmi, visible_devices: list[int]) -> list[dict]:
    handles = amdsmi.amdsmi_get_processor_handles()
    devices = []
    for hip_index in visible_devices:
        amdsmi_index = HIP_TO_AMDSMI_DEVICE.get(hip_index)
        if amdsmi_index is None or amdsmi_index >= len(handles):
            raise ValueError(f"no known amd-smi Device index for HIP_VISIBLE_DEVICES={hip_index}")
        handle = handles[amdsmi_index]
        asic = amdsmi.amdsmi_get_gpu_asic_info(handle)
        vram = amdsmi.amdsmi_get_gpu_vram_info(handle)
        bdf = amdsmi.amdsmi_get_gpu_device_bdf(handle)
        uuid = amdsmi.amdsmi_get_gpu_device_uuid(handle)
        mem_clock = amdsmi.amdsmi_get_clock_info(handle, amdsmi.AmdSmiClkType.MEM)
        bit_width = _numeric(vram.get("vram_bit_width"))
        max_mem_clock_mhz = _numeric(mem_clock.get("max_clk"))
        # DDR HBM: 2 transfers/cycle. peak_GBps = clock_Hz * 2 * bit_width_bytes / 1e9.
        peak_gbps = max_mem_clock_mhz * 1e6 * 2 * (bit_width / 8) / 1e9
        devices.append({
            "hip_index": hip_index,
            "amdsmi_index": amdsmi_index,
            "handle": handle,
            "name": asic.get("market_name", "unknown"),
            "bus_id": bdf,
            "uuid": uuid,
            "peak_hbm_gbps": peak_gbps,
        })
    return devices


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument(
        "--visible-devices", required=True,
        help="comma-separated HIP_VISIBLE_DEVICES values for this replica, e.g. 0,1",
    )
    parser.add_argument("--frequency-hz", type=float, default=10.0)
    parser.add_argument("--experiment-start-epoch-s", type=float)
    args = parser.parse_args()

    if args.output_csv.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_csv}")
    if args.metadata_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.metadata_json}")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)

    import amdsmi

    amdsmi.amdsmi_init()
    visible_devices = [int(part) for part in args.visible_devices.split(",") if part.strip() != ""]
    devices = resolve_devices(amdsmi, visible_devices)

    stop = False

    def handle_signal(_signum, _frame) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    session_start_epoch_s = time.time()
    interval_s = 1.0 / args.frequency_hz
    written = 0

    with args.output_csv.open("w", newline="", buffering=1) as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        while not stop:
            loop_start = time.monotonic()
            now = time.time()
            for device in devices:
                try:
                    activity = amdsmi.amdsmi_get_gpu_activity(device["handle"])
                    umc_percent = min(100.0, max(0.0, _numeric(activity.get("umc_activity"))))
                except Exception:
                    continue
                peak_gbps = device["peak_hbm_gbps"]
                combined_gbps = peak_gbps * umc_percent / 100.0
                writer.writerow({
                    "timestamp_epoch_s": f"{now:.9f}",
                    "timestamp_iso": datetime.fromtimestamp(now, timezone.utc).isoformat(),
                    "elapsed_trace_s": f"{now - session_start_epoch_s:.9f}",
                    "elapsed_experiment_s": (
                        "" if args.experiment_start_epoch_s is None
                        else f"{now - args.experiment_start_epoch_s:.9f}"
                    ),
                    "gpu_index": device["hip_index"],
                    "gpu_name": device["name"],
                    "gpu_bus_id": device["bus_id"],
                    "gpu_uuid": device["uuid"],
                    "peak_hbm_gbps": f"{peak_gbps:.3f}",
                    "hbm_read_percent": f"{umc_percent:.3f}",
                    "hbm_write_percent": f"{umc_percent:.3f}",
                    "hbm_aggregate_percent_raw": f"{umc_percent:.3f}",
                    "hbm_utilization_percent": f"{umc_percent:.3f}",
                    "hbm_read_gbps": f"{combined_gbps:.3f}",
                    "hbm_write_gbps": f"{combined_gbps:.3f}",
                    "hbm_aggregate_gbps": f"{combined_gbps:.3f}",
                })
                written += 1
            if stop:
                break
            elapsed = time.monotonic() - loop_start
            time.sleep(max(0.0, interval_s - elapsed))

    metadata = {
        "schema_version": "rivf26.hbm.v4",
        "collector": "amdsmi (out-of-process, umc_activity)",
        "session_start_epoch_s": session_start_epoch_s,
        "session_start_utc": datetime.fromtimestamp(session_start_epoch_s, timezone.utc).isoformat(),
        "sample_rows": written,
        "gpu_count": len(devices),
        "expected_gpu_count": len(visible_devices),
        "frequency_hz": args.frequency_hz,
        "normalization": (
            "hbm_utilization_percent/hbm_aggregate_gbps come from AMD SMU "
            "average_umc_activity (combined memory-controller busy %, via amdsmi, "
            "confirmed working against a live vLLM server 2026-08-18), scaled by "
            "each device's theoretical peak HBM bandwidth (max mem clock * 2 * bus "
            "width). This hardware/tooling combination cannot separate HBM read "
            "from write bandwidth -- amdsmi only exposes the combined UMC signal; "
            "hbm_read_percent/hbm_read_gbps and hbm_write_percent/hbm_write_gbps "
            "therefore both report this same combined value, not independent "
            "measurements. gpu_index is the HIP_VISIBLE_DEVICES value for this "
            "replica; gpu_bus_id/gpu_uuid are resolved via this host's known "
            "HIP-to-amdsmi Device index mapping (see HIP_TO_AMDSMI_DEVICE in this "
            "script)."
        ),
        "devices": [
            {
                "hip_index": d["hip_index"],
                "amdsmi_index": d["amdsmi_index"],
                "name": d["name"],
                "bus_id": d["bus_id"],
                "uuid": d["uuid"],
                "peak_hbm_gbps": d["peak_hbm_gbps"],
            }
            for d in devices
        ],
    }
    args.metadata_json.write_text(json.dumps(metadata, indent=2) + "\n")
    amdsmi.amdsmi_shut_down()
    print(f"wrote {written} HBM samples to {args.output_csv}")
    return 0 if written else 2


if __name__ == "__main__":
    sys.exit(main())
