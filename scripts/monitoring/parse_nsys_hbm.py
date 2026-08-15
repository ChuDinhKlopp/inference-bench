#!/home/ducct/repos/vllm/.venv/bin/python
"""Convert a GA100 Nsight Systems GPU Metrics report to compact HBM CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


READ_NAME = "DRAM Read Bandwidth [Throughput %]"
WRITE_NAME = "DRAM Write Bandwidth [Throughput %]"


def export_sqlite(report: Path, sqlite_path: Path) -> None:
    result = subprocess.run(
        ["nsys", "export", "--type", "sqlite", "--output", str(sqlite_path), str(report)],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nsys export failed: {result.stderr.strip()}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help=".nsys-rep or exported .sqlite")
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--experiment-start-epoch-s", type=float)
    parser.add_argument("--keep-sqlite", action="store_true")
    args = parser.parse_args()

    source = args.input.resolve()
    temporary_sqlite = False
    if source.suffix == ".sqlite":
        sqlite_path = source
    else:
        sqlite_path = args.output_csv.with_suffix(".sqlite")
        if sqlite_path.exists():
            raise FileExistsError(f"refusing to overwrite {sqlite_path}")
        export_sqlite(source, sqlite_path)
        temporary_sqlite = True

    connection = sqlite3.connect(sqlite_path)
    connection.row_factory = sqlite3.Row
    session = connection.execute(
        "SELECT utcEpochNs, utcTime, localTime FROM TARGET_INFO_SESSION_START_TIME LIMIT 1"
    ).fetchone()
    if session is None:
        raise RuntimeError("Nsight report has no session start timestamp")

    gpu_rows = connection.execute(
        "SELECT id, name, busLocation, uuid, totalMemory, memoryBandwidth "
        "FROM TARGET_INFO_GPU ORDER BY id"
    ).fetchall()
    type_ids = [row[0] for row in connection.execute(
        "SELECT DISTINCT typeId FROM TARGET_INFO_GPU_METRICS ORDER BY typeId"
    ).fetchall()]
    gpu_ids = [int(row["id"]) for row in gpu_rows]
    if len(type_ids) != len(gpu_ids):
        raise RuntimeError(f"cannot map metric sources ({len(type_ids)}) to GPUs ({len(gpu_ids)})")
    type_to_gpu = dict(zip(type_ids, gpu_ids, strict=True))
    gpu_info = {int(row["id"]): dict(row) for row in gpu_rows}

    samples: dict[tuple[int, int], dict[str, float]] = {}
    query = """
        SELECT g.typeId, g.timestamp, i.metricName, g.value
        FROM GPU_METRICS AS g
        JOIN TARGET_INFO_GPU_METRICS AS i
          ON i.typeId = g.typeId AND i.metricId = g.metricId
        WHERE i.metricName IN (?, ?)
        ORDER BY g.typeId, g.timestamp
    """
    for row in connection.execute(query, (READ_NAME, WRITE_NAME)):
        gpu_id = type_to_gpu[int(row["typeId"])]
        item = samples.setdefault((gpu_id, int(row["timestamp"])), {})
        item["read_percent" if row["metricName"] == READ_NAME else "write_percent"] = float(row["value"])
    connection.close()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.output_csv.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_csv}")
    fields = [
        "timestamp_epoch_s", "timestamp_iso", "elapsed_trace_s", "elapsed_experiment_s",
        "gpu_index", "gpu_name", "gpu_bus_id", "gpu_uuid", "peak_hbm_gbps",
        "hbm_read_percent", "hbm_write_percent", "hbm_aggregate_percent_raw",
        "hbm_utilization_percent", "hbm_read_gbps", "hbm_write_gbps", "hbm_aggregate_gbps",
    ]
    start_epoch_s = int(session["utcEpochNs"]) / 1e9
    written = 0
    with args.output_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for (gpu_id, timestamp_ns), values in sorted(samples.items(), key=lambda pair: (pair[0][1], pair[0][0])):
            if "read_percent" not in values or "write_percent" not in values:
                continue
            info = gpu_info[gpu_id]
            peak_gbps = float(info["memoryBandwidth"]) / 1e9
            read_percent = values["read_percent"]
            write_percent = values["write_percent"]
            aggregate_raw = read_percent + write_percent
            utilization = min(100.0, max(0.0, aggregate_raw))
            elapsed_trace_s = timestamp_ns / 1e9
            epoch_s = start_epoch_s + elapsed_trace_s
            writer.writerow({
                "timestamp_epoch_s": f"{epoch_s:.9f}",
                "timestamp_iso": datetime.fromtimestamp(epoch_s, timezone.utc).isoformat(),
                "elapsed_trace_s": f"{elapsed_trace_s:.9f}",
                "elapsed_experiment_s": "" if args.experiment_start_epoch_s is None else f"{epoch_s - args.experiment_start_epoch_s:.9f}",
                "gpu_index": gpu_id,
                "gpu_name": info["name"],
                "gpu_bus_id": info["busLocation"],
                "gpu_uuid": info["uuid"],
                "peak_hbm_gbps": f"{peak_gbps:.3f}",
                "hbm_read_percent": f"{read_percent:.3f}",
                "hbm_write_percent": f"{write_percent:.3f}",
                "hbm_aggregate_percent_raw": f"{aggregate_raw:.3f}",
                "hbm_utilization_percent": f"{utilization:.3f}",
                "hbm_read_gbps": f"{peak_gbps * read_percent / 100.0:.3f}",
                "hbm_write_gbps": f"{peak_gbps * write_percent / 100.0:.3f}",
                "hbm_aggregate_gbps": f"{peak_gbps * utilization / 100.0:.3f}",
            })
            written += 1

    metadata_path = args.metadata_json or args.output_csv.with_suffix(".metadata.json")
    metadata = {
        "schema_version": "rivf26.hbm.v1",
        "source_report": str(source),
        "session_start_epoch_s": start_epoch_s,
        "session_start_utc": session["utcTime"],
        "sample_rows": written,
        "gpu_count": len(gpu_rows),
        "normalization": "read and write percentages are summed, bounded to [0,100], then multiplied by the per-device peak memoryBandwidth from TARGET_INFO_GPU",
        "devices": list(gpu_info.values()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    if temporary_sqlite and not args.keep_sqlite:
        sqlite_path.unlink()
    print(f"wrote {written} HBM samples to {args.output_csv}")
    return 0 if written else 2


if __name__ == "__main__":
    sys.exit(main())
