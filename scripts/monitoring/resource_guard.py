#!/home/ducct/repos/vllm/.venv/bin/python
"""Record disk/RAM headroom and interrupt a run before output becomes unsafe."""

from __future__ import annotations

import argparse
import csv
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


GIB = 1024**3


def memory_available() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-s", type=float, default=30.0)
    parser.add_argument("--min-free-gib", type=float, default=20.0)
    parser.add_argument("--min-host-available-gib", type=float, default=128.0)
    parser.add_argument("--required-pid", type=int, action="append", default=[])
    parser.add_argument("--interrupt-pid", type=int, action="append", default=[])
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    start = time.monotonic()
    fields = ["timestamp_epoch_s", "timestamp_iso", "elapsed_s", "filesystem_free_bytes", "host_available_bytes", "status", "detail"]
    with args.output.open("w", newline="", buffering=1) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        while True:
            stat = os.statvfs(args.path)
            disk_free = stat.f_bavail * stat.f_frsize
            host_free = memory_available()
            missing = [pid for pid in args.required_pid if not alive(pid)]
            status = "OK"
            detail = ""
            if disk_free < args.min_free_gib * GIB:
                status, detail = "CRITICAL", "filesystem headroom below threshold"
            elif host_free < args.min_host_available_gib * GIB:
                status, detail = "CRITICAL", "host available RAM below threshold"
            elif missing:
                status, detail = "CRITICAL", f"required monitoring/workload process exited: {missing}"
            now = datetime.now(timezone.utc)
            writer.writerow({
                "timestamp_epoch_s": f"{now.timestamp():.6f}",
                "timestamp_iso": now.isoformat(),
                "elapsed_s": f"{time.monotonic() - start:.6f}",
                "filesystem_free_bytes": disk_free,
                "host_available_bytes": host_free,
                "status": status,
                "detail": detail,
            })
            if status == "CRITICAL":
                marker = args.output.with_suffix(args.output.suffix + ".FAILED")
                marker.write_text(detail + "\n")
                for pid in args.interrupt_pid:
                    if alive(pid):
                        os.kill(pid, signal.SIGINT)
                return 2
            time.sleep(args.interval_s)


if __name__ == "__main__":
    sys.exit(main())
