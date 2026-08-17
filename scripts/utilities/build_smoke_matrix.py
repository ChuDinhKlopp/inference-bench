#!/usr/bin/env python3
"""Build a host-local four-precision long-run gate from smoke summaries."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PRECISIONS = ("w16kv16", "w8kv16", "w8kv8", "w16kv8")


def git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def build_gate(summary_paths: list[Path], root: Path) -> dict:
    runs: dict[str, dict] = {}
    errors: list[str] = []

    for path in summary_paths:
        try:
            summary = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read {path}: {exc}")
            continue

        precision = summary.get("precision")
        if precision not in PRECISIONS:
            errors.append(f"{path}: unsupported precision {precision!r}")
            continue
        if precision in runs:
            errors.append(f"duplicate smoke summary for {precision}")
            continue

        run_errors: list[str] = []
        if summary.get("status") != "PASS":
            run_errors.append("summary status is not PASS")
        if summary.get("max_num_batched_tokens") != 8192:
            run_errors.append("max_num_batched_tokens is not 8192")
        if summary.get("requests_successful") != 4:
            run_errors.append("four requests did not succeed")
        if not isinstance(summary.get("hbm_samples"), int) or summary["hbm_samples"] <= 0:
            run_errors.append("HBM telemetry is absent")
        if not isinstance(summary.get("plot_bins"), int) or summary["plot_bins"] <= 0:
            run_errors.append("plot-ready data is absent")
        if summary.get("errors"):
            run_errors.append(f"summary errors: {summary['errors']}")
        runtime_checks = summary.get("runtime_checks", [])
        if not runtime_checks:
            run_errors.append("runtime checks are absent")
        failed_checks = [
            check.get("name", "unnamed")
            for check in runtime_checks
            if check.get("result") != "PASS"
        ]
        if failed_checks:
            run_errors.append(f"failed runtime checks: {failed_checks}")

        runs[precision] = {
            "run_id": summary.get("run_id"),
            "status": "PASS" if not run_errors else "FAIL",
            "summary": str(path.resolve()),
            "requests_successful": summary.get("requests_successful"),
            "hbm_samples": summary.get("hbm_samples"),
            "plot_bins": summary.get("plot_bins"),
            "fp8_kv_scale_policy": summary.get("fp8_kv_scale_policy"),
            "log_growth": summary.get("log_growth"),
            "errors": run_errors,
        }
        errors.extend(f"{precision}: {error}" for error in run_errors)

    missing = sorted(set(PRECISIONS).difference(runs))
    if missing:
        errors.append(f"missing smoke summaries: {missing}")

    return {
        "schema_version": "rivf26.smoke_matrix.v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "git_commit": git_commit(root),
        "status": "PASS" if not errors else "FAIL",
        "max_num_batched_tokens": 8192,
        "required_precisions": list(PRECISIONS),
        "validated_data_path": [
            "per-request TTFT/TPOT",
            "Prometheus scheduler and KV metrics",
            "10 Hz four-GPU GA100 HBM telemetry",
            "rivf26.plot_data.v1 conversion",
        ],
        "runs": runs,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    gate = build_gate(args.summary, args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(gate, indent=2) + "\n")
    print(json.dumps(gate, indent=2))
    return 0 if gate["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
