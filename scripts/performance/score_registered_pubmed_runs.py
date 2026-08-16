#!/home/ducct/repos/vllm/.venv/bin/python
"""Score every completed PubMed run registered in e2e_metrics_record.csv."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


def registered_run_id(row: dict[str, str]) -> str:
    return (
        f"{row['timestamp']}_performance_pubmed_{row['variant']}_mns{int(row['concurrency'])}"
    )


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    bulk_root = Path(os.environ.get("RIVF26_BULK_ROOT", "/run/user/1009/ducct/rivf26"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2e-csv", type=Path, default=root / "e2e_metrics_record.csv")
    parser.add_argument(
        "--workload",
        type=Path,
        default=bulk_root / "datasets/processed/pubmed_azure_bursty_1000.jsonl",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with args.e2e_csv.open(newline="", encoding="utf-8") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row.get("dataset") == "ccdv/pubmed-summarization"
        ]
    if not rows:
        raise ValueError(f"no registered PubMed runs in {args.e2e_csv}")

    scorer = Path(__file__).with_name("score_pubmed_rouge.py")
    for row in rows:
        run_id = registered_run_id(row)
        run_dir = root / "results/part1/performance" / run_id
        manifest = run_dir / "manifest.json"
        responses = run_dir / "raw/responses.jsonl"
        output_summary = run_dir / "rouge_summary.json"
        output_per_request = run_dir / "rouge_per_request.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(f"registered run has no manifest: {manifest}")
        if not responses.is_file():
            raise FileNotFoundError(f"registered run has no responses: {responses}")
        if output_summary.exists() and not args.force:
            print(f"skip existing: {output_summary}")
            continue
        subprocess.run(
            [
                sys.executable,
                str(scorer),
                "--workload",
                str(args.workload),
                "--responses",
                str(responses),
                "--run-id",
                run_id,
                "--output-summary",
                str(output_summary),
                "--output-per-request",
                str(output_per_request),
            ],
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
