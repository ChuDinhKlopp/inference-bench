#!/home/ducct/repos/vllm/.venv/bin/python
"""Create compact GPQA repeat scoring, log-growth, and run manifests."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def extract_choice(text: str, num_options: int = 4) -> str | None:
    valid = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:num_options])
    patterns = (
        r"[Aa]nswer\s*[:：]\s*\*{0,2}\(?\s*([A-Z])\s*\)?",
        r"\\boxed\{\s*\(?\s*([A-Z])\s*\)?\s*\}",
        r"\*{0,2}\(?\s*([A-Z])\s*\)?\*{0,2}\s*$",
    )
    for pattern in patterns:
        for match in reversed(re.findall(pattern, text or "")):
            if match in valid:
                return match
    for match in reversed(re.findall(r"\b([A-Z])\b", text or "")):
        if match in valid:
            return match
    return None


def tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def artifact_inventory(path: Path) -> list[dict]:
    inventory = []
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        stat = item.stat()
        inventory.append(
            {
                "relative_path": str(item.relative_to(path)),
                "size_bytes": stat.st_size,
                "modified_epoch_s": stat.st_mtime,
            }
        )
    return inventory


def git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower), 2)


def distribution(values: list[int]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "avg": round(sum(values) / len(values), 2) if values else None,
        "p25": percentile(values, 0.25),
        "p50": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def kv_capacity_estimates(
    smoke_matrix: Path,
    total_tokens: dict[str, float | int | None],
    capacity_overrides: dict[str, int] | None = None,
) -> dict[str, dict[str, int]]:
    smoke = json.loads(smoke_matrix.read_text())
    capacity_overrides = capacity_overrides or {}
    estimates = {}
    for precision, run in smoke.get("runs", {}).items():
        capacity = capacity_overrides.get(precision, int(run["kv_capacity_tokens"]))
        row = {"kv_capacity_tokens": capacity}
        for statistic in ("avg", "p90", "p95", "p99", "max"):
            tokens = total_tokens.get(statistic)
            if tokens:
                row[f"max_sequences_at_{statistic}_length"] = math.floor(capacity / float(tokens))
        p90 = total_tokens.get("p90")
        if p90:
            row["max_sequences_at_90pct_capacity_p90_length"] = math.floor(
                capacity * 0.9 / float(p90)
            )
        estimates[precision] = row
    return estimates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-results", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--post-server", type=Path, required=True)
    parser.add_argument("--server-command", type=Path, required=True)
    parser.add_argument("--client-command", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--hbm-csv", type=Path, required=True)
    parser.add_argument("--bulk-run-dir", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--run-kind", choices=("official", "length_pilot"), default="official")
    parser.add_argument("--num-samples", type=int, choices=(1, 5), default=5)
    parser.add_argument("--smoke-matrix", type=Path, required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--max-num-batched-tokens", type=int, choices=(16384,), default=16384)
    parser.add_argument("--started-epoch-s", type=float, required=True)
    parser.add_argument("--ended-epoch-s", type=float, required=True)
    args = parser.parse_args()

    expected_samples = 1 if args.run_kind == "length_pilot" else 5
    if args.num_samples != expected_samples:
        raise ValueError(
            f"{args.run_kind} requires num_samples={expected_samples}; got {args.num_samples}"
        )

    result = json.loads(args.bench_results.read_text())
    bundle = json.loads(args.generations.read_text())
    records = bundle["requests"]
    expected_records = 198 * args.num_samples
    if len(records) != expected_records:
        raise ValueError(
            f"GPQA {args.run_kind} must contain {expected_records} generation records; "
            f"found {len(records)}"
        )
    per_repeat: dict[int, list[bool]] = defaultdict(list)
    successful = 0
    output_tokens = 0
    for record in records:
        metadata = record.get("metadata") or {}
        sample_idx = int(metadata.get("sample_idx", 0))
        success = bool(record.get("success"))
        successful += int(success)
        output_tokens += int(record.get("output_tokens") or 0)
        predicted = extract_choice(record.get("generated_text") or "", int(metadata.get("num_options", 4)))
        per_repeat[sample_idx].append(success and predicted == metadata.get("answer_letter"))

    repeats = []
    for sample_idx, flags in sorted(per_repeat.items()):
        correct = sum(flags)
        repeats.append(
            {
                "repeat": sample_idx,
                "correct": correct,
                "total": len(flags),
                "pass_at_1_accuracy": correct / len(flags) if flags else None,
            }
        )
    expected_repeats = set(range(args.num_samples))
    if set(per_repeat) != expected_repeats or any(len(per_repeat[index]) != 198 for index in expected_repeats):
        raise ValueError(
            "GPQA repeat attribution is incomplete: "
            f"counts={dict(sorted((key, len(value)) for key, value in per_repeat.items()))}"
        )
    if args.run_kind == "length_pilot" and successful != expected_records:
        raise ValueError(
            f"length pilot requires 198 successful requests; found {successful}/{expected_records}"
        )

    successful_records = [record for record in records if record.get("success")]
    input_lengths = [int(record["prompt_len"]) for record in successful_records]
    output_lengths = [int(record["output_tokens"]) for record in successful_records]
    total_lengths = [input_len + output_len for input_len, output_len in zip(input_lengths, output_lengths)]
    total_distribution = distribution(total_lengths)
    server_text = args.server_log.read_text(errors="replace")
    capacity_match = re.search(r"GPU KV cache size:\s*([0-9,]+) tokens", server_text, re.I)
    runtime_kv_capacity = (
        int(capacity_match.group(1).replace(",", "")) if capacity_match else None
    )
    length_analysis = {
        "successful_requests": len(successful_records),
        "failed_requests": len(records) - len(successful_records),
        "input_tokens": distribution(input_lengths),
        "output_tokens": distribution(output_lengths),
        "total_tokens": total_distribution,
        "pilot_runtime_kv_capacity_tokens": runtime_kv_capacity,
        "kv_capacity_concurrency_estimates": kv_capacity_estimates(
            args.smoke_matrix,
            total_distribution,
            {args.precision: runtime_kv_capacity} if runtime_kv_capacity else None,
        ),
        "estimate_note": (
            "Capacity ratios are theoretical length-based bounds. The 90%-capacity/p90 row "
            "is a planning reference, not a measured scheduler optimum."
        ),
    }
    total_correct = sum(item["correct"] for item in repeats)
    duration = args.ended_epoch_s - args.started_epoch_s
    bytes_written = tree_bytes(args.bulk_run_dir)
    growth = {
        "bytes_written": bytes_written,
        "elapsed_s": duration,
        "requests": len(records),
        "successful_requests": successful,
        "generated_tokens": output_tokens,
        "gib_per_hour": bytes_written / 1024**3 / (duration / 3600) if duration > 0 else None,
        "mib_per_request": bytes_written / 1024**2 / len(records) if records else None,
        "bytes_per_generated_token": bytes_written / output_tokens if output_tokens else None,
    }
    summary = {
        "schema_version": "rivf26.gpqa_summary.v2",
        "run_id": args.run_id,
        "run_kind": args.run_kind,
        "precision": args.precision,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "source_questions": 198,
        "repeats_per_question": args.num_samples,
        "total_requests": len(records),
        "reasoning_effort": "high",
        "max_gen_toks": 32768,
        "arrival_mode": "none",
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
        "pass_at_1_repeats": repeats,
        "mean_pass_at_1": total_correct / len(records) if records else None,
        "bench_evaluation_metrics": result.get("evaluation_metrics"),
        "length_analysis": length_analysis,
        "log_growth": growth,
    }
    args.output_summary.write_text(json.dumps(summary, indent=2) + "\n")

    now = datetime.now(timezone.utc)
    root = Path(__file__).resolve().parents[2]
    preflight = json.loads(args.preflight.read_text())
    post_server = json.loads(args.post_server.read_text())
    manifest = {
        "schema_version": "rivf26.run_manifest.v1",
        "status": "PASS",
        "created_epoch_s": now.timestamp(),
        "created_iso": now.isoformat(),
        "run_id": args.run_id,
        "run_kind": args.run_kind,
        "mode": "accuracy",
        "dataset": "gpqa_diamond",
        "precision": args.precision,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "repeats_per_question": args.num_samples,
        "git_commit": git_commit(root),
        "started_epoch_s": args.started_epoch_s,
        "ended_epoch_s": args.ended_epoch_s,
        "preflight_status": preflight.get("status"),
        "post_server_status": post_server.get("status"),
        "fp8_kv_scale_policy": post_server.get("fp8_kv_scale_policy"),
        "model": preflight.get("model"),
        "commands": {
            "server": args.server_command.read_text().strip(),
            "client": args.client_command.read_text().strip(),
        },
        "bulk_run_dir": str(args.bulk_run_dir),
        "artifacts": artifact_inventory(args.bulk_run_dir),
        "log_growth": growth,
    }
    record_input = args.output_summary.parent / f"{args.run_id}_e2e_record_input.json"
    record_input.write_text(json.dumps(result, indent=2) + "\n")
    record_script = root.parent / "record_e2e_metrics.py"
    e2e_csv = root / "e2e_metrics_record.csv"
    if not record_script.is_file():
        raise FileNotFoundError(f"legacy e2e recorder is missing: {record_script}")
    subprocess.run(
        [
            sys.executable,
            str(record_script),
            str(record_input),
            "--csv",
            str(e2e_csv),
            "--server-log",
            str(args.server_log),
            "--model-suffix",
            args.precision if args.run_kind == "official" else f"{args.precision}_length_pilot",
            "--attn-backend",
            "FLASHINFER",
        ],
        check=True,
    )
    manifest["e2e_metrics"] = {
        "csv": str(e2e_csv),
        "record_input": str(record_input),
        "record_script": str(record_script),
    }
    args.output_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
