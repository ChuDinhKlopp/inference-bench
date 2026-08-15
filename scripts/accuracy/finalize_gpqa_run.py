#!/home/ducct/repos/vllm/.venv/bin/python
"""Create compact GPQA repeat scoring, log-growth, and run manifests."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
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
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--started-epoch-s", type=float, required=True)
    parser.add_argument("--ended-epoch-s", type=float, required=True)
    args = parser.parse_args()

    result = json.loads(args.bench_results.read_text())
    bundle = json.loads(args.generations.read_text())
    records = bundle["requests"]
    if len(records) != 990:
        raise ValueError(f"full GPQA run must contain 990 generation records; found {len(records)}")
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
    expected_repeats = set(range(5))
    if set(per_repeat) != expected_repeats or any(len(per_repeat[index]) != 198 for index in expected_repeats):
        raise ValueError(
            "GPQA repeat attribution is incomplete: "
            f"counts={dict(sorted((key, len(value)) for key, value in per_repeat.items()))}"
        )
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
        "schema_version": "rivf26.gpqa_summary.v1",
        "run_id": args.run_id,
        "precision": args.precision,
        "max_num_seqs": args.max_num_seqs,
        "source_questions": 198,
        "repeats_per_question": 5,
        "total_requests": len(records),
        "reasoning_effort": "high",
        "max_gen_toks": 32768,
        "arrival_mode": "none",
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
        "pass_at_1_repeats": repeats,
        "mean_pass_at_1": total_correct / len(records) if records else None,
        "bench_evaluation_metrics": result.get("evaluation_metrics"),
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
        "mode": "accuracy",
        "dataset": "gpqa_diamond",
        "precision": args.precision,
        "max_num_seqs": args.max_num_seqs,
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
    args.output_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
