#!/home/ducct/repos/vllm/.venv/bin/python
"""Finalize a PubMed/Azure run with growth accounting and provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--trace-csv", type=Path, required=True)
    parser.add_argument("--client-summary", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--post-server", type=Path, required=True)
    parser.add_argument("--thinking-budget-probe", type=Path, required=True)
    parser.add_argument("--server-command", type=Path, required=True)
    parser.add_argument("--client-command", type=Path, required=True)
    parser.add_argument("--bulk-run-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--max-num-batched-tokens", type=int, choices=(16384,), default=16384)
    parser.add_argument("--started-epoch-s", type=float, required=True)
    parser.add_argument("--ended-epoch-s", type=float, required=True)
    args = parser.parse_args()

    summary = json.loads(args.client_summary.read_text())
    if summary.get("request_count") != 1000:
        raise ValueError(f"full performance run must contain 1000 requests; got {summary.get('request_count')}")
    if summary.get("failed_requests") != 0:
        raise ValueError(f"performance run has {summary.get('failed_requests')} failed requests")
    if summary.get("enable_thinking") is not True:
        raise ValueError("performance run must keep Qwen thinking enabled")
    if summary.get("thinking_token_budget") != 6144:
        raise ValueError("performance run must use thinking_token_budget=6144")
    if summary.get("client_open_file_soft_limit", 0) < 65536:
        raise ValueError("performance client open-file limit must be at least 65536")
    thinking_probe = json.loads(args.thinking_budget_probe.read_text())
    if (
        thinking_probe.get("status") != "PASS"
        or not isinstance(thinking_probe.get("measured_reasoning_tokens"), int)
        or not 0 <= thinking_probe["measured_reasoning_tokens"] <= 6144
        or thinking_probe.get("answer_nonempty") is not True
    ):
        raise ValueError("thinking-budget runtime probe did not pass")

    response_count = 0
    generated_tokens = 0
    with args.responses.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            response = json.loads(line)
            response_count += 1
            generated_tokens += int(response.get("output_tokens") or 0)
    if response_count != 1000:
        raise ValueError(f"responses artifact must contain 1000 records; got {response_count}")

    duration = args.ended_epoch_s - args.started_epoch_s
    bytes_written = tree_bytes(args.bulk_run_dir)
    growth = {
        "bytes_written": bytes_written,
        "elapsed_s": duration,
        "requests": response_count,
        "successful_requests": summary["successful_requests"],
        "generated_tokens": generated_tokens,
        "gib_per_hour": bytes_written / 1024**3 / (duration / 3600) if duration > 0 else None,
        "mib_per_request": bytes_written / 1024**2 / response_count,
        "bytes_per_generated_token": bytes_written / generated_tokens if generated_tokens else None,
    }
    summary.update(
        {
            "schema_version": "rivf26.pubmed_summary.v1",
            "run_id": args.run_id,
            "precision": args.precision,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "reasoning_effort": "low",
            "enable_thinking": True,
            "thinking_token_budget": 6144,
            "max_gen_toks": 10240,
            "arrival_mode": "azure",
            "log_growth": growth,
        }
    )
    args.client_summary.write_text(json.dumps(summary, indent=2) + "\n")

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
        "mode": "performance",
        "dataset": "ccdv/pubmed-summarization",
        "precision": args.precision,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_gen_toks": 10240,
        "reasoning_effort": "low",
        "enable_thinking": True,
        "thinking_token_budget": 6144,
        "arrival_mode": "azure",
        "git_commit": git_commit(root),
        "started_epoch_s": args.started_epoch_s,
        "ended_epoch_s": args.ended_epoch_s,
        "preflight_status": preflight.get("status"),
        "post_server_status": post_server.get("status"),
        "thinking_budget_probe": thinking_probe,
        "fp8_kv_scale_policy": post_server.get("fp8_kv_scale_policy"),
        "model": preflight.get("model"),
        "inputs": {
            "workload": str(args.workload),
            "workload_sha256": sha256(args.workload),
            "trace_csv": str(args.trace_csv),
            "trace_csv_sha256": sha256(args.trace_csv),
            "trace_rows": 1000,
        },
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
