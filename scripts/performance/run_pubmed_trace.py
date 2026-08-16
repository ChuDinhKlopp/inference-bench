#!/home/ducct/repos/vllm/.venv/bin/python
"""Replay the frozen RIVF26 PubMed workload using exact Azure offsets.

The request transport, streaming timing, TTFT/TPOT calculation, and vLLM
Prometheus collector are reused from the repository's existing bench.py.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import dataclasses
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--trace-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen3.6-35B-A3B")
    parser.add_argument("--server-log-file", type=Path)
    parser.add_argument("--server-metrics-poll-interval", type=float, default=0.2)
    parser.add_argument("--max-num-batched-tokens", type=int, choices=(16384,), default=16384)
    parser.add_argument("--timeout", type=float, default=43200.0)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def load_trace_offsets(path: Path) -> list[float]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1000:
        raise ValueError(f"Azure trace CSV must contain 1000 rows; got {len(rows)}")
    offsets = [float(row["ARRIVAL_OFFSET_S"]) for row in rows]
    if offsets[0] != 0.0 or any(right < left for left, right in zip(offsets, offsets[1:])):
        raise ValueError("Azure trace offsets must begin at zero and be monotonic")
    return offsets


def load_and_validate(path: Path, trace_csv: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON on {path}:{line_number}") from error

    if len(records) != 1000:
        raise ValueError(f"performance workload must contain 1000 requests; got {len(records)}")
    indices = [record["request_index"] for record in records]
    if indices != list(range(1000)):
        raise ValueError("request_index must be exactly 0..999")
    offsets = [float(record["trace"]["arrival_offset_s"]) for record in records]
    if offsets[0] != 0.0 or any(right < left for left, right in zip(offsets, offsets[1:])):
        raise ValueError("arrival offsets must begin at zero and be monotonic")
    if any(record["request"]["max_tokens"] != 10240 for record in records):
        raise ValueError("every PubMed performance request must use max_tokens=10240")
    if any(
        record["request"].get("chat_template_kwargs", {}).get("enable_thinking") is not False
        for record in records
    ):
        raise ValueError("performance mode requires low reasoning (enable_thinking=false)")
    if any(record["request"]["prompt_tokens"] + 10240 > 65536 for record in records):
        raise ValueError("frozen workload contains a request exceeding max_model_len=65536")
    trace_offsets = load_trace_offsets(trace_csv)
    if any(abs(workload_offset - trace_offset) > 1e-6 for workload_offset, trace_offset in zip(offsets, trace_offsets, strict=True)):
        raise ValueError("frozen workload arrival offsets do not match the supplied Azure trace CSV")
    return records


async def run(args: argparse.Namespace, records: list[dict]) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    import bench  # Reuse the existing benchmark transport and metrics path.

    instances = []
    prompt_lengths = []
    offsets = []
    for record in records:
        request = record["request"]
        instances.append(
            bench.BenchmarkInstance(
                args=(
                    request["messages"],
                    {
                        "max_gen_toks": request["max_tokens"],
                        "temperature": request["temperature"],
                        "top_p": request["top_p"],
                        "seed": request["seed"],
                        "request_mode": "chat",
                    },
                ),
                task_name="pubmed_summarization",
                metadata={"request_id": record["request_id"]},
            )
        )
        prompt_lengths.append(int(request["prompt_tokens"]))
        offsets.append(float(record["trace"]["arrival_offset_s"]))

    # run_benchmark expects delay-before-request. Derive it from normalized
    # offsets rather than re-parsing timestamps, avoiding pandas time-unit bugs.
    delays = [0.0] + [right - left for left, right in zip(offsets, offsets[1:])]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_prefix = args.output_dir / "raw" / "iteration_metrics"
    metrics_prefix.parent.mkdir(parents=True, exist_ok=True)

    outputs, duration, metrics_summary, timings = await bench.run_benchmark(
        api_url=f"{args.base_url.rstrip('/')}/v1/chat/completions",
        base_url=args.base_url.rstrip("/"),
        model=args.model,
        instances=instances,
        prompt_lens=prompt_lengths,
        delays=delays,
        # Do not impose a client semaphore: arrivals must be released on the
        # trace schedule; vLLM's max-num-seqs governs running vs waiting work.
        max_concurrency=None,
        temperature=0.0,
        timeout_s=args.timeout,
        server_metrics_poll_interval_s=args.server_metrics_poll_interval,
        server_log_file=str(args.server_log_file) if args.server_log_file else None,
        iteration_metrics_artifact_prefix=str(metrics_prefix),
        enable_thinking=False,
        profile=False,
    )

    per_request_path = args.output_dir / "raw" / "per_request.csv"
    with per_request_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "request_idx",
                "send_epoch_s",
                "ttft_s",
                "tpot_s",
                "latency_s",
                "prompt_len",
                "output_tokens",
            ]
        )
        writer.writerows(timings)

    responses_path = args.output_dir / "raw" / "responses.jsonl"
    with responses_path.open("w", encoding="utf-8") as stream:
        for record, output in zip(records, outputs, strict=True):
            response = dataclasses.asdict(output)
            response["request_index"] = record["request_index"]
            response["request_id"] = record["request_id"]
            response["scheduled_arrival_offset_s"] = record["trace"]["arrival_offset_s"]
            stream.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")

    successful = sum(output.success for output in outputs)
    summary = {
        "schema_version": 1,
        "mode": "performance",
        "dataset": "ccdv/pubmed-summarization",
        "arrival_mode": "azure",
        "trace_csv": str(args.trace_csv),
        "request_count": len(records),
        "successful_requests": successful,
        "failed_requests": len(records) - successful,
        "benchmark_duration_s": duration,
        "trace_duration_s": offsets[-1],
        "max_gen_toks": 10240,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_thinking": False,
        "client_concurrency_limit": None,
        "metrics_summary": dataclasses.asdict(metrics_summary) if metrics_summary else None,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    records = load_and_validate(args.workload, args.trace_csv)
    validation = {
        "request_count": len(records),
        "trace_duration_s": records[-1]["trace"]["arrival_offset_s"],
        "trace_csv": str(args.trace_csv),
        "max_gen_toks": 10240,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "min_prompt_tokens": min(record["request"]["prompt_tokens"] for record in records),
        "max_prompt_tokens": max(record["request"]["prompt_tokens"] for record in records),
    }
    print(json.dumps(validation, indent=2))
    if not args.validate_only:
        asyncio.run(run(args, records))


if __name__ == "__main__":
    main()
