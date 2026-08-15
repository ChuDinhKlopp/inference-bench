#!/home/ducct/repos/vllm/.venv/bin/python
"""Small Part 1 client that exercises the existing timing/metrics pipeline."""

from __future__ import annotations

import argparse
import asyncio
import csv
import dataclasses
import json
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer


PROMPTS = (
    "Explain in a few sentences why GPU memory bandwidth matters for autoregressive decoding.",
    "Summarize the difference between time to first token and time per output token.",
    "Describe one reason KV-cache pressure can cause inference requests to wait.",
    "Give a concise definition of tensor parallelism in language-model inference.",
)


def prompt_length(tokenizer: AutoTokenizer, messages: list[dict[str, str]]) -> int:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return len(encoded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen3.6-35B-A3B")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--server-log-file", type=Path, required=True)
    parser.add_argument("--max-gen-toks", type=int, default=128)
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--metrics-poll-interval", type=float, default=0.2)
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    import bench

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path), local_files_only=True, trust_remote_code=False
    )
    instances = []
    prompt_lengths = []
    for index, prompt in enumerate(PROMPTS):
        messages = [{"role": "user", "content": prompt}]
        instances.append(
            bench.BenchmarkInstance(
                args=(messages, {
                    "max_gen_toks": args.max_gen_toks,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "seed": 260815 + index,
                    "request_mode": "chat",
                }),
                task_name="rivf26_smoke",
                metadata={"request_index": index},
            )
        )
        prompt_lengths.append(prompt_length(tokenizer, messages))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / "iteration_metrics"
    start_epoch_s = time.time()
    outputs, duration, metrics_summary, timings = await bench.run_benchmark(
        api_url=f"{args.base_url.rstrip('/')}/v1/chat/completions",
        base_url=args.base_url.rstrip("/"),
        model=args.model,
        instances=instances,
        prompt_lens=prompt_lengths,
        delays=None,
        max_concurrency=args.max_concurrency,
        temperature=0.0,
        timeout_s=600.0,
        server_metrics_poll_interval_s=args.metrics_poll_interval,
        server_log_file=str(args.server_log_file),
        iteration_metrics_artifact_prefix=str(prefix),
        enable_thinking=False,
        profile=False,
    )

    per_request = args.output_dir / "per_request.csv"
    with per_request.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "request_idx", "send_epoch_s", "ttft_s", "tpot_s", "latency_s",
            "prompt_len", "output_tokens",
        ])
        writer.writerows(timings)

    responses = args.output_dir / "responses.jsonl"
    with responses.open("w", encoding="utf-8") as stream:
        for index, output in enumerate(outputs):
            row = dataclasses.asdict(output)
            row["request_index"] = index
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary = {
        "schema_version": "rivf26.smoke.v1",
        "experiment_start_epoch_s": start_epoch_s,
        "duration_s": duration,
        "request_count": len(outputs),
        "successful_requests": sum(output.success for output in outputs),
        "max_gen_toks": args.max_gen_toks,
        "max_concurrency": args.max_concurrency,
        "prompt_lengths": prompt_lengths,
        "metrics_summary": dataclasses.asdict(metrics_summary) if metrics_summary else None,
    }
    (args.output_dir / "smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["successful_requests"] != summary["request_count"]:
        raise RuntimeError(f"smoke requests failed; see {responses}")


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
