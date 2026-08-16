#!/home/ducct/repos/vllm/.venv/bin/python
"""Build the bench.py-compatible JSON consumed by record_e2e_metrics.py."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from vllm.benchmarks.datasets import SampleRequest
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncOutput
from vllm.benchmarks.serve import calculate_metrics


PERCENTILES = [50.0, 90.0, 95.0, 99.0]
OUTPUT_FIELDS = {
    "generated_text",
    "success",
    "latency",
    "output_tokens",
    "ttft",
    "itl",
    "tpot",
    "prompt_len",
    "error",
    "start_time",
    "input_audio_duration",
}


def percentile(values: list[int], value: float) -> float:
    if not values:
        return 0
    ordered = sorted(values)
    index = (len(ordered) - 1) * value / 100
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return round(
        ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower),
        2,
    )


def distribution(values: list[int]) -> dict[str, float]:
    return {
        "avg": round(sum(values) / len(values), 2) if values else 0,
        "p25": percentile(values, 25),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
    }


def load_outputs(path: Path) -> list[RequestFuncOutput]:
    outputs = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            outputs.append(
                RequestFuncOutput(
                    **{name: record[name] for name in OUTPUT_FIELDS if name in record}
                )
            )
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-summary", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--trace-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen3.6-35B-A3B")
    args = parser.parse_args()

    summary = json.loads(args.client_summary.read_text())
    outputs = load_outputs(args.responses)
    if len(outputs) != summary.get("request_count"):
        raise ValueError(
            f"response count {len(outputs)} does not match summary request_count "
            f"{summary.get('request_count')}"
        )

    max_gen_toks = int(summary["max_gen_toks"])
    requests = [
        SampleRequest(
            prompt="",
            prompt_len=output.prompt_len,
            expected_output_len=max_gen_toks,
        )
        for output in outputs
    ]
    serving_metrics, actual_output_lens = calculate_metrics(
        input_requests=requests,
        outputs=outputs,
        dur_s=float(summary["benchmark_duration_s"]),
        tokenizer=None,
        selected_percentiles=PERCENTILES,
        goodput_config_dict={},
    )
    iteration_summary = summary.get("metrics_summary") or {}
    payload = {
        "model": args.model,
        "dataset": summary["dataset"],
        "num_prompts": len(outputs),
        "num_requests": len(outputs),
        "isl_stats": distribution([output.prompt_len for output in outputs]),
        "osl_stats": distribution(actual_output_lens),
        "arrival_source": "azure_trace",
        "azure_trace_csv": str(args.trace_csv),
        "max_concurrency": None,
        "benchmark_duration_s": summary["benchmark_duration_s"],
        "serving_metrics": asdict(serving_metrics),
        "num_preempted": iteration_summary.get("num_preemptions"),
        "evaluation_metrics": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Prepared record_e2e_metrics.py input: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
