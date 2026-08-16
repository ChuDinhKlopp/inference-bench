#!/home/ducct/repos/vllm/.venv/bin/python
"""Offline ROUGE-1/2/L F1 scoring for a completed RIVF26 PubMed run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rouge_score import rouge_scorer


METRICS = ("rouge1", "rouge2", "rougeL")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"record in {path}:{line_number} is not an object")
            records.append(value)
    return records


def index_unique(records: list[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    indexed = {}
    for record in records:
        request_id = record.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError(f"record in {path} has no request_id")
        if request_id in indexed:
            raise ValueError(f"duplicate request_id in {path}: {request_id}")
        indexed[request_id] = record
    return indexed


def distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "mean": statistics.fmean(ordered),
        "p25": percentile(0.25),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "min": ordered[0],
        "max": ordered[-1],
    }


def score_records(
    workload_records: list[dict[str, Any]],
    response_records: list[dict[str, Any]],
    expected_count: int,
    use_stemmer: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(workload_records) != expected_count:
        raise ValueError(
            f"workload must contain {expected_count} records; found {len(workload_records)}"
        )
    if len(response_records) != expected_count:
        raise ValueError(
            f"responses must contain {expected_count} records; found {len(response_records)}"
        )

    workload_by_id = index_unique(workload_records, Path("workload"))
    responses_by_id = index_unique(response_records, Path("responses"))
    missing = sorted(set(workload_by_id) - set(responses_by_id))
    extra = sorted(set(responses_by_id) - set(workload_by_id))
    if missing or extra:
        raise ValueError(
            f"workload/response request IDs do not match: missing={missing[:5]} extra={extra[:5]}"
        )

    scorer = rouge_scorer.RougeScorer(list(METRICS), use_stemmer=use_stemmer)
    per_request = []
    values = {
        metric: {component: [] for component in ("precision", "recall", "f1")}
        for metric in METRICS
    }
    split = None
    revision = None
    for workload in sorted(workload_records, key=lambda item: int(item["request_index"])):
        request_id = workload["request_id"]
        response = responses_by_id[request_id]
        if response.get("success") is not True:
            raise ValueError(f"cannot score unsuccessful response: {request_id}")
        if int(response.get("request_index", -1)) != int(workload["request_index"]):
            raise ValueError(f"request_index mismatch for {request_id}")

        dataset = workload.get("dataset") or {}
        reference = dataset.get("reference_abstract")
        prediction = response.get("generated_text")
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError(f"missing reference_abstract for {request_id}")
        if not isinstance(prediction, str):
            raise ValueError(f"generated_text is not a string for {request_id}")
        if dataset.get("split") != "test":
            raise ValueError(f"PubMed scoring requires the test split; got {dataset.get('split')!r}")
        split = split or dataset.get("split")
        revision = revision or dataset.get("revision")
        if dataset.get("split") != split or dataset.get("revision") != revision:
            raise ValueError("workload mixes dataset splits or revisions")

        scores = scorer.score(reference, prediction)
        item = {
            "request_index": int(workload["request_index"]),
            "request_id": request_id,
            "source_index": int(dataset["source_index"]),
            "prediction_chars": len(prediction),
            "reference_chars": len(reference),
        }
        for metric in METRICS:
            score = scores[metric]
            item[f"{metric}_precision"] = score.precision
            item[f"{metric}_recall"] = score.recall
            item[f"{metric}_f1"] = score.fmeasure
            values[metric]["precision"].append(score.precision)
            values[metric]["recall"].append(score.recall)
            values[metric]["f1"].append(score.fmeasure)
        per_request.append(item)

    aggregates = {
        metric: {component: distribution(samples) for component, samples in components.items()}
        for metric, components in values.items()
    }
    dataset_info = {
        "repo": "ccdv/pubmed-summarization",
        "config": "document",
        "split": split,
        "revision": revision,
    }
    return {"dataset": dataset_info, "metrics": aggregates}, per_request


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--output-per-request", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=1000)
    parser.add_argument("--no-stemmer", action="store_true")
    args = parser.parse_args()

    workload_records = read_jsonl(args.workload)
    response_records = read_jsonl(args.responses)
    scored, per_request = score_records(
        workload_records,
        response_records,
        expected_count=args.expected_count,
        use_stemmer=not args.no_stemmer,
    )
    write_jsonl(args.output_per_request, per_request)
    now = datetime.now(timezone.utc)
    summary = {
        "schema_version": "rivf26.pubmed_rouge.v1",
        "run_id": args.run_id,
        "created_epoch_s": now.timestamp(),
        "created_iso": now.isoformat(),
        **scored,
        "request_count": len(per_request),
        "scoring": {
            "implementation": "rouge-score",
            "implementation_version": importlib.metadata.version("rouge-score"),
            "metrics": list(METRICS),
            "score_scale": "0_to_1",
            "use_stemmer": not args.no_stemmer,
            "aggregation": "macro mean of per-request scores",
            "prediction_field": "generated_text",
            "reference_field": "dataset.reference_abstract",
            "rougeL_variant": "sentence-agnostic rougeL (not rougeLsum)",
        },
        "scores": {
            f"{metric}_f1": scored["metrics"][metric]["f1"]["mean"] for metric in METRICS
        },
        "inputs": {
            "workload": str(args.workload.resolve()),
            "workload_sha256": sha256(args.workload),
            "responses": str(args.responses.resolve()),
            "responses_sha256": sha256(args.responses),
        },
        "per_request": {
            "path": str(args.output_per_request.resolve()),
            "sha256": sha256(args.output_per_request),
        },
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"{args.run_id}: "
        + " ".join(f"{name}={value:.6f}" for name, value in summary["scores"].items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
