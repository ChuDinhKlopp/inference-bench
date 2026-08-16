#!/home/ducct/repos/vllm/.venv/bin/python
"""Freeze PubMed prompts onto an already-selected Azure arrival window."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-repo", default="ccdv/pubmed-summarization")
    parser.add_argument(
        "--dataset-revision", default="6b30a2cae59b11ed77cb19959bffccbbd18e1106"
    )
    parser.add_argument("--dataset-file", default="document/test-00000-of-00001.parquet")
    parser.add_argument("--seed", type=int, default=260815)
    parser.add_argument("--max-gen-toks", type=int, default=10240)
    parser.add_argument("--thinking-token-budget", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=65536)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace = pd.read_csv(args.trace)
    required_trace = {
        "TRACE_INDEX",
        "TIMESTAMP",
        "SOURCE_ROW",
        "ARRIVAL_OFFSET_S",
        "INTER_ARRIVAL_S",
    }
    missing = required_trace.difference(trace.columns)
    if missing:
        raise ValueError(f"selected trace is missing columns: {sorted(missing)}")
    if len(trace) != 1000:
        raise ValueError(f"performance workload requires 1000 trace rows; got {len(trace)}")
    if trace["TRACE_INDEX"].tolist() != list(range(1000)):
        raise ValueError("TRACE_INDEX must be exactly 0..999")
    if not np.all(np.diff(trace["ARRIVAL_OFFSET_S"].to_numpy(float)) >= 0):
        raise ValueError("arrival offsets are not monotonic")
    if args.max_gen_toks != 10240:
        raise ValueError("RIVF26 performance mode requires --max-gen-toks 10240")
    if args.thinking_token_budget != 1024:
        raise ValueError("RIVF26 performance mode requires --thinking-token-budget 1024")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = Path(
        hf_hub_download(
            repo_id=args.dataset_repo,
            repo_type="dataset",
            revision=args.dataset_revision,
            filename=args.dataset_file,
            cache_dir=args.cache_dir,
        )
    )
    table = pq.read_table(parquet_path)
    columns = set(table.column_names)
    if not {"article", "abstract"}.issubset(columns):
        raise ValueError(f"unexpected PubMed schema: {sorted(columns)}")
    if table.num_rows < 1000:
        raise ValueError(f"PubMed split has only {table.num_rows} rows")

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path), local_files_only=True, trust_remote_code=False
    )
    prefix = "Summarize the following PubMed article.\n\n"
    dataset_rows = table.select(["article", "abstract"]).to_pylist()
    permutation = np.random.default_rng(args.seed).permutation(table.num_rows)
    selected: list[tuple[int, dict, list[dict[str, str]], int]] = []
    excluded_over_context: list[dict[str, int]] = []

    for source_index_raw in permutation:
        source_index = int(source_index_raw)
        example = dataset_rows[source_index]
        messages = [{"role": "user", "content": prefix + example["article"]}]
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        if hasattr(token_ids, "keys"):
            token_ids = token_ids["input_ids"]
        if token_ids and isinstance(token_ids[0], list):
            if len(token_ids) != 1:
                raise ValueError("expected a single tokenized prompt")
            token_ids = token_ids[0]
        prompt_tokens = len(token_ids)
        if prompt_tokens + args.max_gen_toks > args.max_model_len:
            excluded_over_context.append(
                {"source_index": source_index, "prompt_tokens": prompt_tokens}
            )
            continue
        selected.append((source_index, example, messages, prompt_tokens))
        if len(selected) == 1000:
            break

    if len(selected) != 1000:
        raise ValueError(
            f"only {len(selected)} intact PubMed prompts fit prompt + "
            f"{args.max_gen_toks} <= {args.max_model_len}"
        )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_metadata.parent.mkdir(parents=True, exist_ok=True)
    prompt_lengths = [item[3] for item in selected]

    with args.output_jsonl.open("w", encoding="utf-8") as output:
        for request_index, (trace_row, selected_item) in enumerate(
            zip(trace.to_dict("records"), selected, strict=True)
        ):
            source_index, example, messages, prompt_tokens = selected_item
            article = example["article"]
            reference = example["abstract"]
            record = {
                "schema_version": 1,
                "request_index": request_index,
                "request_id": f"pubmed-test-{source_index:05d}",
                "trace": {
                    "trace_index": int(trace_row["TRACE_INDEX"]),
                    "source_csv_row": int(trace_row["SOURCE_ROW"]),
                    "timestamp": trace_row["TIMESTAMP"],
                    "arrival_offset_s": float(trace_row["ARRIVAL_OFFSET_S"]),
                    "inter_arrival_s": float(trace_row["INTER_ARRIVAL_S"]),
                },
                "dataset": {
                    "repo": args.dataset_repo,
                    "revision": args.dataset_revision,
                    "config": "document",
                    "split": "test",
                    "source_index": int(source_index),
                    "article": article,
                    "reference_abstract": reference,
                },
                "request": {
                    "messages": messages,
                    "prompt_tokens": prompt_tokens,
                    "max_tokens": args.max_gen_toks,
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "seed": args.seed + request_index,
                    "chat_template_kwargs": {"enable_thinking": True},
                    "thinking_token_budget": args.thinking_token_budget,
                },
            }
            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workload": "rivf26_part1_performance_pubmed_azure_bursty_1000",
        "request_count": 1000,
        "attachment_policy": (
            "seeded permutation; first 1000 intact documents satisfying "
            f"prompt_tokens + {args.max_gen_toks} <= {args.max_model_len}; "
            "paired in trace order"
        ),
        "seed": args.seed,
        "dataset": {
            "repo": args.dataset_repo,
            "revision": args.dataset_revision,
            "config": "document",
            "split": "test",
            "file": args.dataset_file,
            "parquet_sha256": sha256_file(parquet_path),
            "split_rows": table.num_rows,
            "candidates_examined": len(selected) + len(excluded_over_context),
            "selected_source_indices": [item[0] for item in selected],
            "excluded_over_context": excluded_over_context,
        },
        "prompt": {
            "prefix": prefix,
            "article_text_modified_or_truncated": False,
            "min_tokens": min(prompt_lengths),
            "median_tokens": float(np.median(prompt_lengths)),
            "max_tokens": max(prompt_lengths),
        },
        "generation": {
            "max_gen_toks": args.max_gen_toks,
            "reasoning_effort": "low",
            "enable_thinking": True,
            "thinking_token_budget": args.thinking_token_budget,
            "max_model_len": args.max_model_len,
        },
        "trace": {
            "path": str(args.trace.resolve()),
            "sha256": sha256_file(args.trace),
            "arrival_scale": 1.0,
            "duration_s": float(trace["ARRIVAL_OFFSET_S"].iloc[-1]),
        },
        "output": {
            "path": str(args.output_jsonl.resolve()),
            "bytes": args.output_jsonl.stat().st_size,
            "sha256": sha256_file(args.output_jsonl),
        },
    }
    args.output_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({
        "request_count": metadata["request_count"],
        "trace_duration_s": metadata["trace"]["duration_s"],
        "prompt_tokens": metadata["prompt"],
        "output": metadata["output"],
    }, indent=2))


if __name__ == "__main__":
    main()
