#!/home/ducct/repos/vllm/.venv/bin/python
"""Upgrade the frozen PubMed workload from non-thinking to budgeted thinking."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--max-gen-toks", type=int, choices=(10240,), default=10240)
    parser.add_argument("--thinking-token-budget", type=int, choices=(6144,), default=6144)
    parser.add_argument("--max-model-len", type=int, default=65536)
    args = parser.parse_args()

    records = [json.loads(line) for line in args.source.read_text().splitlines() if line]
    if len(records) != 1000:
        raise ValueError(f"expected 1000 source requests; got {len(records)}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path), local_files_only=True, trust_remote_code=False
    )
    prompt_lengths: list[int] = []
    for record in records:
        request = record["request"]
        token_ids = tokenizer.apply_chat_template(
            request["messages"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        if hasattr(token_ids, "keys"):
            token_ids = token_ids["input_ids"]
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        prompt_tokens = len(token_ids)
        if prompt_tokens + args.max_gen_toks > args.max_model_len:
            raise ValueError(
                f"{record['request_id']} exceeds context: "
                f"{prompt_tokens}+{args.max_gen_toks}>{args.max_model_len}"
            )
        request.update(
            {
                "prompt_tokens": prompt_tokens,
                "max_tokens": args.max_gen_toks,
                "temperature": 1.0,
                "top_p": 0.95,
                "chat_template_kwargs": {"enable_thinking": True},
                "thinking_token_budget": args.thinking_token_budget,
            }
        )
        prompt_lengths.append(prompt_tokens)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    metadata = {
        "schema_version": "rivf26.pubmed_frozen_workload.v2",
        "created_iso": datetime.now(timezone.utc).isoformat(),
        "workload": "rivf26_part1_performance_pubmed_azure_bursty_1000",
        "conversion": "preserve messages/dataset/trace; retokenize only for thinking-enabled template",
        "attachment_policy": (
            "preserve the original seeded PubMed selection and exact Azure "
            "pairing; no article text modified or truncated"
        ),
        "seed": records[0]["request"]["seed"],
        "source": {"path": str(args.source), "sha256": sha256(args.source)},
        "output": {
            "path": str(args.output),
            "sha256": sha256(args.output),
            "bytes": args.output.stat().st_size,
        },
        "request_count": len(records),
        "dataset": {
            "repo": records[0]["dataset"]["repo"],
            "revision": records[0]["dataset"]["revision"],
            "config": records[0]["dataset"]["config"],
            "split": records[0]["dataset"]["split"],
            "file": "document/test-00000-of-00001.parquet",
            "parquet_sha256": "d315d092a3b2d9eeec9cc5869f5526ade3c5b4495693f0f8c7f1de1feeb8e6c3",
            "split_rows": 6658,
            "candidates_examined": 1001,
            "selected_source_indices": [row["dataset"]["source_index"] for row in records],
            "excluded_over_context": [{"source_index": 5059, "prompt_tokens": 101309}],
        },
        "prompt": {
            "prefix": "Summarize the following PubMed article.\n\n",
            "article_text_modified_or_truncated": False,
            "min_tokens": min(prompt_lengths),
            "median_tokens": statistics.median(prompt_lengths),
            "mean_tokens": sum(prompt_lengths) / len(prompt_lengths),
            "max_tokens": max(prompt_lengths),
        },
        "generation": {
            "reasoning_effort": "low",
            "enable_thinking": True,
            "thinking_token_budget": args.thinking_token_budget,
            "max_gen_toks": args.max_gen_toks,
            "temperature": 1.0,
            "top_p": 0.95,
            "max_model_len": args.max_model_len,
        },
        "trace": {
            "path": "traces/processed/azure_multimodal_bursty_1000.csv",
            "sha256": "3b487d345f3ae02d5a4dcfc8d303a060322d6fe03763af638cda2695d11977d0",
            "arrival_scale": 1.0,
            "duration_s": records[-1]["trace"]["arrival_offset_s"],
        },
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
