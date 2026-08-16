#!/home/ducct/repos/vllm/.venv/bin/python
"""Use the canonical local GPQA CSV with the repository's existing bench.py.

The vLLM virtualenv intentionally remains the only Python environment used by
RIVF26. It does not currently contain Hugging Face ``datasets``, while the
canonical gated CSV is already cached locally. This adapter replaces only
bench.py's dataset-loading function; request transport, Prometheus/iteration
telemetry, generation capture, and GPQA scoring remain in bench.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_COLUMNS = (
    "Question",
    "Correct Answer",
    "Incorrect Answer 1",
    "Incorrect Answer 2",
    "Incorrect Answer 3",
)


def read_rows(path: Path, expected_sha256: str, expected_rows: int) -> tuple[list[dict[str, str]], str]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"GPQA CSV SHA256 mismatch: expected {expected_sha256}, got {digest}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = [name for name in REQUIRED_COLUMNS if name not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"GPQA CSV is missing required columns: {missing}")
        rows = list(reader)
    if len(rows) != expected_rows:
        raise ValueError(f"GPQA Diamond must contain {expected_rows} rows; found {len(rows)}")
    for index, row in enumerate(rows):
        empty = [name for name in REQUIRED_COLUMNS if not (row.get(name) or "").strip()]
        if empty:
            raise ValueError(f"GPQA row {index} has empty required fields: {empty}")
    return rows, digest


def write_validation(path: Path, csv_path: Path, digest: str, row_count: int, args: argparse.Namespace) -> None:
    now = datetime.now(timezone.utc)
    payload = {
        "schema_version": "rivf26.gpqa_source.v1",
        "status": "PASS",
        "timestamp_epoch_s": now.timestamp(),
        "timestamp_iso": now.isoformat(),
        "logical_id": args.logical_id,
        "config": args.config,
        "split": args.split,
        "revision": args.gpqa_revision,
        "local_csv": str(csv_path.resolve()),
        "sha256": digest,
        "rows": row_count,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_bench(repo_root: Path):
    bench_path = repo_root / "bench.py"
    # bench.py imports repository-local helpers as top-level modules when run
    # normally. Reproduce that script execution path without changing Python.
    sys.path.insert(0, str(repo_root))
    spec = importlib.util.spec_from_file_location("rivf26_parent_bench", bench_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import existing benchmark client: {bench_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_adapter_parser() -> argparse.ArgumentParser:
    # Do not let argparse consume bench.py options as prefixes of adapter
    # options. In particular, --dataset gpqa previously abbreviated
    # --dataset-metadata-json, removed the dataset selector from bench_argv,
    # and made bench.py fall back to MMLU-Pro.
    adapter = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    adapter.add_argument("--gpqa-local-csv", type=Path, required=True)
    adapter.add_argument("--local-tokenizer", type=Path, required=True)
    adapter.add_argument("--gpqa-expected-sha256", required=True)
    adapter.add_argument("--gpqa-expected-rows", type=int, default=198)
    adapter.add_argument("--gpqa-revision", required=True)
    adapter.add_argument("--gpqa-logical-id", dest="logical_id", default="Idavidrein/gpqa")
    adapter.add_argument("--gpqa-source-config", dest="config", default="gpqa_diamond")
    adapter.add_argument("--gpqa-source-split", dest="split", default="train")
    adapter.add_argument("--sampling-top-k", type=int, required=True)
    adapter.add_argument("--dataset-metadata-json", type=Path)
    adapter.add_argument("--validate-only", action="store_true")
    return adapter


def main() -> int:
    adapter = build_adapter_parser()
    args, bench_argv = adapter.parse_known_args()

    if args.sampling_top_k != 20:
        raise ValueError(f"RIVF26 GPQA requires model-default sampling top_k=20; got {args.sampling_top_k}")
    rows, digest = read_rows(args.gpqa_local_csv, args.gpqa_expected_sha256, args.gpqa_expected_rows)
    tokenizer_files = ("tokenizer.json", "tokenizer_config.json")
    missing_tokenizer = [name for name in tokenizer_files if not (args.local_tokenizer / name).is_file()]
    if missing_tokenizer:
        raise ValueError(
            f"local tokenizer directory {args.local_tokenizer} is missing {missing_tokenizer}"
        )
    if args.dataset_metadata_json:
        write_validation(args.dataset_metadata_json, args.gpqa_local_csv, digest, len(rows), args)
    print(
        f"Validated canonical GPQA Diamond CSV: rows={len(rows)} "
        f"revision={args.gpqa_revision} sha256={digest}"
    )
    if args.validate_only:
        return 0

    repo_root = Path(__file__).resolve().parents[3]
    bench = load_bench(repo_root)

    # bench.py uses --model for both the OpenAI API model identity and local
    # prompt tokenization. Keep the served logical name for requests, but force
    # tokenizer resolution to the exact local model directory with no network.
    from transformers import AutoTokenizer

    original_from_pretrained = AutoTokenizer.from_pretrained

    def local_from_pretrained(name_or_path, *loader_args, **loader_kwargs):
        if str(name_or_path) == "Qwen3.6-35B-A3B":
            name_or_path = str(args.local_tokenizer.resolve())
            loader_kwargs["local_files_only"] = True
        return original_from_pretrained(name_or_path, *loader_args, **loader_kwargs)

    AutoTokenizer.from_pretrained = local_from_pretrained

    original_chat_request = bench.async_request_openai_chat_completions

    async def chat_request_with_top_k(*request_args, **request_kwargs):
        request_input = request_kwargs.get("request_func_input")
        if request_input is None and request_args:
            request_input = request_args[0]
        if request_input is None:
            raise RuntimeError("bench.py chat request did not provide RequestFuncInput")
        request_input.extra_body = dict(request_input.extra_body or {})
        request_input.extra_body["top_k"] = args.sampling_top_k
        return await original_chat_request(*request_args, **request_kwargs)

    bench.async_request_openai_chat_completions = chat_request_with_top_k

    def load_gpqa_from_csv(
        dataset_id: str = "Idavidrein/gpqa",
        config: str | None = "gpqa_diamond",
        split: str = "train",
        num_prompts: int | None = None,
        seed: int = 42,
        max_gen_toks: int = 4096,
    ):
        if (dataset_id, config, split) != (args.logical_id, args.config, args.split):
            raise ValueError(
                "RIVF26 GPQA source identity mismatch: "
                f"got {(dataset_id, config, split)}, expected {(args.logical_id, args.config, args.split)}"
            )
        selected = list(rows)
        random.Random(seed).shuffle(selected)
        if num_prompts is not None:
            selected = selected[:num_prompts]
        instances = []
        letters = bench.SUPERGPQA_LETTERS
        for index, row in enumerate(selected):
            options = [
                row["Correct Answer"].strip(),
                row["Incorrect Answer 1"].strip(),
                row["Incorrect Answer 2"].strip(),
                row["Incorrect Answer 3"].strip(),
            ]
            order = list(range(4))
            random.Random(seed + index).shuffle(order)
            shuffled = [options[position] for position in order]
            gold = letters[order.index(0)]
            option_block = "\n".join(
                f"({letter}) {answer}" for letter, answer in zip(letters[:4], shuffled)
            )
            prompt = bench._GPQA_INSTRUCTION + (
                f"Question: {row['Question'].strip()}\n\nOptions:\n{option_block}\n"
            )
            instances.append(
                bench.BenchmarkInstance(
                    args=(prompt, {"max_gen_toks": max_gen_toks, "request_mode": "chat"}),
                    task_name="gpqa",
                    metadata={
                        "answer_letter": gold,
                        "num_options": 4,
                        "subdomain": (row.get("Subdomain") or "").strip(),
                        "record_id": (row.get("Record ID") or "").strip(),
                    },
                )
            )
        return instances, {
            "dataset_id": args.logical_id,
            "config": args.config,
            "split": args.split,
            "revision": args.gpqa_revision,
            "schema": "canonical_csv",
            "local_csv": str(args.gpqa_local_csv.resolve()),
            "sha256": digest,
            "num_instances": len(instances),
            "max_gen_toks": max_gen_toks,
            "sampling_top_k": args.sampling_top_k,
        }

    bench.load_gpqa_instances = load_gpqa_from_csv
    sys.argv = [str(repo_root / "bench.py"), *bench_argv]
    bench.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
