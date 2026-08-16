#!/home/ducct/repos/vllm/.venv/bin/python
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("rivf26_test_gpqa", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GPQAHarnessTest(unittest.TestCase):
    def test_adapter_does_not_consume_bench_dataset_options(self) -> None:
        module = load_module(ROOT / "scripts/accuracy/run_gpqa.py")
        adapter = module.build_adapter_parser()
        _, bench_argv = adapter.parse_known_args(
            [
                "--gpqa-local-csv", "/tmp/gpqa.csv",
                "--local-tokenizer", "/tmp/model",
                "--gpqa-expected-sha256", "digest",
                "--gpqa-revision", "revision",
                "--sampling-top-k", "20",
                "--dataset", "gpqa",
                "--gpqa-dataset", "Idavidrein/gpqa",
                "--gpqa-config", "gpqa_diamond",
            ]
        )
        self.assertEqual(
            bench_argv,
            [
                "--dataset", "gpqa",
                "--gpqa-dataset", "Idavidrein/gpqa",
                "--gpqa-config", "gpqa_diamond",
            ],
        )

    def test_canonical_csv_validation(self) -> None:
        module = load_module(ROOT / "scripts/accuracy/run_gpqa.py")
        fields = list(module.REQUIRED_COLUMNS) + ["Subdomain", "Record ID"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpqa.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for index in range(2):
                    writer.writerow(
                        {
                            "Question": f"question {index}",
                            "Correct Answer": "correct",
                            "Incorrect Answer 1": "wrong one",
                            "Incorrect Answer 2": "wrong two",
                            "Incorrect Answer 3": "wrong three",
                            "Subdomain": "test",
                            "Record ID": str(index),
                        }
                    )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows, actual = module.read_rows(path, digest, 2)
            self.assertEqual(len(rows), 2)
            self.assertEqual(actual, digest)
            metadata = Path(directory) / "metadata.json"
            module.write_validation(
                metadata,
                path,
                digest,
                len(rows),
                Namespace(
                    logical_id="Idavidrein/gpqa",
                    config="gpqa_diamond",
                    split="train",
                    gpqa_revision="fixture-revision",
                ),
            )
            self.assertIn("fixture-revision", metadata.read_text())
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                module.read_rows(path, "0" * 64, 2)

    def test_repeat_score_extraction(self) -> None:
        module = load_module(ROOT / "scripts/accuracy/finalize_gpqa_run.py")
        self.assertEqual(module.extract_choice("reasoning\nAnswer: C"), "C")
        self.assertEqual(module.extract_choice("final: \\boxed{B}"), "B")
        self.assertIsNone(module.extract_choice("no final choice"))

    def test_length_distribution_and_capacity_estimates(self) -> None:
        module = load_module(ROOT / "scripts/accuracy/finalize_gpqa_run.py")
        stats = module.distribution([100, 200, 300, 400])
        self.assertEqual(stats["avg"], 250.0)
        self.assertEqual(stats["p90"], 370.0)
        self.assertEqual(stats["max"], 400)
        with tempfile.TemporaryDirectory() as directory:
            smoke = Path(directory) / "smoke.json"
            smoke.write_text(
                json.dumps({"runs": {"w16kv16": {"kv_capacity_tokens": 1000}}})
            )
            estimates = module.kv_capacity_estimates(smoke, stats)
            self.assertEqual(estimates["w16kv16"]["max_sequences_at_avg_length"], 4)
            self.assertEqual(
                estimates["w16kv16"]["max_sequences_at_90pct_capacity_p90_length"], 2
            )


if __name__ == "__main__":
    unittest.main()
