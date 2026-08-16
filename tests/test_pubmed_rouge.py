#!/home/ducct/repos/vllm/.venv/bin/python
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("rivf26_test_pubmed_rouge", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PubMedRougeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module(ROOT / "scripts/performance/score_pubmed_rouge.py")
        self.workload = [
            {
                "request_index": 0,
                "request_id": "pubmed-test-00001",
                "dataset": {
                    "repo": "ccdv/pubmed-summarization",
                    "revision": "revision",
                    "config": "document",
                    "split": "test",
                    "source_index": 1,
                    "reference_abstract": "the cat sat on the mat",
                },
            },
            {
                "request_index": 1,
                "request_id": "pubmed-test-00002",
                "dataset": {
                    "repo": "ccdv/pubmed-summarization",
                    "revision": "revision",
                    "config": "document",
                    "split": "test",
                    "source_index": 2,
                    "reference_abstract": "a second exact summary",
                },
            },
        ]
        self.responses = [
            {
                "request_index": item["request_index"],
                "request_id": item["request_id"],
                "generated_text": item["dataset"]["reference_abstract"],
                "success": True,
            }
            for item in self.workload
        ]

    def test_exact_predictions_score_one(self) -> None:
        summary, records = self.module.score_records(
            self.workload, self.responses, expected_count=2
        )
        self.assertEqual(len(records), 2)
        for metric in self.module.METRICS:
            self.assertEqual(summary["metrics"][metric]["f1"]["mean"], 1.0)

    def test_request_ids_must_match(self) -> None:
        self.responses[1]["request_id"] = "unexpected"
        with self.assertRaisesRegex(ValueError, "request IDs do not match"):
            self.module.score_records(self.workload, self.responses, expected_count=2)

    def test_unsuccessful_response_is_rejected(self) -> None:
        self.responses[1]["success"] = False
        with self.assertRaisesRegex(ValueError, "unsuccessful response"):
            self.module.score_records(self.workload, self.responses, expected_count=2)


if __name__ == "__main__":
    unittest.main()
