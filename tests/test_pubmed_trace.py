#!/home/ducct/repos/vllm/.venv/bin/python
from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("rivf26_test_pubmed", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PubMedTraceTest(unittest.TestCase):
    def test_frozen_workload_matches_explicit_trace(self) -> None:
        module = load_module(ROOT / "scripts/performance/run_pubmed_trace.py")
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            trace = temp / "trace.csv"
            workload = temp / "workload.jsonl"
            with trace.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["TRACE_INDEX", "ARRIVAL_OFFSET_S"])
                writer.writeheader()
                for index in range(1000):
                    writer.writerow({"TRACE_INDEX": index, "ARRIVAL_OFFSET_S": index / 10})
            with workload.open("w", encoding="utf-8") as stream:
                for index in range(1000):
                    record = {
                        "request_index": index,
                        "request_id": f"pubmed-{index:04d}",
                        "trace": {"arrival_offset_s": index / 10},
                        "request": {
                            "messages": [{"role": "user", "content": "summarize"}],
                            "prompt_tokens": 10,
                            "max_tokens": 10240,
                            "temperature": 0.0,
                            "top_p": 1.0,
                            "seed": 42,
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    }
                    stream.write(json.dumps(record) + "\n")

            records = module.load_and_validate(workload, trace)
            self.assertEqual(len(records), 1000)

            with trace.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            rows[500]["ARRIVAL_OFFSET_S"] = "50.0001"
            with trace.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "do not match"):
                module.load_and_validate(workload, trace)


if __name__ == "__main__":
    unittest.main()
