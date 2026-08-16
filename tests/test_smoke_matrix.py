#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "utilities" / "build_smoke_matrix.py"
SPEC = importlib.util.spec_from_file_location("build_smoke_matrix", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SmokeMatrixTest(unittest.TestCase):
    def write_summary(self, directory: Path, precision: str, status: str = "PASS") -> Path:
        path = directory / f"{precision}.json"
        path.write_text(
            __import__("json").dumps(
                {
                    "status": status,
                    "run_id": f"smoke-{precision}",
                    "precision": precision,
                    "max_num_batched_tokens": 16384,
                    "requests_successful": 4,
                    "hbm_samples": 40,
                    "plot_bins": 2,
                    "runtime_checks": [{"name": "server_health", "result": "PASS"}],
                    "fp8_kv_scale_policy": "not_applicable",
                    "log_growth": {"gib_per_hour": 1.0},
                    "errors": [],
                }
            )
            + "\n"
        )
        return path

    def test_accepts_exact_four_passing_precisions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            paths = [self.write_summary(directory, precision) for precision in MODULE.PRECISIONS]
            gate = MODULE.build_gate(paths, ROOT)
            self.assertEqual(gate["status"], "PASS")
            self.assertEqual(set(gate["runs"]), set(MODULE.PRECISIONS))

    def test_fails_when_one_precision_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            paths = [self.write_summary(directory, precision) for precision in MODULE.PRECISIONS[:-1]]
            gate = MODULE.build_gate(paths, ROOT)
            self.assertEqual(gate["status"], "FAIL")
            self.assertTrue(any("missing smoke summaries" in error for error in gate["errors"]))


if __name__ == "__main__":
    unittest.main()
