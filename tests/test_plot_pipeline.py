#!/home/ducct/repos/vllm/.venv/bin/python
from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = Path.home() / "repos/vllm/.venv/bin/python"


class PlotPipelineTest(unittest.TestCase):
    def test_four_precision_comparison_overlay(self) -> None:
        fixture = ROOT / "tests/fixtures"
        variants = ("w16kv16", "w8kv16", "w16kv8", "w8kv8")
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            inputs = []
            for precision in variants:
                output = directory_path / f"{precision}.json"
                subprocess.run(
                    [
                        str(VENV_PYTHON), str(ROOT / "analysis/build_plot_data.py"),
                        "--per-request", str(fixture / "per_request.csv"),
                        "--prometheus", str(fixture / "prometheus.jsonl"),
                        "--hbm", str(fixture / "hbm.csv"),
                        "--output", str(output),
                        "--run-id", f"fixture_{precision}",
                        "--precision", precision,
                        "--mode", "performance",
                        "--max-num-seqs", "2",
                        "--max-num-batched-tokens", "16384",
                        "--bin-seconds", "1",
                    ],
                    check=True,
                )
                inputs.append(str(output))
            svg = directory_path / "comparison.svg"
            png = directory_path / "comparison.png"
            subprocess.run(
                [
                    str(VENV_PYTHON), str(ROOT / "analysis/plot_stacked_timeline.py"),
                    *inputs,
                    "--output-svg", str(svg),
                    "--output-png", str(png),
                ],
                check=True,
            )
            ET.parse(svg)
            svg_text = svg.read_text()
            for precision in variants:
                self.assertIn(precision, svg_text)
            self.assertGreater(png.stat().st_size, 0)

    def test_compatibility_and_hbm_extensions(self) -> None:
        fixture = ROOT / "tests/fixtures"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plot_data.json"
            subprocess.run(
                [
                    str(VENV_PYTHON), str(ROOT / "analysis/build_plot_data.py"),
                    "--per-request", str(fixture / "per_request.csv"),
                    "--prometheus", str(fixture / "prometheus.jsonl"),
                    "--hbm", str(fixture / "hbm.csv"),
                    "--output", str(output),
                    "--run-id", "fixture_w16kv16_mns2",
                    "--precision", "w16kv16",
                    "--mode", "smoke",
                    "--max-num-seqs", "2",
                    "--max-num-batched-tokens", "16384",
                    "--bin-seconds", "2",
                ],
                check=True,
            )
            data = json.loads(output.read_text())
            svg = Path(directory) / "stacked.svg"
            png = Path(directory) / "stacked.png"
            subprocess.run(
                [
                    str(VENV_PYTHON), str(ROOT / "analysis/plot_stacked_timeline.py"),
                    str(output),
                    "--run-id", "fixture_w16kv16_mns2",
                    "--output-svg", str(svg),
                    "--output-png", str(png),
                ],
                check=True,
            )
            ET.parse(svg)
            self.assertGreater(png.stat().st_size, 0)
        self.assertIn("DATA", data)
        self.assertIn("TS", data)
        series = data["TS"]["Qwen3.6-35B-A3B|w16kv16"]
        run = series["runs"]["fixture_w16kv16_mns2"]
        for legacy in ("thr", "kv", "run", "wait", "pre"):
            self.assertIn(legacy, run)
        for extension in ("ttft", "tpot", "hbm", "hbm_read", "hbm_write"):
            self.assertIn(extension, run)
        lengths = {len(value) for value in run.values() if isinstance(value, list)}
        self.assertEqual(lengths, {3})
        self.assertEqual(run["hbm"][0], 22.5)
        self.assertEqual(run["hbm"][1], 40.0)
        self.assertEqual(run["tok"], 16384)


if __name__ == "__main__":
    unittest.main()
