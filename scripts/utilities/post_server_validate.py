#!/home/ducct/repos/vllm/.venv/bin/python
"""Stage B validation: block request release until the loaded server is credible."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--precision", choices=("w16kv16", "w8kv16", "w8kv8", "w16kv8"), required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--monitor-pid", type=int, action="append", default=[])
    parser.add_argument("--min-post-load-free-mib", type=int, default=1024)
    parser.add_argument(
        "--accept-fp8-kv-scale-one",
        action="store_true",
        help="Record the study decision to accept vLLM's default FP8 KV/Q/prob scale 1.0.",
    )
    args = parser.parse_args()

    checks: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "result": "PASS" if passed else "FAIL", "detail": detail})

    def warn(name: str, detail: str, evidence: list[str], *, blocking: bool) -> None:
        warnings.append(
            {
                "name": name,
                "blocking": blocking,
                "detail": detail,
                "evidence": evidence[:10],
            }
        )

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=3) as response:
            health = response.status
    except Exception as exc:  # readiness failure needs its exact exception
        health = 0
        health_error = str(exc)
    else:
        health_error = ""
    check("server_health", health == 200, f"status={health}; error={health_error}")

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/metrics", timeout=3) as response:
            metrics_text = response.read().decode("utf-8", errors="replace")
    except Exception as exc:
        metrics_text = ""
        metrics_error = str(exc)
    else:
        metrics_error = ""
    required_metric_groups = {
        "requests_running": ("vllm:num_requests_running",),
        "requests_waiting": ("vllm:num_requests_waiting",),
        # vLLM 0.27 uses kv_cache_usage_perc; older supported harness versions
        # exposed gpu_cache_usage_perc. Both have the same 0..1 semantics.
        "kv_cache_usage": (
            "vllm:kv_cache_usage_perc",
            "vllm:gpu_cache_usage_perc",
        ),
    }
    missing_metrics = [
        alias
        for alias, candidates in required_metric_groups.items()
        if not any(name in metrics_text for name in candidates)
    ]
    resolved_metrics = {
        alias: next((name for name in candidates if name in metrics_text), None)
        for alias, candidates in required_metric_groups.items()
    }
    check(
        "prometheus_metrics",
        not missing_metrics,
        f"resolved={resolved_metrics}; missing={missing_metrics}; error={metrics_error}",
    )

    log_text = args.server_log.read_text(errors="replace") if args.server_log.is_file() else ""
    batched_token_pattern = rf"--max-num-batched-tokens(?:=|\s+){args.max_num_batched_tokens}(?:\s|$)"
    batched_token_evidence = [
        match.group(0).strip()
        for match in re.finditer(batched_token_pattern, log_text, re.IGNORECASE)
    ]
    check(
        "max_num_batched_tokens_runtime_evidence",
        bool(batched_token_evidence),
        f"expected={args.max_num_batched_tokens}; evidence={batched_token_evidence[:5]}",
    )
    expected_kv = "fp8" if args.precision.endswith("kv8") else "bfloat16"
    kv_patterns = [rf"kv.?cache.?dtype[^\n]*{expected_kv}", rf"cache.?dtype[^\n]*{expected_kv}"]
    kv_evidence = [match.group(0)[:500] for pattern in kv_patterns for match in re.finditer(pattern, log_text, re.IGNORECASE)]
    check("kv_dtype_runtime_evidence", bool(kv_evidence), f"expected={expected_kv}; evidence={kv_evidence[:5]}")

    if args.precision.endswith("kv8"):
        scale_warning_fragments = (
            "accuracy drop without a proper scaling factor",
            "Checkpoint does not provide a q scaling factor",
            "Using KV cache scaling factor 1.0",
            "Using uncalibrated q_scale",
            "Disabling calculate_kv_scales for hybrid model",
            "Using default scale of 1.0 instead",
        )
        scale_evidence = [
            line[:1000]
            for line in log_text.splitlines()
            if any(fragment.lower() in line.lower() for fragment in scale_warning_fragments)
        ]
        if scale_evidence:
            warn(
                "fp8_kv_scale_accuracy_risk",
                (
                    "vLLM default FP8 KV/Q/prob scale 1.0 was explicitly accepted for this study."
                    if args.accept_fp8_kv_scale_one
                    else "A long run is blocked until scale 1.0 is explicitly accepted or calibrated scales are supplied."
                ),
                scale_evidence,
                blocking=not args.accept_fp8_kv_scale_one,
            )

    backend_evidence = [line[:500] for line in log_text.splitlines() if "triton_attn" in line.lower() and "backend" in line.lower()]
    check("attention_backend_runtime_evidence", bool(backend_evidence), f"expected=TRITON_ATTN; evidence={backend_evidence[:5]}")

    reasoning_evidence = [
        line[:1000]
        for line in log_text.splitlines()
        if "reasoning_parser" in line and "qwen3" in line.lower()
    ]
    check(
        "reasoning_parser_runtime_evidence",
        bool(reasoning_evidence),
        f"expected=qwen3; evidence={reasoning_evidence[:5]}",
    )

    expects_fp8_weights = args.precision.startswith("w8")
    quant_evidence = [line[:500] for line in log_text.splitlines() if "quant" in line.lower() and "fp8" in line.lower()]
    if expects_fp8_weights:
        check("weight_format_runtime_evidence", bool(quant_evidence), f"expected=fp8; evidence={quant_evidence[:5]}")
    else:
        model_evidence = [line[:500] for line in log_text.splitlines() if "Qwen3.6-35B-A3B" in line and "FP8" not in line]
        check("weight_format_runtime_evidence", bool(model_evidence), f"expected=BF16 directory; evidence={model_evidence[:5]}")

    capacity_patterns = (
        r"GPU KV cache size:\s*([0-9,]+) tokens",
        r"Available KV cache memory:\s*([^\n]+)",
    )
    capacity_evidence = [match.group(0) for pattern in capacity_patterns for match in re.finditer(pattern, log_text, re.IGNORECASE)]
    check("kv_capacity_runtime_evidence", bool(capacity_evidence), f"evidence={capacity_evidence[:5]}")

    gpu_rows = []
    gpu_query_error = ""
    try:
        import amdsmi

        amdsmi.amdsmi_init()
        try:
            for index, handle in enumerate(amdsmi.amdsmi_get_processor_handles()):
                vram = amdsmi.amdsmi_get_gpu_vram_info(handle)
                used_bytes = amdsmi.amdsmi_get_gpu_memory_usage(handle, amdsmi.AmdSmiMemoryType.VRAM)
                total_mib = float(vram.get("vram_size")) if isinstance(vram.get("vram_size"), (int, float)) else 0.0
                used_mib = used_bytes / (1024**2)
                gpu_rows.append({"index": index, "used_mib": round(used_mib), "free_mib": round(total_mib - used_mib)})
        finally:
            amdsmi.amdsmi_shut_down()
    except Exception as exc:
        gpu_query_error = str(exc)
    visible_devices_raw = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES") or "0,1"
    replica_indices = {int(part) for part in visible_devices_raw.split(",") if part.strip() != ""}
    replica_rows = [row for row in gpu_rows if row["index"] in replica_indices]
    gpu_ok = (
        len(gpu_rows) == 8
        and len(replica_rows) == len(replica_indices)
        and all(row["used_mib"] > 1000 and row["free_mib"] >= args.min_post_load_free_mib for row in replica_rows)
    )
    check(
        "tp2_hbm_after_load",
        gpu_ok,
        f"replica_gpus({sorted(replica_indices)})={replica_rows}; all_gpus={gpu_rows}; stderr={gpu_query_error}",
    )

    dead_monitors = [pid for pid in args.monitor_pid if not alive(pid)]
    check("monitor_processes", not dead_monitors, f"configured={args.monitor_pid}; dead={dead_monitors}")
    preflight = json.loads(args.preflight.read_text())
    result_path = Path(preflight["resources"]["result_filesystem"]["path"])
    writable = result_path.is_dir() and os.access(result_path, os.W_OK)
    stat = os.statvfs(result_path)
    free_gib = stat.f_bavail * stat.f_frsize / 1024**3
    check(
        "output_writable_and_headroom",
        writable and free_gib >= 20.0,
        f"path={result_path}; writable={writable}; free={free_gib:.1f} GiB",
    )

    preflight_batched_tokens = preflight.get("run", {}).get("max_num_batched_tokens")
    check(
        "max_num_batched_tokens_preflight_consistency",
        preflight_batched_tokens == args.max_num_batched_tokens,
        f"expected={args.max_num_batched_tokens}; preflight={preflight_batched_tokens}",
    )
    model_path = Path(preflight["model"]["local_path"])
    expected_model = "Qwen3.6-35B-A3B-FP8" if expects_fp8_weights else "Qwen3.6-35B-A3B"
    check("model_path_identity", model_path.name == expected_model, f"path={model_path}")

    failures = [item["name"] for item in checks if item["result"] == "FAIL"]
    status = "PASS" if not failures else "FAIL"
    blocking_warnings = [item["name"] for item in warnings if item["blocking"]]
    long_run_eligible = status == "PASS" and not blocking_warnings
    now = datetime.now(timezone.utc)
    report = {
        "schema_version": "rivf26.post_server.v1",
        "stage": "B",
        "status": status,
        "timestamp_epoch_s": now.timestamp(),
        "timestamp_iso": now.isoformat(),
        "run_id": args.run_id,
        "precision": args.precision,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "checks": checks,
        "failures": failures,
        "warnings": warnings,
        "blocking_warnings": blocking_warnings,
        "fp8_kv_scale_policy": (
            "vllm_default_1.0_explicitly_accepted"
            if args.precision.endswith("kv8") and args.accept_fp8_kv_scale_one
            else "not_applicable" if not args.precision.endswith("kv8") else "not_accepted"
        ),
        "long_run_eligible": long_run_eligible,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"RIVF26 Stage B validation: {status}")
    for item in checks:
        print(f"[{item['result']}] {item['name']}: {item['detail']}")
    for item in warnings:
        print(f"[WARN] {item['name']}: {item['detail']}")
    print(f"long_run_eligible={str(long_run_eligible).lower()}")
    print(status)
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
