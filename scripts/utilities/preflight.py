#!/home/ducct/repos/vllm/.venv/bin/python
"""Two-stage RIVF26 preflight with machine-readable PASS/FAIL output."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GIB = 1024**3


def run(command: list[str], timeout: int = 20) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command, text=True, capture_output=True, timeout=timeout, check=False
        )
        return {
            "command": command,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "returncode": 127, "stdout": "", "stderr": str(exc)}


def filesystem(path: Path) -> dict[str, Any]:
    stat = os.statvfs(path)
    return {
        "path": str(path.resolve()),
        "bytes_total": stat.f_frsize * stat.f_blocks,
        "bytes_free": stat.f_frsize * stat.f_bavail,
        "inodes_total": stat.f_files,
        "inodes_free": stat.f_favail,
    }


def meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.strip().split()[0]) * 1024
    return values


def dir_size(path: Path) -> int | None:
    result = run(["du", "-sb", str(path)], timeout=60)
    if result["returncode"] != 0 or not result["stdout"]:
        return None
    return int(result["stdout"].split()[0])


def port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) != 0


def matching_processes(needles: tuple[str, ...]) -> list[dict[str, Any]]:
    matches = []
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit() or int(proc_dir.name) == os.getpid():
            continue
        try:
            command = (proc_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
            status = (proc_dir / "status").read_text()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if command and any(needle in command for needle in needles):
            uid_match = next((line.split()[1] for line in status.splitlines() if line.startswith("Uid:")), "unknown")
            matches.append({"pid": int(proc_dir.name), "uid": uid_match, "command": command})
    return sorted(matches, key=lambda item: item["pid"])


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    mandatory: bool,
    detail: str,
) -> None:
    checks.append(
        {"name": name, "result": "PASS" if passed else "FAIL", "mandatory": mandatory, "detail": detail}
    )


def parse_gpu_csv(text: str) -> list[dict[str, Any]]:
    rows = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 8:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_total_mib": int(parts[2]),
                "memory_used_mib": int(parts[3]),
                "memory_free_mib": int(parts[4]),
                "utilization_gpu_percent": int(parts[5]),
                "temperature_c": int(parts[6]),
                "power_w": float(parts[7]),
            }
        )
    return rows


def main() -> int:
    script = Path(__file__).resolve()
    root = script.parents[2]
    repo = root
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("smoke", "accuracy", "performance"), required=True)
    parser.add_argument("--precision", choices=("w16kv16", "w8kv16", "w8kv8", "w16kv8"), required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--max-num-batched-tokens", type=int, choices=(16384,), default=16384)
    parser.add_argument("--estimated-output-gib", type=float, default=80.0)
    parser.add_argument("--safety-reserve-gib", type=float, default=50.0)
    parser.add_argument("--min-host-available-gib", type=float, default=256.0)
    parser.add_argument("--min-shm-free-gib", type=float)
    parser.add_argument("--min-gpu-free-mib", type=int, default=38_000)
    parser.add_argument("--port", type=int, default=8000)
    venv_bin = Path(os.environ.get("RIVF26_VENV_BIN", Path.home() / "repos/vllm/.venv/bin"))
    os.environ["PATH"] = f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    parser.add_argument("--python", default=str(venv_bin / "python"))
    default_bulk_root = Path(os.environ.get("RIVF26_BULK_ROOT", "/run/user/1009/ducct/rivf26"))
    parser.add_argument("--output-root", type=Path, default=default_bulk_root)
    args = parser.parse_args()

    precision_config = json.loads((root / "configs/precision_configs.json").read_text())
    spec = precision_config["precisions"][args.precision]
    env_key = "RIVF26_FP8_MODEL_PATH" if args.precision.startswith("w8") else "RIVF26_BF16_MODEL_PATH"
    model_path = Path(os.environ.get(env_key, spec["model_path_default"]))
    # Model weights are the only large intentional /dev/shm residents. Runtime
    # logs live under RIVF26_BULK_ROOT, while NCCL/multiprocessing only require
    # bounded IPC headroom. Keep this independent and configurable.
    min_shm_gib = args.min_shm_free_gib if args.min_shm_free_gib is not None else 32.0
    long_run = args.mode != "smoke"
    checks: list[dict[str, Any]] = []

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_dir = root / "manifests" / args.run_id
    snapshot_dir = manifest_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    git = run(["git", "-C", str(repo), "rev-parse", "HEAD"])
    git_commit = git["stdout"] if git["returncode"] == 0 else None
    add_check(checks, "git_worktree", git_commit is not None, long_run, git_commit or git["stderr"])

    out_fs = filesystem(args.output_root)
    required_bytes = int((args.estimated_output_gib + args.safety_reserve_gib) * GIB)
    add_check(
        checks,
        "result_filesystem_capacity",
        out_fs["bytes_free"] >= required_bytes,
        True,
        f"free={out_fs['bytes_free']/GIB:.1f} GiB; required={required_bytes/GIB:.1f} GiB",
    )
    inode_fraction = out_fs["inodes_free"] / max(1, out_fs["inodes_total"])
    add_check(checks, "result_filesystem_inodes", inode_fraction >= 0.05, True, f"free_fraction={inode_fraction:.3f}")

    memory = meminfo()
    host_available = memory.get("MemAvailable", 0)
    swap_total = memory.get("SwapTotal", 0)
    swap_free = memory.get("SwapFree", 0)
    swap_used_fraction = (swap_total - swap_free) / swap_total if swap_total else 0.0
    add_check(
        checks,
        "host_ram",
        host_available >= args.min_host_available_gib * GIB,
        True,
        f"available={host_available/GIB:.1f} GiB; minimum={args.min_host_available_gib:.1f} GiB",
    )
    add_check(checks, "swap_usage", swap_used_fraction < 0.95, False, f"used_fraction={swap_used_fraction:.3f}; host available RAM is the mandatory pressure gate")
    mount = run(["findmnt", "-T", str(args.output_root), "-n", "-o", "TARGET,SOURCE,FSTYPE"])
    mount_fields = mount["stdout"].split() if mount["returncode"] == 0 else []
    output_fs_type = mount_fields[-1] if mount_fields else None
    if output_fs_type == "tmpfs":
        tmpfs_required = int(args.min_host_available_gib * GIB) + required_bytes
        add_check(
            checks,
            "tmpfs_host_ram_after_estimated_output",
            host_available >= tmpfs_required,
            True,
            f"available={host_available/GIB:.1f} GiB; required host floor + output/reserve={tmpfs_required/GIB:.1f} GiB",
        )

    shm_path = Path("/dev/shm")
    shm_fs = filesystem(shm_path)
    add_check(
        checks,
        "dev_shm_capacity",
        shm_fs["bytes_free"] >= min_shm_gib * GIB,
        True,
        f"free={shm_fs['bytes_free']/GIB:.1f} GiB; minimum={min_shm_gib:.1f} GiB",
    )

    model_required = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    missing = [name for name in model_required if not (model_path / name).is_file()]
    weight_files = sorted(model_path.glob("*.safetensors")) if model_path.is_dir() else []
    model_ok = model_path.is_dir() and not missing and bool(weight_files)
    add_check(
        checks,
        "local_model",
        model_ok,
        True,
        f"path={model_path}; safetensors={len(weight_files)}; missing={missing}",
    )

    gpu_query = run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ])
    gpus = parse_gpu_csv(gpu_query["stdout"]) if gpu_query["returncode"] == 0 else []
    gpu_shape_ok = len(gpus) == 4 and all("A100" in gpu["name"] for gpu in gpus)
    add_check(checks, "gpu_inventory", gpu_shape_ok, True, f"found={[(g['index'], g['name']) for g in gpus]}; stderr={gpu_query['stderr']}")
    gpu_free_ok = gpu_shape_ok and all(gpu["memory_free_mib"] >= args.min_gpu_free_mib for gpu in gpus)
    add_check(checks, "gpu_hbm_before_server", gpu_free_ok, True, f"free_mib={[(g['index'], g['memory_free_mib']) for g in gpus]}")

    gpu_process_query = run([
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    gpu_process_lines = [line for line in gpu_process_query["stdout"].splitlines() if line.strip()]
    add_check(checks, "gpu_processes", not gpu_process_lines, True, "none" if not gpu_process_lines else "; ".join(gpu_process_lines))

    stale_servers = matching_processes(("vllm.entrypoints", "vllm serve", "api_server.py"))
    add_check(checks, "stale_vllm_servers", not stale_servers, True, "none" if not stale_servers else json.dumps(stale_servers[:20]))
    other_benchmarks = matching_processes(("bench.py",))
    add_check(
        checks,
        "other_benchmark_clients",
        not other_benchmarks,
        False,
        "none" if not other_benchmarks else json.dumps(other_benchmarks[:20]),
    )
    add_check(checks, "server_port", port_is_free("127.0.0.1", args.port), True, f"127.0.0.1:{args.port}")

    python_path = Path(args.python)
    runtime = run([str(python_path), "-c", "import torch,vllm; print(torch.__version__); print(vllm.__version__)"]) if python_path.is_file() else {"returncode": 127, "stdout": "", "stderr": "python not found"}
    add_check(checks, "vllm_runtime", runtime["returncode"] == 0, True, runtime["stdout"] or runtime["stderr"])
    extension_probe = run([
        str(python_path),
        "-c",
        (
            "import inspect,torch; import vllm._custom_ops as ops; "
            "schema=torch.ops._moe_C.topk_softmax.default._schema; "
            "wrapper=inspect.signature(ops.topk_softmax); "
            "print(schema); print(wrapper); "
            "assert len(schema.arguments)==len(wrapper.parameters), "
            "f'compiled/Python topk_softmax arity mismatch: {len(schema.arguments)} != {len(wrapper.parameters)}'"
        ),
    ]) if runtime["returncode"] == 0 else {"returncode": 127, "stdout": "", "stderr": "runtime import failed"}
    add_check(
        checks,
        "vllm_compiled_extension_api",
        extension_probe["returncode"] == 0,
        True,
        (extension_probe["stdout"] + "\n" + extension_probe["stderr"]).strip(),
    )

    nsys = shutil.which("nsys")
    add_check(checks, "hbm_sampler", nsys is not None, True, nsys or "nsys not found")

    mandatory_failures = [check["name"] for check in checks if check["mandatory"] and check["result"] == "FAIL"]
    status = "PASS" if not mandatory_failures else "FAIL"
    now = datetime.now(timezone.utc)
    report: dict[str, Any] = {
        "schema_version": "rivf26.preflight.v1",
        "stage": "A",
        "status": status,
        "timestamp_epoch_s": now.timestamp(),
        "timestamp_iso": now.isoformat(),
        "run": {
            "run_id": args.run_id,
            "mode": args.mode,
            "precision": args.precision,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "estimated_output_gib": args.estimated_output_gib,
            "safety_reserve_gib": args.safety_reserve_gib,
        },
        "git_commit": git_commit,
        "model": {
            "logical_id": spec["model_logical_id"],
            "revision": spec["model_revision"],
            "local_path": str(model_path),
            "size_bytes": dir_size(model_path) if model_path.is_dir() else None,
            "weight_files": [{"name": path.name, "size_bytes": path.stat().st_size} for path in weight_files],
        },
        "resources": {
            "result_filesystem": {**out_fs, "mount": mount["stdout"], "filesystem_type": output_fs_type},
            "dev_shm": shm_fs,
            "host": {
                "memory_total_bytes": memory.get("MemTotal"),
                "memory_available_bytes": host_available,
                "swap_total_bytes": swap_total,
                "swap_free_bytes": swap_free,
            },
            "gpus": gpus,
            "gpu_processes": gpu_process_lines,
        },
        "checks": checks,
        "mandatory_failures": mandatory_failures,
    }

    json_path = manifest_dir / "preflight.json"
    text_path = manifest_dir / "preflight.txt"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    lines = [f"RIVF26 Stage A preflight: {status}", f"run_id: {args.run_id}"]
    lines.extend(f"[{c['result']}] {'MANDATORY' if c['mandatory'] else 'ADVISORY'} {c['name']}: {c['detail']}" for c in checks)
    lines.append(status)
    text_path.write_text("\n".join(lines) + "\n")
    print(text_path.read_text(), end="")
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
