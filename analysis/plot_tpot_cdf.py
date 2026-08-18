#!/home/ducct/repos/vllm/.venv/bin/python
"""Plot TPOT CDF from an RIVF26 run directory or per-request CSV."""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def resolve_csv(spec: str) -> Path:
    path = Path(spec)
    if path.is_dir():
        path = path / "raw" / "per_request.csv"
    if not path.is_file():
        raise SystemExit(f"per-request CSV not found: {path}")
    return path


def precision_label(spec: str) -> str:
    match = re.search(r"(w16kv16|w8kv16|w16kv8|w8kv8)", str(spec))
    return match.group(1) if match else Path(spec).parent.parent.name


def load_tpot(spec: str, unit: str) -> tuple[np.ndarray, str]:
    csv = resolve_csv(spec)
    df = pd.read_csv(csv)
    if "tpot_s" in df and np.any(df["tpot_s"] > 0):
        tpot = df["tpot_s"].to_numpy(dtype=float)
    elif {"latency_s", "ttft_s", "output_tokens"}.issubset(df.columns):
        output = df["output_tokens"].to_numpy(dtype=float)
        tpot = np.where(output > 1, (df["latency_s"] - df["ttft_s"]) / np.maximum(output - 1, 1), 0.0)
    else:
        raise SystemExit(f"{csv}: no usable TPOT columns")
    scale = 1000.0 if unit == "ms" else 1.0
    values = np.sort(tpot[np.isfinite(tpot) & (tpot > 0)]) * scale
    if not len(values):
        raise SystemExit(f"{csv}: no positive TPOT values")
    return values, precision_label(spec)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", nargs="+", help="RIVF26 run directories or raw/per_request.csv files")
    ap.add_argument("--unit", choices=("ms", "s"), default="ms")
    ap.add_argument("--xlim", type=float, nargs="+")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    loaded = [load_tpot(spec, args.unit) for spec in args.run]
    out = args.out or Path("tpot_cdf.png")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tpot, label in loaded:
        cdf = np.arange(1, len(tpot) + 1) / len(tpot)
        ax.plot(tpot, cdf, linewidth=1.6, label=f"{label} (n={len(tpot)})")
    ax.set(xlabel=f"TPOT ({args.unit})", ylabel="CDF", title="RIVF26 TPOT CDF")
    ax.set_ylim(0, 1)
    if args.xlim:
        ax.set_xlim(0, args.xlim[0]) if len(args.xlim) == 1 else ax.set_xlim(*args.xlim)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
