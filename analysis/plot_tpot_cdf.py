#!/home/ducct/repos/vllm/.venv/bin/python
"""Plot TPOT CDF from an RIVF26 run directory or per-request CSV."""

import argparse
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", help="RIVF26 run directory or raw/per_request.csv")
    ap.add_argument("--unit", choices=("ms", "s"), default="ms")
    ap.add_argument("--xlim", type=float, nargs="+")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    csv = resolve_csv(args.run)
    df = pd.read_csv(csv)
    if "tpot_s" in df and np.any(df["tpot_s"] > 0):
        tpot = df["tpot_s"].to_numpy(dtype=float)
    elif {"latency_s", "ttft_s", "output_tokens"}.issubset(df.columns):
        output = df["output_tokens"].to_numpy(dtype=float)
        tpot = np.where(output > 1, (df["latency_s"] - df["ttft_s"]) / np.maximum(output - 1, 1), 0.0)
    else:
        raise SystemExit(f"{csv}: no usable TPOT columns")
    tpot = np.sort(tpot[np.isfinite(tpot) & (tpot > 0)]) * (1000.0 if args.unit == "ms" else 1.0)
    if not len(tpot):
        raise SystemExit(f"{csv}: no positive TPOT values")
    cdf = np.arange(1, len(tpot) + 1) / len(tpot)
    out = args.out or csv.with_name(f"{csv.stem}.tpot_cdf.png")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(tpot, cdf, linewidth=1.6, label=f"w16kv16 (n={len(tpot)})")
    ax.set(xlabel=f"TPOT ({args.unit})", ylabel="CDF", title=csv.parent.parent.name)
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
