#!/home/ducct/repos/vllm/.venv/bin/python
"""Plot binned per-request TTFT for an RIVF26 run."""

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
    ap.add_argument("--bin-ms", type=int, default=1000)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    csv = resolve_csv(args.run)
    df = pd.read_csv(csv)
    required = {"send_epoch_s", "ttft_s"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{csv}: missing columns: {sorted(missing)}")
    df = df[np.isfinite(df["send_epoch_s"]) & np.isfinite(df["ttft_s"])].copy()
    df["prefill_done_epoch_s"] = df["send_epoch_s"] + df["ttft_s"]
    t0 = float(df["send_epoch_s"].min())
    df["end_rel_s"] = df["prefill_done_epoch_s"] - t0
    bin_s = args.bin_ms / 1000.0
    edges = np.arange(0, float(df["end_rel_s"].max()) + bin_s, bin_s)
    if len(edges) < 2:
        edges = np.array([0.0, bin_s])
    df["bin"] = pd.cut(df["end_rel_s"], bins=edges, right=False, labels=False)
    grouped = df.groupby("bin").agg(avg_ttft_s=("ttft_s", "mean"), n_requests=("ttft_s", "size"))
    grouped = grouped.reindex(range(len(edges) - 1))
    grouped["bin_center_s"] = edges[:-1] + bin_s / 2.0
    out_csv = csv.with_name(f"{csv.stem}.ttft_bin{args.bin_ms}ms.csv")
    grouped.to_csv(out_csv, index=False)
    out = args.out or csv.with_name(f"{csv.stem}.ttft_bin{args.bin_ms}ms.png")
    fig, ax = plt.subplots(figsize=(10, 4))
    observed = grouped.dropna(subset=["avg_ttft_s"])
    ax.plot(
        observed["bin_center_s"], observed["avg_ttft_s"],
        linewidth=1.2, marker=".", markersize=2.5,
    )
    ax.set(xlabel="time since first request sent (s)", ylabel=f"avg TTFT (s), {args.bin_ms}ms bins", title=csv.parent.parent.name)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out_csv}\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
