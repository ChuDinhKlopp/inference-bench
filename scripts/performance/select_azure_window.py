#!/usr/bin/env python3
"""Select a reproducible bursty 1,000-request Azure trace window.

Every contiguous fixed-request-count window is evaluated. To avoid selecting a
mostly idle interval containing one short burst, selection is two-stage:

1. retain the highest-load decile (the 10% shortest window durations);
2. maximize the coefficient of variation of the 999 inter-arrival times.

The shortest duration is the deterministic tie-breaker, followed by source row.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rolling_sum(values: np.ndarray, width: int) -> np.ndarray:
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    return cumulative[width:] - cumulative[:-width]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=1000)
    parser.add_argument(
        "--high-load-quantile",
        type=float,
        default=0.10,
        help="Fraction of shortest-duration windows eligible for CV ranking",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.window_size < 2:
        raise ValueError("--window-size must be at least 2")
    if not 0.0 < args.high_load_quantile <= 1.0:
        raise ValueError("--high-load-quantile must be in (0, 1]")

    frame = pd.read_csv(args.input)
    required = {"TIMESTAMP", "GeneratedTokens"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"trace is missing columns: {sorted(missing)}")

    frame = frame.loc[frame["GeneratedTokens"] > 0].copy()
    frame["SOURCE_ROW"] = frame.index.to_numpy(dtype=np.int64) + 2
    frame["_timestamp"] = pd.to_datetime(frame["TIMESTAMP"], utc=True)
    frame.sort_values(["_timestamp", "SOURCE_ROW"], kind="stable", inplace=True)
    frame.reset_index(drop=True, inplace=True)

    count = len(frame)
    width = args.window_size
    if count < width:
        raise ValueError(f"trace has {count} usable rows; need {width}")

    # pandas 3 may preserve parsed strings as datetime64[us, UTC]. Dividing its
    # raw integers by 1e9 would compress the trace by 1000x. Normalize to ns
    # explicitly before converting to epoch seconds.
    timestamp_ns = frame["_timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    timestamp_s = timestamp_ns.astype(np.float64) / 1e9
    gaps = np.diff(timestamp_s)
    gap_width = width - 1
    gap_sums = rolling_sum(gaps, gap_width)
    gap_square_sums = rolling_sum(np.square(gaps), gap_width)
    gap_means = gap_sums / gap_width
    gap_variances = np.maximum(gap_square_sums / gap_width - np.square(gap_means), 0.0)
    gap_cvs = np.divide(
        np.sqrt(gap_variances),
        gap_means,
        out=np.full_like(gap_means, np.inf),
        where=gap_means > 0,
    )

    eligible_count = max(1, math.ceil(len(gap_sums) * args.high_load_quantile))
    eligible = np.argpartition(gap_sums, eligible_count - 1)[:eligible_count]
    # np.lexsort uses the final key as primary: highest CV, then shortest
    # duration, then earliest source position for deterministic ties.
    ranked = eligible[np.lexsort((eligible, gap_sums[eligible], -gap_cvs[eligible]))]
    start = int(ranked[0])
    stop = start + width

    selected = frame.iloc[start:stop].copy()
    selected_ts = timestamp_s[start:stop]
    offsets = selected_ts - selected_ts[0]
    inter_arrivals = np.concatenate(([0.0], np.diff(selected_ts)))
    selected.insert(0, "TRACE_INDEX", np.arange(width, dtype=np.int64))
    selected["ARRIVAL_OFFSET_S"] = offsets
    selected["INTER_ARRIVAL_S"] = inter_arrivals
    selected.drop(columns=["_timestamp"], inplace=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_metadata.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(
        args.output_csv,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
        float_format="%.9f",
    )

    duration = float(offsets[-1])
    mean_gap = float(np.mean(inter_arrivals[1:]))
    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(args.input.resolve()),
            "sha256": sha256_file(args.input),
            "rows_total": int(len(pd.read_csv(args.input, usecols=["TIMESTAMP"]))),
            "rows_usable_generated_tokens_gt_zero": count,
        },
        "selection": {
            "window_size": width,
            "candidate_windows_evaluated": int(len(gap_sums)),
            "high_load_definition": "shortest-duration contiguous windows",
            "high_load_quantile": args.high_load_quantile,
            "eligible_windows": eligible_count,
            "high_load_duration_cutoff_s": float(np.max(gap_sums[eligible])),
            "burstiness_metric": "population coefficient of variation of inter-arrival times",
            "selected_cv_rank_within_high_load_set": 1,
            "selected_cv_percentile_among_all_windows": float(
                np.mean(gap_cvs <= gap_cvs[start])
            ),
            "selected_duration_percentile_among_all_windows": float(
                np.mean(gap_sums <= gap_sums[start])
            ),
            "tie_breakers": ["shorter_duration", "earlier_source_position"],
            "sorted_start_index_zero_based": start,
            "sorted_stop_index_exclusive": stop,
            "source_first_csv_row_one_based": int(selected["SOURCE_ROW"].iloc[0]),
            "source_last_csv_row_one_based": int(selected["SOURCE_ROW"].iloc[-1]),
            "first_timestamp": str(selected["TIMESTAMP"].iloc[0]),
            "last_timestamp": str(selected["TIMESTAMP"].iloc[-1]),
            "duration_s": duration,
            "mean_request_rate_rps": width / duration if duration > 0 else None,
            "mean_inter_arrival_s": mean_gap,
            "inter_arrival_cv": float(np.std(inter_arrivals[1:]) / mean_gap),
            "max_requests_in_same_timestamp": int(selected["TIMESTAMP"].value_counts().max()),
        },
        "output": {
            "path": str(args.output_csv.resolve()),
            "sha256": sha256_file(args.output_csv),
            "timestamp_semantics": "original UTC timestamp plus normalized offset from first selected request",
            "arrival_scale": 1.0,
        },
    }
    args.output_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata["selection"], indent=2))


if __name__ == "__main__":
    main()
