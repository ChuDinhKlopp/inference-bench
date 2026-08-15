# Part 1 telemetry and plot schema

Raw sources remain authoritative:

1. `*.per_request.csv`: one request per row with send epoch, TTFT, TPOT,
   latency, prompt tokens, and output tokens.
2. Prometheus samples: wall-clock epoch/ISO and monotonic elapsed time plus
   running, waiting, cumulative preemptions, KV usage, and token counters.
3. Iteration metrics: one engine iteration with its timestamp and scheduled
   token count. Raw vLLM log text should not be repeated in every parsed row.
4. `hbm.csv`: one row per GPU/sample with epoch and elapsed time, read/write
   percentages, read/write/aggregate GB/s, and aggregate utilization.

The existing root `latency_plots.html` is a static document with embedded
JavaScript constants, not a generic log reader. Its current schemas are:

- `DATA`: aggregate rows keyed by model/precision, with `seqs`, `tok`, `thr`,
  `req`, `tot`, `preempt`, and TTFT/TPOT/ITL mean and percentiles.
- `TS`: a model/precision-keyed object. Each value is `{bin_s, runs, golden}`;
  each named run has equal-length `thr`, `kv`, `run`, `wait`, and `pre` arrays
  plus scalar `seqs`, `tok`, and `cap`.

RIVF26 `plot_data.json` preserves `DATA` and `TS` and extends each time-series
run with `ttft`, `tpot`, `hbm`, `hbm_read`, and `hbm_write`. Old keys remain
unchanged. The renderer will produce a self-contained copy from the root HTML
template so no hand-edited values are required and historical embedded data is
not destroyed.

Timestamp convention: collectors retain `timestamp_epoch_s` and
`elapsed_s`. Offline conversion rebases all sources to experiment start epoch
from the run manifest and applies deterministic fixed-width bins.
