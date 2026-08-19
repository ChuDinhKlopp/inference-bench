# Part 1 precision performance report

Builds `results/part1/performance/precision_performance_report.html` — a single
self-contained page with the four-arm precision comparison and its live charts.

This file is developer reference for the builder. For reproducing the report on
another machine — run selection, transfer traps, and what must be updated by
hand — see `guide-report.md` at the repo root.

## Rebuild

```bash
cd "$HOME/repos/inference-bench/rivf26"
python analysis/report/build_report.py
```

No arguments needed on a machine that has the runs. The script discovers one run
directory per precision, extracts the series the charts need, injects them into
`template.html`, and writes the report. Check it afterwards with:

```bash
analysis/report/tests/run.sh          # needs node
```

## What the report needs

The published HTML has **no runtime dependencies** — every value is inlined, so
viewing it elsewhere is just copying the one file. Paths matter only when
rebuilding.

Per precision arm, three files are read (~1.7 MB of a ~248 MB run directory):

| File | Feeds |
|---|---|
| `<run>/raw/per_request.csv` | TTFT CDF (5.3), TPOT CDF (5.2), ISL/OSL (5.6) |
| `<run>/plot_data.json` | stacked engine timeline (5.4) |
| `<run>/<run_id>_e2e_record_input.json` | audit of the Pareto constants (5.1) |

Plus, once: `traces/processed/azure_multimodal_bursty_1000.csv` — the Azure
arrival-window chart in §3.

Total input for every chart is about **6.8 MB**. The bulk telemetry under
`raw/` is not needed and should not be copied.

## Run discovery

Directories are matched by precision suffix:

```
results/part1/performance/*_performance_pubmed_1000_longest_<precision>_mns256_bin500ms
```

The newest match wins. Pin one explicitly when that is not what you want:

```bash
python analysis/report/build_report.py \
  --run w8kv16=results/part1/performance/20260817_182741_..._w8kv16_mns256_bin500ms
```

Other flags: `--results-dir`, `--trace`, `--template`, `--out`, `--emit-json`
(dump the extracted `cdf`/`ts`/`trace` JSON for inspection).

## Porting to another machine

`raw/` is a **symlink into `$RIVF26_BULK_ROOT`** (typically `/run/user/<uid>/…`,
which is volatile tmpfs and encodes a specific uid and username). A plain copy
leaves it dangling and `per_request.csv` unreachable, which silently costs you
three charts. Dereference when copying:

```bash
rsync -aL <src>/results/part1/performance/<run_id>/ <dst>/…/<run_id>/
```

`build_report.py` detects a dangling `raw/` and says so rather than failing
obscurely.

## Known gap: hand-written constants

The Pareto chart (5.1) and the numeric tables in the prose are **literals inside
`template.html`**, not generated from the runs. Rebuilding against different runs
updates the CDFs and the timeline while those stay at their old values — the page
still renders, so the mismatch is easy to miss.

To make that failure loud, every build audits the template's `PARETO` constants
against the measured `serving_metrics` and refuses to write if they have drifted
beyond 0.2 %:

```
FAIL: template.html's hand-written PARETO constants no longer match the measured runs:
  w16kv16.tot: template 1300.0 vs measured 1266.4896
```

Fix by editing `template.html`, or bypass with `--no-check-pareto`. The prose
tables are **not** covered by this audit — update them by hand when the runs
change.

## Files

```
analysis/report/
  build_report.py     extract + inject; the only command you need
  template.html       the page, with __CDF__ / __TS__ / __TRACE__ placeholders
  tests/run.sh        extracts the report's chart code and runs the checks below
  tests/harness.js        renders every chart against a stub DOM; asserts
                          curves, panels, markers and crosshair value labels
  tests/harness2.js       exercises every control (scale, clip, smoothing,
                          bin size, arm toggles)
  tests/pareto_check.js   re-derives the Pareto frontier over all 24 metric
                          combinations and checks the claims in the prose
  tests/tooltip_check.js  static CSS check that every tooltip host is
                          positioned (guards a past off-screen-tooltip bug)
```

Editing charts means editing `template.html`, then rebuilding. Do not edit the
generated report directly — the next build overwrites it.
