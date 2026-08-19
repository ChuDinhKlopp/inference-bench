# RIVF26 report reproduction guide

Operations guide for an agent asked to reproduce
`results/part1/performance/precision_performance_report.html` — the Part 1
four-arm precision comparison, on a machine other than the one that first
built it.

Read `AGENTS.md` and `guide.md` first for research and safety policy. This
guide covers only the report: which data feeds which chart, how to select the
right runs, and what will silently go wrong.

Developer reference for the builder itself is `analysis/report/README.md`.

## 1. Decide which job you are doing

These are different tasks with very different requirements:

| Goal | What you need | Effort |
|---|---|---|
| **View / share** the existing report | the `.html` file, nothing else | copy one file |
| **Rebuild** it from this machine's runs | builder + run artifacts | this guide |

The published HTML has **no runtime dependencies**: no server, no filesystem
access, no network, no Python. Every value is inlined as a JSON literal and the
charts are drawn in the browser at load time. If the request is only "show me
the report elsewhere", copy the file and stop. Do not rebuild to relocate.

## 2. What the charts are built from

Five chart families. Per precision arm, exactly three files are read — about
1.7 MB out of a ~248 MB run directory:

| File | Feeds |
|---|---|
| `<run>/raw/per_request.csv` | TPOT CDF (5.2), TTFT CDF (5.3), ISL/OSL (5.6) |
| `<run>/plot_data.json` | stacked engine timeline (5.4) |
| `<run>/<run_id>_e2e_record_input.json` | audit of the Pareto constants (5.1) |

Plus, once: `traces/processed/azure_multimodal_bursty_1000.csv` — the Azure
arrival-window chart in section 3.

Total input for every chart is about **6.8 MB**. Never copy `raw/` wholesale;
you need one 88 KB CSV from inside it.

Required columns and fields:

```text
per_request.csv   ttft_s, tpot_s, prompt_len, output_tokens   (one row per request)
plot_data.json    TS[<model>|<precision>].bin_s == 0.5
                  TS[...].runs[<run-dir-name>] with keys:
                  cap, kv, hbm, run, wait, pre, arrivals
e2e_record_input  serving_metrics.{total_token_throughput, output_throughput,
                  request_throughput, mean_ttft_ms, mean_tpot_ms,
                  percentiles_ttft_ms, percentiles_tpot_ms}
trace CSV         ARRIVAL_OFFSET_S
```

## 3. Rebuild

```bash
cd "$HOME/repos/inference-bench/rivf26"
python analysis/report/build_report.py
```

No arguments needed when this machine's runs match the expected naming. The
script prints the run directory it selected for each precision, the extracted
series sizes, and the result of the hand-written-constant audit. Then check it:

```bash
analysis/report/tests/run.sh        # needs node
```

Editing the page means editing `analysis/report/template.html` and rebuilding.
Never hand-edit the generated report; the next build overwrites it.

## 4. Selecting run directories

Discovery matches this glob and takes the newest match:

```text
results/part1/performance/*_performance_pubmed_1000_longest_<precision>_mns256_bin500ms
```

for each of `w16kv16`, `w8kv16`, `w16kv8`, `w8kv8`.

**Always read the printed selection before trusting the output.** Two traps
have already been observed:

1. **The glob can match failed runs.** A failed arm still leaves a directory
   behind — sometimes containing only `failure.json`. On the `mi250` branch all
   four `*_1000_longest_*_mns256_bin500ms` directories are failures
   (`exit_code 143`), while the complete runs live under entirely different
   names (`20260818_070537_repro_w16kv16` and siblings). Newest-wins discovery
   would select the wrecks.

2. **Repeated arms.** Where an arm was run twice, the newest wins. That is
   usually right, but confirm it is the run you mean.

Pin any arm explicitly:

```bash
python analysis/report/build_report.py \
  --run w16kv16=results/part1/performance/20260818_070537_repro_w16kv16 \
  --run w8kv16=results/part1/performance/20260818_070546_repro_w8kv16 \
  --run w16kv8=results/part1/performance/20260818_070720_repro_w16kv8 \
  --run w8kv8=results/part1/performance/20260818_070653_repro_w8kv8
```

A run is a valid candidate only if `guide.md` section 14 would call it
complete. At minimum, before selecting it:

```bash
d=results/part1/performance/<run-id>
test -f "$d/plot_data.json" && test -f "$d/summary.json" && ! test -f "$d/failure.json"
python -c "
import json,sys
d=json.load(open('$d/plot_data.json')); e=list(d['TS'].values())[0]
r=list(e['runs'].values())[0]
assert e['bin_s']==0.5, f\"bin_s={e['bin_s']}, builder needs 0.5\"
missing=[k for k in ('cap','kv','hbm','run','wait','pre','arrivals') if not r.get(k)]
assert not missing, f'empty/missing series: {missing}'
print('ok:', len(r['kv']), 'bins, cap', r['cap'])
"
```

Two further constraints the builder assumes:

- **`bin_s` must be 0.5.** Runs binned at 1 s (older `*_mns256` arms without the
  `bin500ms` suffix) are rejected with a clear message. Re-run
  `analysis/build_plot_data.py` at 500 ms rather than editing the builder.
- **The key inside `plot_data.json`'s `runs` map must equal the run directory
  name.** The builder looks up `entry["runs"][run_dir.name]`. If a directory was
  renamed after conversion, this raises `KeyError`; re-run the conversion or
  rename the directory back.

## 5. The blocker you will most likely hit

`raw/` is a **symlink into `$RIVF26_BULK_ROOT`** — typically
`/run/user/<uid>/...`, which is volatile tmpfs and encodes a specific uid and
username. Two consequences:

**It is never in Git.** `.gitignore` excludes `results/**/raw/`, so
`per_request.csv` cannot arrive by pulling a branch. Confirm before planning:

```bash
git ls-tree -r --name-only <branch> | grep per_request.csv
```

On the `mi250` branch this returns only `tests/fixtures/per_request.csv`, which
is test data, not run output.

**A plain copy leaves it dangling.** Dereference when transferring a run:

```bash
rsync -aL <src>/results/part1/performance/<run-id>/ <dst>/.../<run-id>/
```

The builder detects a dangling `raw/` and says so explicitly rather than failing
obscurely:

```text
missing .../raw/per_request.csv  (.../raw -> /run/user/1009/...)
  raw/ is a symlink into RIVF26_BULK_ROOT. If this checkout was
  copied from another machine, re-copy with `cp -L`/`rsync -L`,
  or pass --run <precision>=<dir> for a local run.
```

If `per_request.csv` genuinely cannot be obtained, three of the five chart
families cannot be built and no code change fixes that. Say so plainly rather
than shipping a partial report as if complete.

## 6. What is NOT generated from data

The Pareto chart (5.1) and the numeric tables in the prose are **literal
constants inside `template.html`**. Rebuilding against different runs updates
the CDFs and the timeline while those keep the previous machine's numbers. The
page still renders, which makes the mismatch easy to miss — worse than a crash.

Every build therefore audits the template's `PARETO` constants against the
measured `serving_metrics` and refuses to write on >0.2% drift:

```text
FAIL: template.html's hand-written PARETO constants no longer match the measured runs:
  w16kv16.tot: template 1300.0 vs measured 1266.4896
  w16kv16.tpot.mean: template 999.9 vs measured 313.3
```

Fix by updating the constants in `template.html`. `--no-check-pareto` bypasses
the audit and should be used only when you have already accepted stale values
for a throwaway build.

**The audit does not cover the prose tables** — roughly 150 numeric cells in
section 5.5 and the workload tables in section 3. When the runs change, update
those by hand, or the report will state numbers no longer supported by its own
charts. Cross-check them against `summary.json` and `e2e_record_input.json`.

The narrative text is likewise written for the A100 result (quantization buys
capacity, not speed; w16kv16 Pareto-dominant). On different hardware the
conclusions may invert; do not ship the old prose beside new charts.

## 7. Verify before reporting success

```bash
analysis/report/tests/run.sh
```

Four suites run against the built page: chart rendering under a stub DOM,
every interactive control path, re-derivation of the Pareto frontier over all
24 metric combinations, and a static CSS check that every tooltip host is
positioned. All must pass.

Additionally confirm by inspection:

- the printed run selection is the four arms you intended;
- extracted request counts match `summary.json` (`successful_requests`);
- the arrival chart's chips read 1,000 arrivals over the trace duration;
- the report's stated hardware, vLLM version and attention backend match this
  machine's manifests — these are prose, not generated.

An unchanged rebuild on the original machine reproduces the published file
byte-for-byte; that is the strongest available check that a port is faithful:

```bash
md5sum results/part1/performance/precision_performance_report.html
```

## 8. Checklist

Only claim the report reproduced when all of these hold:

- [ ] `analysis/report/` is present on this branch (it may live only on `main`).
- [ ] Four complete runs identified, one per precision, none with `failure.json`.
- [ ] Each has `plot_data.json` at `bin_s = 0.5` with non-empty series.
- [ ] Each `plot_data.json` `runs` key equals its directory name.
- [ ] Each has `raw/per_request.csv` reachable (symlink resolved).
- [ ] Each has its `*_e2e_record_input.json`.
- [ ] The Azure trace CSV is present and hash-matches `guide.md` section 6.
- [ ] `build_report.py` selected the intended directories (read its output).
- [ ] The `PARETO` audit passed, or the constants were updated.
- [ ] Prose tables and narrative reviewed against the new runs.
- [ ] `tests/run.sh` passed.

If any item fails, report exactly which and stop. Do not publish a report whose
charts and prose describe different runs.

