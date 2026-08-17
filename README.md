# RIVF 2026: A100 vLLM precision characterization

This directory is the portable experiment package for Part 1 (runtime/KV-cache
characterization) and, later, the separate short-window Part 2 profiler study.
Generated data stays here but large artifacts are ignored by Git.

## Current machine discovery (2026-08-15 UTC)

- Hardware: 4 x NVIDIA A100-PCIE-40GB, TP=4, PCIe topology (no NVLink).
- Local BF16 model: `/dev/shm/Qwen3.6-35B-A3B` (about 67 GiB on disk).
- Local FP8 model: `/dev/shm/Qwen3.6-35B-A3B-FP8` (about 35 GiB on disk).
- The FP8 checkpoint declares block-quantized E4M3 weights with 128x128 blocks
  and dynamic FP8 activations. Runtime evidence on A100/SM80 differs: vLLM
  0.27 selects `MarlinFP8ScaledMMLinearKernel` and the Marlin FP8 MoE backend,
  warns that A100 lacks native FP8 compute, and uses weight-only FP8 compression
  with BF16 activations. Thus `w8` is a checkpoint-format label; the effective
  A100 execution path is FP8 weights/Marlin with BF16 activations.
- `rivf26/` is a standalone Git repository, as requested. Manifests record its
  commit without treating the parent benchmark directory as the Git root.

No script downloads or copies model weights. Server launch forces Hugging Face
offline mode and passes both model and tokenizer as the same explicit local
directory.

## Layout

- `configs/`: precision mapping, SLO metadata, and the complete Part 1 matrix.
- `scripts/servers/`: four thin precision launchers plus their common launcher.
- `scripts/monitoring/`: GA100 HBM telemetry and resource guards.
- `scripts/accuracy/`, `scripts/performance/`: workload wrappers.
- `analysis/`: deterministic raw-to-plot-data conversion.
- `manifests/`: small preflight/run records.
- `results/`: compact run metadata plus links to bulk `raw/` and `logs/` data.

High-volume logs and raw telemetry default to:

```text
/run/user/1009/ducct/rivf26/results/part1/<mode>/<run-id>/
```

Repository run directories contain `logs` and `raw` symlinks to those paths.
Override the location only through `RIVF26_BULK_ROOT`. The current destination
is a volatile RAM-backed tmpfs, so important results must eventually be archived
to durable storage; preflight accounts for both tmpfs capacity and the host RAM
that new output will consume.

## Safety gates

Every server launch requires a machine-generated preflight JSON ending in
`PASS`. Long-run defaults require 150 GiB free on the result filesystem and
substantial resource headroom. Smoke tests use a smaller, explicit output
estimate but still require all four GPUs to be idle and the model files to be
present. A successful server start is followed by a second validation before
requests are released.

Run preflight (from the repository root):

```bash
$HOME/repos/vllm/.venv/bin/python rivf26/scripts/utilities/preflight.py \
  --run-id 20260815_180000_smoke_w16kv16 \
  --mode smoke --precision w16kv16 --max-num-seqs 2 \
  --max-num-batched-tokens 16384 \
  --estimated-output-gib 2 --safety-reserve-gib 20
```

The report is written under `rivf26/manifests/<run-id>/`. A FAIL is intentional:
fix the reported resource or environment issue rather than bypassing it.

All scripts use `$HOME/repos/vllm/.venv/bin` by default. On this machine the
environment currently provides Python 3.13.14, PyTorch 2.13.0+cu130, CUDA 13.0,
and vLLM 0.27.0 (`vllm --version` reports `0.27.0+precompiled`).

The first v0.26.0 server smoke exposed a mismatched precompiled MoE extension.
The v0.27.0 upgrade resolved it: the Python wrapper and compiled
`_moe_C::topk_softmax` now both include `is_padding`, and the ABI preflight
passes. The check remains mandatory to detect future mixed installations.

The first v0.27.0 `w16kv16` TP=4 smoke passed Stage A, model loading, Stage B,
four requests, 10 Hz GA100 HBM capture, Prometheus/iteration collection, and
plot-data conversion. Runtime evidence reported BF16 weights, BF16 KV,
FLASHINFER attention, 1,533,440 KV-cache tokens, and 4.34 GiB free HBM per GPU
after loading. vLLM 0.27 renamed `vllm:gpu_cache_usage_perc` to
`vllm:kv_cache_usage_perc`; the validator accepts both names and records which
one it resolved.

All four precision arms completed the four-request mechanical smoke on
2026-08-15. Each produced per-request results, Prometheus/iteration telemetry,
10 Hz HBM telemetry, plot-ready JSON, and a log-growth estimate. The KV8 runs
resolved the cache to `torch.float8_e4m3fn`, but vLLM warned that proper scaling
is required; `w8kv8` additionally reported missing q scale, KV scale 1.0, and
uncalibrated q/prob scales of 1.0. The study owner subsequently accepted that
vLLM-default scale policy, as recorded below.

The built-in runtime alternative was tested explicitly with
`--calculate-kv-scales` on both `w8kv8` and `w16kv8`. vLLM 0.27 recognizes the
flag and then disables it for Qwen3.6 because this is a hybrid model containing
GDN/recurrent layers: recurrent state is uninitialized in the calibration pass,
so vLLM considers the derived scales unreliable and falls back to 1.0. No
per-layer runtime scale calculation occurred. The study owner subsequently
accepted vLLM's default scale 1.0 as the intended experimental configuration.
Every KV8 Stage B command must include `--accept-fp8-kv-scale-one`; the resulting
manifest retains the warning as non-blocking and records the accepted policy.

## Server launch

The launchers adapt the existing Qwen3-4B harness pattern while keeping common
logic in one place:

```bash
RIVF26_RUN_ID=... RIVF26_MAX_NUM_SEQS=24 \
  rivf26/scripts/servers/run_server_Qwen3.6-35B-A3B_w16kv16.sh
```

Set `RIVF26_DRY_RUN=1` to inspect the exact command without importing vLLM or
starting a process. Current vLLM rejects FP8 KV with `TRITON_ATTN` on SM80, so
all four arms pin `FLASHINFER`; startup logs must confirm this and the effective
KV/weight configuration.

Every RIVF26 server arm pins `max-num-batched-tokens=16384`. The shared
scheduler helper rejects any conflicting `RIVF26_MAX_NUM_BATCHED_TOKENS` value,
and the same value is recorded by preflight, Stage B, summaries, manifests, and
plot metadata.

## HBM telemetry

Part 1 uses Nsight Systems' GA100 GPU Metrics sampler with CUDA tracing and CPU
sampling disabled. It records device-level DRAM read and write throughput at a
default 10 Hz. `parse_nsys_hbm.py` exports a compact CSV with wall-clock and
elapsed timestamps, read/write/aggregate GB/s, and normalized utilization. The
normalization uses each device's `memoryBandwidth` value embedded in the report
(1.5552 TB/s on this machine), not a hard-coded value.

Plot conversion uses one-second timesteps by default. Each scheduler/KV point
aggregates about five Prometheus samples, while each HBM point aggregates about
ten GA100 samples. Override this with
`RIVF26_PLOT_BIN_SECONDS=<seconds>`; raw telemetry remains the source of truth.

This sampler is intentionally separate from Part 2: no kernel trace or PyTorch
Profiler is enabled in Part 1.

## Experiment matrix

`configs/part1_matrix.csv` records four official 198-request GPQA arms at the
owner-selected `max-num-seqs=256`, replacing the original provisional 24/48/96
sweep. Interactive and Server SLOs are two evaluations of
the same performance run unless a later trace-selection study establishes a
scenario-specific arrival policy; they do not duplicate measurements by
default.

The original four mechanical precision smokes completed HBM capture, runtime
precision evidence, plot conversion, and log-growth measurement, and the study
owner accepted vLLM's intended default FP8 KV scale 1.0. Those smokes used
vLLM's prior implicit A100 token budget, however. Pinning
`max-num-batched-tokens=16384` changes the scientific scheduler configuration,
so all four arms must pass a new smoke at 16384 before any full run. Long runs
also remain fail-closed on current Stage A and Stage B resource evidence.

Renew one precision arm at the pinned scheduler budget with:

```bash
scripts/utilities/run_smoke.sh w16kv16
```

The same command accepts `w8kv16`, `w8kv8`, and `w16kv8`. It owns Stage A,
server startup, Stage B, four requests, HBM capture, plot conversion, validation,
and shutdown. Its compact PASS/FAIL summary is stored with the smoke result.

## GPQA accuracy harness

The four integrated launchers under `scripts/accuracy/` reuse the existing
parent `bench.py` for streaming requests, Prometheus/iteration collection,
generation capture, and GPQA answer scoring. They additionally own the server
lifecycle, both safety stages, 10 Hz HBM capture, resource guarding, plot-data
conversion, repeat-level Pass@1 summaries, and clean shutdown.

The canonical gated GPQA Diamond CSV is validated offline by revision, SHA256,
schema, and its exact 198-row count before vLLM starts. The dataset is not
copied into Git. Set `RIVF26_GPQA_CSV` when the canonical CSV is outside its
normal Hugging Face cache location. The adapter also redirects `bench.py`'s
tokenizer lookup from the served logical model name to the matching local model
directory in `/dev/shm`, with `local_files_only=True`, preventing a hidden model
or tokenizer download.

Before choosing the official `max-num-seqs`, run the 198-question BF16
length pilot once (no repeats):

```bash
scripts/accuracy/run_gpqa_length_pilot_w16kv16.sh
```

It keeps the official high-thinking, 32,768-token, sampling, monitoring, and
scoring settings. Its `summary.json` includes ISL, OSL, and ISL+OSL
min/average/p25/p50/p75/p90/p95/p99/max distributions plus theoretical
KV-capacity concurrency ratios for all four precisions using the validated
smoke capacities. Choose the official `max-num-seqs` only after inspecting
these results; the capacity ratios are planning bounds, not measured optima.

Inspect an exact matrix command without starting the server:

```bash
RIVF26_DRY_RUN=1 \
RIVF26_RUN_ID=DRYRUN_accuracy_gpqa_w16kv16_mns256 \
RIVF26_MAX_NUM_SEQS=256 \
scripts/accuracy/run_trace_azure_gpqa_Qwen3.6-35B-A3B_w16kv16.sh
```

Remove `RIVF26_DRY_RUN=1` only when the generated long-run preflight can pass.
Use the corresponding `w8kv16`, `w8kv8`, or `w16kv8` launcher at the selected
MNS 256. Every accuracy request uses high thinking,
`MAX_GEN_TOKS=253952`, model-default sampling temperature 1.0 and top-p 0.95,
top-k 20, and `BENCH_ARRIVAL_RATE=none`. The RIVF adapter injects top-k into the
existing vLLM request object without modifying the parent client. The 198
questions are sampled once for 198 requests at `max-num-seqs=256`;
`summary.json` reports Pass@1 for that pass. After final validation, the
unchanged parent `record_e2e_metrics.py` also appends the run to
`rivf26/e2e_metrics_record.csv` using the original `bench.py` result JSON.

After selecting a concurrency from the pilot, inspect the four exact precision
commands without allocating a GPU:

```bash
RIVF26_ACCURACY_MAX_NUM_SEQS=256 \
RIVF26_DRY_RUN=1 \
scripts/accuracy/run_accuracy_matrix.sh
```

After the GPUs are idle and a fresh resource preflight can pass, launch the
fail-fast matrix with:

```bash
RIVF26_ACCURACY_MAX_NUM_SEQS=256 \
scripts/accuracy/run_accuracy_matrix.sh
```

The driver uses the same selected concurrency for all four precision formats
and logs each start/PASS/FAIL event to
`$RIVF26_BULK_ROOT/logs/<matrix-id>/status.jsonl`. Each individual wrapper
enforces the version-controlled four-precision `max-num-batched-tokens=16384`
smoke gate before starting vLLM.

## PubMed performance trace

`scripts/performance/select_azure_window.py` evaluates every contiguous
1,000-request window in `AzureLMMInferenceTrace_multimodal.csv`. It first keeps
the highest-load decile (the 10% shortest-duration windows), then selects the
window with the largest population coefficient of variation of its 999
inter-arrival times. This prevents an otherwise idle interval containing one
short cluster from winning solely because of a large gap.

The selected trace is `traces/processed/azure_multimodal_bursty_1000.csv`; its
source hash, selection parameters, ranks, and exact source rows are recorded in
`traces/source_metadata/azure_multimodal_bursty_1000.json`. Arrival offsets are
not scaled.

`scripts/performance/prepare_pubmed_azure_workload.py` deterministically binds
the trace rows to the `document/test` split of `ccdv/pubmed-summarization` at a
pinned revision. It takes the first 1,000 intact documents from a seeded
permutation that satisfy prompt tokens + 10,240 output tokens <= 65,536. It
never truncates article text. Oversized source rows are retained in the small
dataset manifest. The frozen workload defaults to
`/run/user/1009/ducct/rivf26/datasets/processed/pubmed_azure_bursty_1000.jsonl`.

The four `scripts/performance/run_trace_azure_pubmed_Qwen3.6-35B-A3B_w*.sh`
wrappers are integrated launchers: each validates the explicit selected-trace
CSV against the frozen prompt-bound workload, runs Stage A, starts its precision
server, runs Stage B, proves the reasoning cap produces a non-empty answer, and
then releases requests with
`BENCH_ARRIVAL_RATE=azure`. HBM capture wraps the same client command; afterward
the launcher parses HBM telemetry, generates `plot_data.json`, records measured
log growth, invokes the repository's unchanged `record_e2e_metrics.py` to append
the legacy summary row to `rivf26/e2e_metrics_record.csv`, writes the run
manifest, and shuts down the server. The bench-compatible recorder input is
retained beside `summary.json`. High-volume artifacts stay under
`/run/user/1009/ducct/rivf26`.

`run_pubmed_trace.py` reuses the existing `bench.py` streaming transport and
Prometheus collection. It releases each request against monotonic time using
the frozen normalized arrival offset, with no client concurrency semaphore;
vLLM's `max-num-seqs` controls running versus waiting requests. The performance
matrix uses `max-num-seqs=256` and all servers use
`max-num-batched-tokens=16384`.

The first mns128 baseline measured 1,525,248 logical KV-token slots and an
average prompt-plus-completion length of 6,140.02 tokens, corresponding to about
248 average-size resident requests. Runtime evidence from the interrupted mns384
baseline showed that its larger CUDA graphs reduced capacity to 1,183,232 KV
tokens; KV utilization reached 99.9% with 200--210 requests running. The mns256
runtime restored capacity to 1,446,912 KV tokens, or about 236 requests at the
observed average length, so its 256 ceiling remains above predicted KV-limited
residency. The reported CUDA-graph memory estimate was not monotonic across
these startups and is not used as the capacity proxy. The selected trace
releases request 256 at 37.378 seconds, so it supplies enough load to reach that
ceiling.

Performance mode keeps Qwen thinking enabled. Because this Qwen checkpoint has
only an on/off template control, the harness uses vLLM 0.27's sampler-level
`thinking_token_budget=6144` with `--reasoning-parser qwen3`. vLLM forces the
`</think>` boundary at that cap, leaving at least 4,096 of the 10,240 completion
tokens for the answer. The launcher's Stage B probe counts token IDs between
the markers and blocks request release unless the count is at most 6,144 and
the answer is non-empty. It also raises the inherited open-file limit to 65,536;
the first invalid run lost seven burst requests at the former 1,024 limit.

Preview any precision arm without allocating GPUs:

```bash
RIVF26_DRY_RUN=1 \
scripts/performance/run_trace_azure_pubmed_Qwen3.6-35B-A3B_w16kv16.sh
```

Remove `RIVF26_DRY_RUN=1` to execute the integrated run after all smoke gates
have passed. Replace the suffix for `w8kv16`, `w8kv8`, or `w16kv8`.

Run all four performance arms sequentially in the predeclared alternating order:

```bash
scripts/performance/run_performance_matrix.sh
```

The matrix stops on the first failed arm and preserves its status JSONL under
`$RIVF26_BULK_ROOT/logs/<matrix-id>/`.

### Offline PubMed ROUGE scoring

Install the pinned scoring dependency into the required vLLM environment once:

```bash
/home/ducct/repos/vllm/.venv/bin/pip install \
  -r "$RIVF26_ROOT/environment/requirements-scoring.txt"
```

Score every completed PubMed run registered in `e2e_metrics_record.csv`:

```bash
nice -n 10 "$RIVF26_VENV_BIN/python" \
  "$RIVF26_ROOT/scripts/performance/score_registered_pubmed_runs.py"
```

The scorer strictly joins all 1,000 successful responses to the frozen test-set
references by `request_id`. It uses `rouge-score==0.1.2` with Porter stemming
and reports macro-mean per-request ROUGE-1, ROUGE-2, and sentence-agnostic
ROUGE-L F1. Each run receives a compact `rouge_summary.json` and an ignored
`rouge_per_request.jsonl` audit artifact. Prediction text is the API's
`generated_text` answer field; reasoning tokens are not included in ROUGE.

## Workload output limits

`MAX_GEN_TOKS` is a client-side per-request output cap. It is deliberately
separate from the server's `RIVF26_MAX_MODEL_LEN`, which limits prompt plus
output context. Workload launchers must source `scripts/common/workload_env.sh`
and call `rivf26_set_workload_env` before constructing client commands:

```bash
# GPQA accuracy: exported MAX_GEN_TOKS=GPQA_MAX_GEN_TOKS=253952
rivf26_set_workload_env accuracy

# PubMed performance: MAX_GEN_TOKS=10240, THINKING_TOKEN_BUDGET=6144
rivf26_set_workload_env performance
```

The helper rejects inherited values from the other mode. The GPQA client must
pass `--gpqa-max-gen-toks "$GPQA_MAX_GEN_TOKS"`; the PubMed request builder must
put `MAX_GEN_TOKS` into every request's `max_tokens` field. Short smoke requests
may use a dedicated smoke-only cap, but that value must never be recorded as a
full accuracy or performance matrix run.

## Stacked runtime timeline

Render any finalized `plot_data.json` as aligned HBM-bandwidth, KV-cache, and
scheduler panels. The x-axis is the deterministic sampled-timestep index; the
subtitle records the seconds represented by each sample.

```bash
run_id=20260816_031714_performance_pubmed_w16kv16_mns128
run_dir="$RIVF26_ROOT/results/part1/performance/$run_id"
"$RIVF26_VENV_BIN/python" "$RIVF26_ROOT/analysis/plot_stacked_timeline.py" \
  "$run_dir/plot_data.json" --run-id "$run_id" \
  --output-svg "$run_dir/stacked_timeline.svg" \
  --output-html "$run_dir/stacked_timeline.html" \
  --output-png "$run_dir/stacked_timeline.png"
```

Pass one plot-data file per precision to overlay variants in every panel. Color
then consistently denotes precision across HBM, KV, running, waiting, and
cumulative-preemption panels; the shared x-axis is elapsed inference time:

```bash
"$RIVF26_VENV_BIN/python" "$RIVF26_ROOT/analysis/plot_stacked_timeline.py" \
  results/part1/performance/*_w16kv16_mns256/plot_data.json \
  results/part1/performance/*_w8kv16_mns256/plot_data.json \
  results/part1/performance/*_w16kv8_mns256/plot_data.json \
  results/part1/performance/*_w8kv8_mns256/plot_data.json \
  --output-svg results/part1/performance/comparison_mns256.svg \
  --output-png results/part1/performance/comparison_mns256.png
```
