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
- The repository directory is not currently inside a usable Git work tree;
  manifests therefore record `git_commit: null` until repository ownership is
  resolved.

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
uncalibrated q/prob scales of 1.0. Therefore KV8 Stage B can pass mechanical
validation while `long_run_eligible` remains false. Full runs stay blocked until
calibrated KV/q/prob scales are available and runtime-verified.

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

## HBM telemetry

Part 1 uses Nsight Systems' GA100 GPU Metrics sampler with CUDA tracing and CPU
sampling disabled. It records device-level DRAM read and write throughput at a
default 10 Hz. `parse_nsys_hbm.py` exports a compact CSV with wall-clock and
elapsed timestamps, read/write/aggregate GB/s, and normalized utilization. The
normalization uses each device's `memoryBandwidth` value embedded in the report
(1.5552 TB/s on this machine), not a hard-coded value.

This sampler is intentionally separate from Part 2: no kernel trace or PyTorch
Profiler is enabled in Part 1.

## Experiment matrix

`configs/part1_matrix.csv` is authoritative. It contains 12 GPQA configurations
(4 precisions x 3 `max-num-seqs` values), each with 198 samples x 5 repeats, and
four PubMed/Azure workloads. Interactive and Server SLOs are two evaluations of
the same performance run unless a later trace-selection study establishes a
scenario-specific arrival policy; they do not duplicate measurements by
default.

Full runs remain disabled until all four precision smoke tests, HBM capture,
runtime precision evidence, browser visualization, log-growth estimation, and
all resource gates pass. In addition, both KV8 arms require calibrated and
runtime-verified KV/q/prob scales.

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
wrappers validate that frozen workload and require a passing Stage B report
before releasing requests. `run_pubmed_trace.py` reuses the existing
`bench.py` streaming transport and Prometheus collection. It releases each
request against monotonic time using the frozen normalized arrival offset,
with no client concurrency semaphore; vLLM's `max-num-seqs` controls running
versus waiting requests.

## Workload output limits

`MAX_GEN_TOKS` is a client-side per-request output cap. It is deliberately
separate from the server's `RIVF26_MAX_MODEL_LEN`, which limits prompt plus
output context. Workload launchers must source `scripts/common/workload_env.sh`
and call `rivf26_set_workload_env` before constructing client commands:

```bash
# GPQA accuracy: exported MAX_GEN_TOKS=GPQA_MAX_GEN_TOKS=32768
rivf26_set_workload_env accuracy

# PubMed performance: exported MAX_GEN_TOKS=PUBMED_MAX_GEN_TOKS=10240
rivf26_set_workload_env performance
```

The helper rejects inherited values from the other mode. The GPQA client must
pass `--gpqa-max-gen-toks "$GPQA_MAX_GEN_TOKS"`; the PubMed request builder must
put `MAX_GEN_TOKS` into every request's `max_tokens` field. Short smoke requests
may use a dedicated smoke-only cap, but that value must never be recorded as a
full accuracy or performance matrix run.
