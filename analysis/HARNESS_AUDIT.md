# Existing harness audit

## Files traced

The requested Qwen3-4B family consists of three server scripts and three Azure
LiveCodeBench wrappers:

- `run_server_Qwen3-4B_w16a16kv16.sh`
- `run_server_Qwen3-4B_w8a16kv16.sh`
- `run_server_Qwen3-4B_w16a16kv8.sh`
- their corresponding `run_trace_azure_livecodebench_*.sh` wrappers

There is no Qwen3-4B `w8kv8` script in that family. The older Qwen3.6 scripts
use an obsolete `/home/tuan/...` model path, TP=2, ROCm-oriented environment
defaults, and an 8,192-token context; they are not suitable for this study.

## Existing execution path

The server scripts are monolithic orchestrators: construct `vllm serve`, launch
it in a process group, poll `/health`, run root `bench.py`, then terminate the
process group through traps. The Azure wrappers only export dataset, arrival,
concurrency, and output settings before invoking the server script.

`bench.py` streams the OpenAI-compatible response. `send_epoch_s` is recorded
after acquiring the concurrency semaphore. TTFT is the first streamed token
delay; TPOT is `(latency - TTFT) / (output_tokens - 1)`. Azure timestamps are
sorted, differenced, scaled, and replayed; `arrival_mode=none` queues tasks
without an artificial trace delay.

`serving_iteration_metrics.py` scrapes `/metrics` (normally every 0.2 seconds)
and parses vLLM iteration-detail log lines. It records running/waiting requests,
cumulative preemptions, KV usage, request completions, and iteration token
counters with epoch, ISO, and monotonic elapsed timestamps. The current merged
iteration JSONL repeats a raw log line and a full Prometheus snapshot for every
engine step; previous captures grew to hundreds of MiB for roughly 100-request
runs. RIVF26 keeps the source logs but will emit compact numeric derived rows
instead of duplicating verbose content.

## Precision mapping found

The legacy Qwen3-4B baseline and KV8 arms share the BF16 checkpoint, while its
FP8 arm changes to an FP8 checkpoint without an explicit quantization flag.
That checkpoint uses dynamic FP8 activations, so its `w8a16` filename is
misleading. Previous A100 notes also show a FLASH_ATTN versus FLASHINFER
attention-backend confound.

For Qwen3.6, local config evidence gives the RIVF26 mapping in
`configs/precision_configs.json`. The current vLLM source explicitly rejects
FP8 KV with `TRITON_ATTN` on SM80 and says native support requires SM89+.
FLASH_ATTN likewise requires FA3/SM90 for FP8 KV, while FLASHINFER supports
SM80 and head size 256. Therefore all arms are pinned to FLASHINFER, subject to
runtime smoke validation.

## Plot path found

The root `latency_plots.html` is a static report with two embedded constants:

- `DATA`: aggregate throughput and TTFT/TPOT/ITL distribution rows.
- `TS`: 15-second arrays named `thr`, `kv`, `run`, `wait`, and `pre`.

It does not load raw logs, and no deterministic generator for those embedded
constants was found. The HTML currently draws throughput, KV utilization, and
request/preemption panels; it does not yet draw TTFT/TPOT or HBM time series.
`build_plot_data.py` now produces compatible `DATA`/`TS` objects and adds
`ttft`, `tpot`, `hbm`, `hbm_read`, and `hbm_write`. A final renderer/import path
and visual browser validation remain part of the smoke-test gate.

## Dataset gaps

Root `bench.py` already supports GPQA Diamond and accuracy scoring, but its
existing GPQA wrapper sends 198 requests once rather than five repeats. It has
no `ccdv/pubmed-summarization` loader. RIVF26 workload preparation must add the
repeat identity explicitly and prepare PubMed prompts without changing their
contents, while retaining the existing streaming request and telemetry path.
