# RIVF26 portable experiment guide

This is the bootstrap, validation, and operations guide for an AI agent moving
the RIVF26 harness to another 8×MI250 machine. It assumes the destination has:

```text
~/repos/inference-bench/
~/repos/vllm/.venv/
```

but does not yet have `~/repos/inference-bench/rivf26/`.

The parent `AGENTS.md` remains the authoritative research and safety policy.
Read it before changing or running the harness. This guide restates the
experiment design so that the receiving agent can understand the study without
depending on the source machine's run history.

This is a MI250 port specification. Before launching, the server common
launcher and validators must be configured for `RIVF26_TENSOR_PARALLEL_SIZE=2`
and `RIVF26_ATTENTION_BACKEND=TRITON_ATTN`; the checked-in A100 defaults remain
TP=4/`FLASHINFER` until that port is applied. Do not treat an A100 smoke PASS
as MI250 runtime validation.

## 1. What the experiment is

The research question is:

> How does vLLM inference behavior change when model-weight and KV-cache
> precision change, and which scheduler, KV-cache, latency, or MI250 HBM effects
> explain the performance difference?

Part 1 is the priority. It records complete runtime time series, not only final
throughput. Part 2 is a separate, later PyTorch Profiler study of roughly
200–300 decode steps for quantization/dequantization kernel costs. Never enable
full-run PyTorch profiling during Part 1.

### Fixed hardware and server configuration

| Item | Required value |
|---|---|
| GPUs | 8 × AMD Instinct MI250 (80 GiB each) |
| Tensor parallelism | 2 |
| Model family | Qwen3.6-35B-A3B |
| vLLM | 0.27.0 environment used by this harness |
| Server context limit | 32,768 tokens |
| `max-num-batched-tokens` | 8,192 default; record any deliberate override |
| Attention backend | `TRITON_ATTN` for every Part 1 arm |
| Server endpoint | `127.0.0.1:8000` by default |

`TRITON_ATTN` is pinned for the ROCm MI250 attention path and must be used
consistently across all arms. `max-num-batched-tokens` is configurable, but
must be recorded identically in the server command, preflight, and plot data.

### Precision arms

| Arm | Checkpoint | Effective MI250 weight path | KV-cache dtype | vLLM quantization |
|---|---|---|---|---|
| `w16kv16` | BF16 | BF16 | `bfloat16` | none |
| `w8kv16` | FP8 | FP8 weight-only Marlin, BF16 activations | `bfloat16` | `fp8` |
| `w8kv8` | FP8 | FP8 weight-only Marlin, BF16 activations | `float8_e4m3fn` | `fp8` |
| `w16kv8` | BF16 | BF16 | `float8_e4m3fn` | none |

MI250 has no native FP8 compute path equivalent to newer architectures. The `w8` arms therefore use vLLM's FP8
weight-only Marlin implementation rather than native FP8 tensor-core
execution. Runtime server logs and Stage B validation are the evidence; names
alone are not.

For KV8, vLLM 0.27 disables runtime KV-scale calculation for this hybrid
GDN/recurrent model and uses scale 1.0. The study owner accepted that intended
vLLM policy. KV8 Stage B records the acceptance explicitly. Do not silently
introduce `--calculate-kv-scales` or a different calibration policy.

### Part 1 workloads

| Mode | Workload | Requests | Arrival policy | Reasoning | Output cap |
|---|---|---:|---|---|---:|
| Performance | 1,000 longest PubMed test articles bound to the selected Azure window | 1,000 | replay normalized Azure offsets | thinking enabled, budget 6,144 | 10,240 |
| Accuracy pilot | GPQA Diamond | 198 × 1 | none | high/unbounded within cap | 32,768 |
| Official accuracy | GPQA Diamond | 198 × 1 | none | high/unbounded within cap | 32,768 |

Performance uses `max-num-seqs=256`. The thinking budget forces the
`</think>` boundary by token 6,144 and leaves at least 4,096 tokens for the
answer. Turning thinking off is not the low-reasoning configuration.

Official GPQA reports Pass@1 over one sample per question, for 198 total
requests per precision arm. Sampling is temperature 1.0,
top-p 0.95, top-k 20, and seed 42. `BENCH_ARRIVAL_RATE=none` means there is no
artificial arrival trace.

The original `AGENTS.md` proposed an accuracy sweep at MNS 24/48/96. The later
experiment decision superseded that provisional sweep: first run one
198-question BF16 length pilot, inspect ISL/OSL and KV-capacity ratios, then use
one deliberately selected MNS for all four official precision arms. The owner
selected MNS 256 and one repeat per question on 2026-08-17, superseding the
briefly launched and interrupted MNS128 attempt.

Performance pairs the burstiest selected Azure 1,000-request window with the
1,000 longest intact PubMed prompts satisfying the context constraint. The
frozen workload metadata must record `selection_policy=longest_prompt_tokens`.
For this MI250 guide, regenerate the workload with `--max-model-len 32768`;
the A100 workload generated with a 65,536-token context is not compatible.

The performance run is evaluated against two borrowed comparison SLOs. These
are not official Qwen requirements:

| Scenario | P99 TTFT | TPOT |
|---|---:|---:|
| Interactive | ≤ 2.0 s | ≤ 15 ms |
| Server | ≤ 3.0 s | ≤ 80 ms |

Both SLO evaluations use the same measured performance run; they are not
separate workloads.

### Required Part 1 measurements

Every arm must retain timestamped data for:

- request arrivals and completions;
- per-request TTFT and TPOT;
- running and waiting requests;
- cumulative/event preemptions;
- KV-cache utilization and runtime KV-token capacity;
- MI250 HBM read, write, aggregate throughput, and normalized utilization;
- resource-guard samples, throughput, latency, token counts, and failures.

Prometheus scheduler/KV data defaults to 5 Hz. Use the MI250-compatible ROCm
telemetry collector at the documented sampling rate; the A100 GA100 sampler is
not portable to MI250.
samples all eight GPUs at 10 Hz. Compact plot data defaults to deterministic
one-second bins while raw samples remain the source of truth.

## 2. Harness architecture

The package adapts the existing parent inference harness instead of replacing
it:

```text
precision wrapper
  -> offline dataset/trace validation
  -> Stage A machine preflight
  -> precision-specific vLLM server, TP=2
  -> Stage B runtime precision/HBM/monitor validation
  -> resource guard + HBM collector
  -> parent bench.py transport and vLLM metrics collection
  -> workload-specific finalizer and scoring
  -> parent record_e2e_metrics.py
  -> analysis/build_plot_data.py
  -> analysis/plot_stacked_timeline.py
```

Important directories are:

```text
rivf26/
  configs/                 scientific configuration
  scripts/servers/         four vLLM precision launchers
  scripts/accuracy/        GPQA pilot and official matrix
  scripts/performance/     Azure/PubMed preparation, run, and ROUGE
  scripts/monitoring/      HBM capture/parser and runtime guard
  scripts/utilities/       preflight, Stage B, smoke gates
  traces/processed/        selected 1,000-arrival Azure window
  datasets/metadata/       frozen PubMed workload identity
  analysis/                raw-to-plot conversion and rendering
  manifests/               Stage A/B and small run records
  results/                 compact artifacts and bulk-data symlinks
```

The package requires these parent repository files:

```text
~/repos/inference-bench/bench.py
~/repos/inference-bench/record_e2e_metrics.py
```

Accuracy and performance finalization intentionally call the unchanged parent
recorder. Do not replace it with hand-written CSV manipulation.

A transferred checkout may include compact summaries and
`e2e_metrics_record.csv` rows from the reference host without their ignored raw
data. Treat them as historical records, not proof that the destination has run
those arms. New runs have unique IDs and host-specific Stage A/B manifests;
archive or merge their bulk artifacts deliberately rather than deleting the
reference records.

## 3. Put `rivf26` on the destination

`rivf26` is a standalone Git repository nested under `inference-bench`. Obtain
the approved repository, bundle, or directory from the experiment owner. No
remote URL is embedded in this checkout, so do not guess one.

For a Git source:

```bash
cd "$HOME/repos/inference-bench"
git clone <approved-rivf26-repository-or-bundle> rivf26
cd rivf26
git checkout <approved-commit>
```

For a transferred directory, preserve its `.git/` directory and place it
exactly at:

```text
$HOME/repos/inference-bench/rivf26
```

Then verify the dependency relationship:

```bash
cd "$HOME/repos/inference-bench/rivf26"
test "$(git rev-parse --show-toplevel)" = "$PWD"
test -f ../bench.py
test -f ../record_e2e_metrics.py
git status --short
```

Never run `git add .`: result directories can contain tens of gigabytes or
symlinks to them. Use explicit file lists.

## 4. Initialize the destination environment

Use the vLLM checkout's environment for every command:

```bash
cd "$HOME/repos/inference-bench/rivf26"
export RIVF26_ROOT="$PWD"
export RIVF26_VENV_BIN="$HOME/repos/vllm/.venv/bin"
export PATH="$RIVF26_VENV_BIN:$PATH"
```

Choose the bulk output filesystem explicitly. This example mirrors the source
layout without hard-coding its UID or username:

```bash
export RIVF26_BULK_ROOT="/run/user/$(id -u)/$USER/rivf26"
mkdir -p "$RIVF26_BULK_ROOT"/{datasets/processed,logs,results/part1}
```

`/run/user/...` is often volatile tmpfs: it disappears at reboot and consumes
host RAM. It is acceptable only when it has the required capacity and results
will be archived deliberately. A large local NVMe path is safer; the harness
supports it through the same variable.

### Runtime paths that must not be copied from the A100 host

The A100 machine used `/run/user/1009/ducct/rivf26`. That path is not portable:
the numeric UID and username differ between machines, and `/run/user` may be a
volatile RAM-backed filesystem. Do not place this literal path in commands,
configuration, or manifests. Always derive it or choose persistent storage:

```bash
# Volatile per-user runtime storage (only if capacity is sufficient)
export RIVF26_BULK_ROOT="/run/user/$(id -u)/$(id -un)/rivf26"

# Preferred for large runs: persistent local/NVMe storage
# export RIVF26_BULK_ROOT=/data/rivf26
mkdir -p "$RIVF26_BULK_ROOT"/{datasets/processed,logs,results/part1}
```

The default is defined in `scripts/common/paths.sh` and is also repeated in
`scripts/utilities/preflight.py` and
`scripts/performance/score_registered_pubmed_runs.py`. Set
`RIVF26_BULK_ROOT` before invoking any harness so all logs, raw telemetry,
profiler output, and generated datasets use the MI250 filesystem. Verify the
resolved location with:

```bash
printf 'RIVF26_ROOT=%s\nRIVF26_BULK_ROOT=%s\n' "$RIVF26_ROOT" "$RIVF26_BULK_ROOT"
df -h "$RIVF26_BULK_ROOT"
df -i "$RIVF26_BULK_ROOT"
```

The checked-in scripts also contain A100-specific `/dev/shm` model defaults
and `/dev/shm` capacity checks. Override `RIVF26_BF16_MODEL_PATH` and
`RIVF26_FP8_MODEL_PATH` after locating the MI250 model files; do not copy
weights into `rivf26/`. If the model is not in `/dev/shm`, the preflight's
`Path("/dev/shm")` check must be ported to the MI250 model/storage location
before a long run.

Confirm the software identity:

```bash
test -x "$RIVF26_VENV_BIN/python"
test -x "$RIVF26_VENV_BIN/vllm"
"$RIVF26_VENV_BIN/python" --version
"$RIVF26_VENV_BIN/vllm" --version
"$RIVF26_VENV_BIN/python" -c \
  'import torch,vllm,transformers,flashinfer; print(torch.__version__, torch.version.cuda); print(vllm.__version__)'
command -v nsys
nsys status --environment
```

The validated reference environment is documented in
`environment/vllm_0.27.0_20260815.md`. At minimum, vLLM must be 0.27.0 and its
Python package and compiled extensions must come from the same installation.
The preflight checks the MoE ABI that previously failed under a mixed install.

Install offline scoring support in this same environment if absent:

```bash
"$RIVF26_VENV_BIN/pip" install \
  -r "$RIVF26_ROOT/environment/requirements-scoring.txt"
```

`pandas` and `pyarrow` are needed only to regenerate the Azure selection or
PubMed workload. They are not required when the frozen inputs are transferred.

## 5. Place and verify model weights

The harness never downloads or copies model weights. The expected defaults are:

```text
/dev/shm/Qwen3.6-35B-A3B
/dev/shm/Qwen3.6-35B-A3B-FP8
```

If the destination uses other local directories, export both paths before any
dry run:

```bash
export RIVF26_BF16_MODEL_PATH=/dev/shm/Qwen3.6-35B-A3B
export RIVF26_FP8_MODEL_PATH=/dev/shm/Qwen3.6-35B-A3B-FP8
```

Verify config, tokenizer, and shard indexes locally:

```bash
for model_dir in "$RIVF26_BF16_MODEL_PATH" "$RIVF26_FP8_MODEL_PATH"; do
  test -d "$model_dir"
  test -f "$model_dir/config.json"
  test -f "$model_dir/tokenizer_config.json"
  find "$model_dir" -maxdepth 1 -type f \
    \( -name '*.safetensors' -o -name '*.index.json' \) -printf '%f\n' | head
  du -sh "$model_dir"
done
```

Expected logical revisions are recorded in `configs/precision_configs.json`:

```text
BF16: 995ad96eacd98c81ed38be0c5b274b04031597b0
FP8:  95a723d08a9490559dae23d0cff1d9466213d989
```

If the weights are missing, stop. Never invoke `hf download`,
`snapshot_download`, `git lfs`, or a remote model ID as a fallback. Both model
and tokenizer must resolve from the explicit local path.

Before and after the first startup, record
`du -sh ~/.cache/huggingface 2>/dev/null || true`; unexpected multi-GB growth
invalidates the startup until explained.

## 6. Prepare the two datasets

### GPQA Diamond

GPQA is gated and intentionally absent from Git. Obtain the canonical file
through the authorized dataset process and set:

```bash
export RIVF26_GPQA_CSV=/path/to/gpqa_diamond.csv
sha256sum "$RIVF26_GPQA_CSV"
```

The harness requires:

```text
repository: Idavidrein/gpqa
configuration: gpqa_diamond
split: train
revision: 633f5ee89ab8ad4522a9f850766b73f62147ffdd
rows: 198
SHA-256: 41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305
```

The GPQA wrapper validates hash, schema, and row count before allocating GPUs.
It does not download a replacement.

### Frozen PubMed/Azure workload

The selected Azure arrival CSV is version controlled:

```text
traces/processed/azure_multimodal_bursty_1000.csv
SHA-256: 3b487d345f3ae02d5a4dcfc8d303a060322d6fe03763af638cda2695d11977d0
```

It is a 1,000-request contiguous window selected by:

1. retaining the 10% shortest-duration candidate windows;
2. maximizing population CV of the 999 inter-arrival times;
3. breaking ties by shorter duration, then earlier source position.

Its duration is 303.145 seconds, mean rate is about 3.299 requests/s, and
inter-arrival CV is about 2.182. Arrival offsets are replayed without scaling.

The 36 MB prompt-bound workload is intentionally outside normal Git history.
The preferred path is to transfer this frozen file from approved experiment
storage to:

```text
$RIVF26_BULK_ROOT/datasets/processed/pubmed_azure_bursty_1000_longest.jsonl
```

Verify it before use:

```bash
export RIVF26_PUBMED_WORKLOAD="$RIVF26_BULK_ROOT/datasets/processed/pubmed_azure_bursty_1000_longest.jsonl"
test -f "$RIVF26_PUBMED_WORKLOAD"
sha256sum "$RIVF26_PUBMED_WORKLOAD"
wc -l "$RIVF26_PUBMED_WORKLOAD"
```

Expected identity:

```text
SHA-256: 3e27694e96b8c297bd3e5ad445e370bb991a39e49ca7c0dfd7c32be822947bbd
rows: 1,000
dataset: ccdv/pubmed-summarization, document/test
revision: 6b30a2cae59b11ed77cb19959bffccbbd18e1106
```

If transfer is impossible, the preparation script may explicitly download the
pinned PubMed parquet—never model weights. Install `pandas` and `pyarrow`, run
the preparation while online, verify the output hash above, then return to
offline execution:

```bash
"$RIVF26_VENV_BIN/pip" install pandas pyarrow
"$RIVF26_VENV_BIN/python" scripts/performance/prepare_pubmed_azure_workload.py \
  --trace traces/processed/azure_multimodal_bursty_1000.csv \
  --output-jsonl "$RIVF26_PUBMED_WORKLOAD" \
  --output-metadata "$RIVF26_BULK_ROOT/datasets/processed/pubmed_azure_bursty_1000_longest.metadata.json" \
  --cache-dir "$RIVF26_BULK_ROOT/datasets/cache" \
  --model-path "$RIVF26_BF16_MODEL_PATH" \
  --select-longest-prompts \
  --max-gen-toks 10240 \
  --thinking-token-budget 6144 \
  --max-model-len 32768
sha256sum "$RIVF26_PUBMED_WORKLOAD"
```

A hash mismatch means package, tokenizer, dataset, or selection drift. Do not
run the matrix until it is explained.

The original million-row Azure file is not needed to execute the frozen
workload. It is only needed to reproduce the selection. Its expected SHA-256 is
`eeaba4bae383eeb3724a4fc804ab49f160e918b6dbe356250111bc3ab50d4a95`.

## 7. Validate the package without GPUs

First run all unit and shell tests:

```bash
cd "$RIVF26_ROOT"
"$RIVF26_VENV_BIN/python" -m unittest discover -s tests -p 'test_*.py' -v
bash tests/test_workload_env.sh
```

Then inspect dry-run commands for every precision:

```bash
for precision in w16kv16 w8kv16 w8kv8 w16kv8; do
  RIVF26_DRY_RUN=1 \
    "scripts/performance/run_trace_azure_pubmed_Qwen3.6-35B-A3B_${precision}.sh"
done

for precision in w16kv16 w8kv16 w8kv8 w16kv8; do
  RIVF26_DRY_RUN=1 RIVF26_MAX_NUM_SEQS=24 \
    "scripts/accuracy/run_trace_azure_gpqa_Qwen3.6-35B-A3B_${precision}.sh"
done
```

Each command must show the intended local model/tokenizer path, TP=2,
MBT=8,192 by default, context 32,768, `TRITON_ATTN`, precision mapping, and mode-specific
output cap. Dry run does not prove runtime precision; the smoke matrix does.

## 8. Record fresh host state and run preflight inspection

Do this before every expensive run:

```bash
df -h
df -h "$RIVF26_BULK_ROOT"
df -i "$RIVF26_BULK_ROOT"
du -sh "$RIVF26_BULK_ROOT"
df -h /dev/shm
du -sh "$RIVF26_BF16_MODEL_PATH" "$RIVF26_FP8_MODEL_PATH"
free -h
cat /proc/meminfo | head -n 30
ps aux --sort=-%mem | head
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw \
  --format=csv
nvidia-smi topo -m
nvidia-smi pmon -c 1
ss -ltnp | rg ':8000\b' || true
pgrep -af 'vllm|EngineCore|bench.py|run_(gpqa|pubmed)|nsys profile' || true
```

Treat result-filesystem space, host RAM/swap, `/dev/shm`, and GPU HBM as four
separate resources. Never kill another user's process. TP=2 requires all eight
GPUs to be available.

Long-run defaults estimate 80 GiB output and require another 50 GiB safety
reserve. Do not reduce these thresholds merely to force a run. If bulk output
uses tmpfs, its growth also consumes host RAM.

## 9. Generate a fresh four-precision smoke gate

The repository may contain a smoke manifest from a reference machine. It is
historical evidence only and must not authorize a run on a different host.

Run the integrated smoke matrix on the destination:

```bash
cd "$RIVF26_ROOT"
matrix_id="$(date -u +%Y%m%d_%H%M%S)_smoke_matrix_mbt16384"
RIVF26_SMOKE_MATRIX_ID="$matrix_id" scripts/utilities/run_smoke_matrix.sh \
  | tee "$RIVF26_BULK_ROOT/logs/$matrix_id.log"
export RIVF26_SMOKE_MATRIX="$RIVF26_ROOT/manifests/$matrix_id.json"
```

This sequentially runs `w16kv16`, `w8kv16`, `w8kv8`, and `w16kv8`. Each smoke
owns Stage A, server startup, Stage B, four requests, TTFT/TPOT, Prometheus,
10 Hz eight-GPU MI250 telemetry, plot conversion, log-growth measurement, and
shutdown. The matrix gate is PASS only if all four summaries are complete.

Verify the gate:

```bash
"$RIVF26_VENV_BIN/python" - "$RIVF26_SMOKE_MATRIX" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(json.dumps({"host": d["host"], "status": d["status"], "runs": d["runs"]}, indent=2))
assert d["status"] == "PASS"
assert set(d["runs"]) == {"w16kv16", "w8kv16", "w8kv8", "w16kv8"}
PY
```

Keep `RIVF26_SMOKE_MATRIX` exported for all long-run wrappers. If any smoke
fails, preserve its evidence, fix the cause, and rerun the whole matrix with a
new ID. Do not edit a FAIL gate into PASS.

Smoke KV capacities may differ from the reference machine because CUDA graphs,
software builds, and free HBM affect allocation. Compare the effective runtime
precision and completeness, not exact token-capacity equality.

## 10. Run performance mode

Performance is exactly four precision runs. Because each MI250 serves one
TP=2 replica, two MI250s are used per replica and up to four replicas may run
in parallel on an eight-GPU host. Use distinct ports and bulk-output
directories; never share a server or run directory between replicas.

| Order | Precision | MNS | Requests |
|---:|---|---:|---:|
| 1 | `w16kv16` | 256 | 1,000 |
| 2 | `w8kv8` | 256 | 1,000 |
| 3 | `w8kv16` | 256 | 1,000 |
| 4 | `w16kv8` | 256 | 1,000 |

Preview the matrix inputs and commands:

```bash
test -f "$RIVF26_SMOKE_MATRIX"
test -f "$RIVF26_PUBMED_WORKLOAD"
for precision in w16kv16 w8kv8 w8kv16 w16kv8; do
  RIVF26_DRY_RUN=1 \
    "scripts/performance/run_trace_azure_pubmed_Qwen3.6-35B-A3B_${precision}.sh"
done
```

Do not pass `RIVF26_DRY_RUN=1` to the performance matrix driver itself: its
children would dry-run, but the driver would still write a misleading PASS
matrix-status file.

Perform the manual resource inspection again, then launch:

```bash
scripts/performance/run_performance_matrix.sh
```

The driver is sequential and fail-fast. Each arm independently validates the
frozen 1,000-row workload and trace, runs both safety stages, probes the 6,144
thinking budget for a non-empty answer, raises the open-file soft limit to
32,768, replays Azure offsets, finalizes telemetry, appends through the parent
recorder, and shuts down its server.

Concurrent arms are supported only when each TP=2 replica has an exclusive
GPU pair, port, run ID, and bulk-output directory. Do not share a server or
GPU between arms. Do not alter arrival offsets or add a client concurrency
semaphore; `max-num-seqs=256` controls vLLM running residency while the trace
supplies arrivals.

After all four runs, score PubMed offline:

```bash
nice -n 10 "$RIVF26_VENV_BIN/python" \
  scripts/performance/score_registered_pubmed_runs.py
```

ROUGE uses the answer field only, excluding thinking text, and reports macro
ROUGE-1, ROUGE-2, and sentence-agnostic ROUGE-L F1.

## 10.1 Run LiveCodeBench release v6

LiveCodeBench accuracy uses the pinned `release_v6` lite set (1,055 prompts).
With the MI250 32,768-token context limit, use a reduced completion cap that
fits the context (`max-gen-toks=22,528`) and a 16,384-token thinking budget:

```bash
RIVF26_MAX_NUM_SEQS=256 \
RIVF26_MAX_MODEL_LEN=32768 \
RIVF26_LCB_MAX_GEN_TOKS=22528 \
RIVF26_LCB_NUM_PROMPTS=1055 \
RIVF26_THINKING_TOKEN_BUDGET=16384 \
RIVF26_RUN_ID=$(date -u +%Y%m%d_%H%M%S)_accuracy_livecodebench_release_v6_w8kv8_mns256 \
scripts/accuracy/run_trace_livecodebench_Qwen3.6-35B-A3B_w8kv8.sh
```

Run the same command with each precision wrapper for the four-arm comparison.
The harness records LiveCodeBench pass@1, TTFT/TPOT, scheduler/KV telemetry,
HBM telemetry, and the legacy `e2e_metrics_record.csv` row.

## 11. Run the GPQA pilot and accuracy matrix

First validate the canonical GPQA file and run one BF16 198-request length
pilot:

```bash
test -f "$RIVF26_GPQA_CSV"
scripts/accuracy/run_gpqa_length_pilot_w16kv16.sh
```

Inspect the pilot's compact summary:

```bash
pilot_dir=$(find results/part1/accuracy -maxdepth 1 -type d \
  -name '*accuracy_gpqa_length_pilot_w16kv16_mns24' | sort | tail -n1)
"$RIVF26_VENV_BIN/python" - "$pilot_dir/summary.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("requests", d["total_requests"], "mean_pass_at_1", d["mean_pass_at_1"])
print("accuracy", json.dumps(d["bench_evaluation_metrics"], indent=2))
print("length_analysis", json.dumps(d["length_analysis"], indent=2))
PY
```

Use ISL+OSL percentiles and the destination's observed KV capacity to select
one official `max-num-seqs`. A capacity quotient is a planning bound, not a
measured scheduler optimum. Record the chosen value and rationale in a compact
manifest or experiment note.

Replace the length-pilot row in `configs/part1_matrix.csv` with four official
accuracy rows using MNS 256, one repeat, and 198 total requests.
This keeps the declared experiment matrix consistent with what the driver will
execute.

Preview the four official commands at that one MNS:

```bash
export RIVF26_ACCURACY_MAX_NUM_SEQS=256
RIVF26_DRY_RUN=1 scripts/accuracy/run_accuracy_matrix.sh
```

Confirm the preview says four precisions, one repeat, 198 requests per arm,
`BENCH_ARRIVAL_RATE=none`, high reasoning, and 32,768 maximum output tokens.
After another full resource inspection, run:

```bash
scripts/accuracy/run_accuracy_matrix.sh
```

The matrix order is `w16kv16`, `w8kv8`, `w8kv16`, `w16kv8` to limit monotonic
precision/thermal bias. It is sequential and fail-fast.

## 12. Monitor a live run

Matrix status is stored under:

```text
$RIVF26_BULK_ROOT/logs/<matrix-id>/status.jsonl
```

Find recent runs:

```bash
find "$RIVF26_BULK_ROOT/results/part1" -mindepth 2 -maxdepth 2 -type d \
  -printf '%T@ %p\n' | sort -n | tail
```

Monitor tqdm request progress without confusing a GPQA score such as `145/198`
for client progress:

```bash
mode=accuracy                 # or performance
run_id=<run-id>
total=198                     # official GPQA; 1000 longest-PubMed performance
log="$RIVF26_BULK_ROOT/results/part1/$mode/$run_id/logs/client.log"
watch -n 5 "n=\$(tr '\r' '\n' < '$log' 2>/dev/null | rg -o -- '[0-9]+/$total \\[' | tail -n1 | cut -d' ' -f1); echo \"\${n:-0/$total}\""
```

Inspect server and resource health:

```bash
tail -f "$RIVF26_BULK_ROOT/results/part1/$mode/$run_id/logs/server.log"
tail -f "$RIVF26_BULK_ROOT/results/part1/$mode/$run_id/logs/resource_guard.log"
curl -fsS http://127.0.0.1:8000/metrics | rg \
  'num_requests_(running|waiting)|kv_cache_usage|num_preemptions'
nvidia-smi
```

A `0/N` client log is not proof that a process is running. Also inspect
`pgrep`, `failure.json`, and the final server-log lines.

## 13. Artifacts and plot pipeline

For each run ID, compact files live in Git-visible paths while large data lives
under the configured bulk root:

```text
$RIVF26_ROOT/results/part1/<mode>/<run-id>/
  manifest.json
  summary.json
  plot_data.json
  client_command.txt
  logs -> $RIVF26_BULK_ROOT/.../logs
  raw  -> $RIVF26_BULK_ROOT/.../raw

$RIVF26_ROOT/manifests/<run-id>/
  preflight.json
  preflight.txt
  post_server.json
  snapshots/
```

Raw data includes server/client logs, per-request output, Prometheus samples,
iteration metrics, resource-guard samples, `hbm.nsys-rep`, and parsed
`hbm.csv`. The deterministic conversion is:

```text
raw request + Prometheus + HBM data
  -> analysis/build_plot_data.py
  -> compact plot_data.json
  -> analysis/plot_stacked_timeline.py
  -> HTML/SVG/PNG
```

Render one run:

```bash
run_dir="$RIVF26_ROOT/results/part1/performance/<run-id>"
"$RIVF26_VENV_BIN/python" analysis/plot_stacked_timeline.py \
  "$run_dir/plot_data.json" \
  --output-html "$run_dir/stacked_timeline.html" \
  --output-svg "$run_dir/stacked_timeline.svg" \
  --output-png "$run_dir/stacked_timeline.png"
```

Render all four precisions as lines on the same panels and shared x-axis:

```bash
"$RIVF26_VENV_BIN/python" analysis/plot_stacked_timeline.py \
  results/part1/performance/*_w16kv16_mns256/plot_data.json \
  results/part1/performance/*_w8kv16_mns256/plot_data.json \
  results/part1/performance/*_w16kv8_mns256/plot_data.json \
  results/part1/performance/*_w8kv8_mns256/plot_data.json \
  --output-html results/part1/performance/comparison_mns256.html \
  --output-svg results/part1/performance/comparison_mns256.svg \
  --output-png results/part1/performance/comparison_mns256.png
```

The stacked panels include HBM bandwidth utilization, KV-cache utilization,
running requests, waiting requests, and cumulative preemptions. TTFT/TPOT are
retained in plot data and per-request artifacts.

## 14. Definition of a valid completed run

Do not call a run complete merely because the client exited zero. Verify:

- Stage A `preflight.json` is PASS;
- Stage B `post_server.json` is PASS and `long_run_eligible` is true;
- all expected requests succeeded;
- runtime logs prove intended weights, KV dtype, TP=2, TRITON_ATTN, and MBT;
- HBM report parsed successfully for all eight GPUs;
- Prometheus includes running, waiting, preemption, and KV metrics;
- per-request TTFT/TPOT and token lengths exist;
- `plot_data.json` contains non-empty required series;
- resource guard remained alive and did not cross a threshold;
- the run was appended by `record_e2e_metrics.py` to
  `rivf26/e2e_metrics_record.csv`;
- output size and post-run free capacity are recorded;
- workload-specific scoring is present or explicitly queued offline.

If preemption is naturally zero, that is a result, not a schema failure.

## 15. Failure and recovery

Wrappers trap exit, stop monitors and the vLLM process group, and write
`failure.json`. When an arm fails:

1. verify that its vLLM workers and Nsight process terminated;
2. preserve its result directory, raw logs, Stage A, and Stage B evidence;
3. inspect `failure.json`, `client.log`, `resource_guard.log`, and the final
   server-log lines;
4. fix and test the cause;
5. rerun with a new timestamped run ID;
6. never overwrite or relabel the failed run.

Do not automatically delete earlier runs. Do not let a benchmark continue if
required telemetry or the resource guard dies. Do not lower safety thresholds
without measured evidence and an explicit experimental decision.

## 16. Reproducibility and Git discipline

Before each matrix, capture at least:

```bash
nvidia-smi > "environment/nvidia-smi_$(hostname)_$(date -u +%Y%m%d).txt"
nvidia-smi topo -m > "environment/topology_$(hostname)_$(date -u +%Y%m%d).txt"
"$RIVF26_VENV_BIN/python" -m pip freeze \
  > "environment/pip-freeze_$(hostname)_$(date -u +%Y%m%d).txt"
git rev-parse HEAD
git status --short
```

Review snapshots before versioning them because they may contain machine paths
or unrelated packages. Commit only small reproducibility artifacts with an
explicit list:

```bash
git diff --check -- <changed-files>
git diff -- <changed-files>
git add -- <explicit-file-list>
git diff --cached --check
git diff --cached
git commit -m 'rivf26: <logical milestone>'
```

Never commit model weights, response JSONL, raw logs, Nsight reports, profiler
traces, dataset caches, or bulk telemetry. Never hand-edit aggregate results to
make a run appear complete.

## 17. Fresh-host execution checklist

An agent may launch a long matrix only when every item is true:

- [ ] `rivf26` is at `~/repos/inference-bench/rivf26` with its Git history.
- [ ] Parent `bench.py` and `record_e2e_metrics.py` exist.
- [ ] `$HOME/repos/vllm/.venv/bin/vllm` reports the validated 0.27.0 install.
- [ ] Nsight Systems supports `--gpu-metrics-set=ga100`.
- [ ] Both exact local model directories and tokenizers are present.
- [ ] No model or tokenizer download occurred.
- [ ] GPQA hash/row count and PubMed workload hash/row count pass.
- [ ] Unit tests and all dry runs pass.
- [ ] Filesystem, host RAM, `/dev/shm`, and every GPU pass separately.
- [ ] No stale server, benchmark, Nsight, or port-8000 listener exists.
- [ ] A fresh host-local four-precision smoke matrix is PASS.
- [ ] `RIVF26_SMOKE_MATRIX` points to that new gate.
- [ ] HBM, TTFT/TPOT, scheduler, KV, and plot conversion passed in smoke.
- [ ] Smoke log-growth evidence supports the configured storage reserve.
- [ ] The exact matrix command and current Git commit are recorded.

Only then run performance or official accuracy. Part 2 remains separate until
Part 1 is complete and the experiment owner requests it.

