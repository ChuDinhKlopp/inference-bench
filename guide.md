# RIVF26 operator handoff

This is the practical runbook for an AI agent taking over the RIVF26 A100
precision-characterization experiments. Read `/home/ducct/repos/inference-bench/AGENTS.md`
in full first; it is authoritative. Then read `README.md` for design details.
This guide records the current state and the shortest safe path to continue.

## 1. Non-negotiable operating rules

- Work only in `/home/ducct/repos/inference-bench/rivf26` for RIVF26 code and
  compact artifacts.
- `rivf26` is its own Git repository. Run Git commands from this directory and
  never use `git add .`.
- Use `/home/ducct/repos/vllm/.venv/bin` for every Python/vLLM command. The
  harness enforces this through `scripts/common/venv.sh`.
- Store high-volume logs and raw telemetry under
  `/run/user/1009/ducct/rivf26`. Repository run directories contain symlinks
  to those bulk paths.
- Use the existing model weights directly. Never download or copy model
  weights:

  - BF16: `/dev/shm/Qwen3.6-35B-A3B`
  - FP8: `/dev/shm/Qwen3.6-35B-A3B-FP8`

- TP is always 4. Never start while any required GPU has another significant
  workload. Never kill another user's process.
- Every full run must pass Stage A preflight and Stage B post-server validation.
- Keep `max-num-batched-tokens=16384`, `max-model-len=65536`, and the
  `FLASHINFER` attention backend fixed across Part 1.
- Keep HBM telemetry, Prometheus telemetry, resource guarding, per-request
  results, plot conversion, and legacy e2e recording enabled. A run without a
  required collector is invalid.
- Preserve failed-run evidence. Never overwrite a run ID or silently delete
  old results.
- Do not start Part 2 profiling until Part 1 is stable and the owner asks for it.

## 2. Current experiment state (2026-08-16 UTC)

No RIVF26 benchmark or vLLM server is currently active.

### Performance mode: complete

The four PubMed/Azure performance arms at `max-num-seqs=256` completed with
status PASS:

| Precision | Run ID | Requests | Preemptions |
|---|---|---:|---:|
| `w16kv16` | `20260816_064441_performance_pubmed_w16kv16_mns256` | 1000/1000 | 319 |
| `w8kv8` | `20260816_081708_performance_pubmed_w8kv8_mns256` | 1000/1000 | 0 |
| `w8kv16` | `20260816_094633_performance_pubmed_w8kv16_mns256` | 1000/1000 | 0 |
| `w16kv8` | `20260816_111414_performance_pubmed_w16kv8_mns256` | 1000/1000 | 0 |

Matrix status:

```text
/run/user/1009/ducct/rivf26/logs/
  20260816_064441_performance_pubmed_mns256_matrix/status.jsonl
```

The four rows are registered in `e2e_metrics_record.csv`. The one-second,
four-precision stacked comparison is available at:

```text
results/part1/performance/comparison_pubmed_mns256_precision_variants/
  stacked_timeline.html
  stacked_timeline.svg
  stacked_timeline.png
```

Do not rerun performance unless the owner explicitly requests it. Offline
ROUGE scoring has not yet been generated for the MNS=256 runs; that is a safe
next analysis action.

### Accuracy mode: pilot complete, official MNS pending owner choice

The BF16 GPQA length pilot completed:

```text
run_id: 20260816_140024_accuracy_gpqa_length_pilot_w16kv16_mns24
requests: 198/198 successful, one request per GPQA Diamond question
accuracy: 145/198 = 0.7323
```

Important length results from its `summary.json`:

| Tokens | Average | P50 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| ISL | 251.91 | 218.50 | 376.50 | 460.00 | 788.39 | 2780 |
| OSL | 17803.62 | 17278 | 32768 | 32768 | 32768 | 32768 |
| ISL + OSL | 18055.53 | 17524 | 33026.30 | 33093.75 | 33300.66 | 33553 |

The high OSL percentiles show that many requests reached the 32,768-token cap.
Do not reduce the cap or add a low-reasoning budget: official GPQA is high
reasoning by design.

The pilot observed 1,648,128 logical BF16 KV-cache token slots. Its summary
contains theoretical concurrency bounds for every precision. The conservative
planning reference (`90% of capacity / P90 total length`) is:

| Precision | Planning reference |
|---|---:|
| `w16kv16` | 44 sequences |
| `w8kv16` | 61 sequences |
| `w16kv8` | 77 sequences |
| `w8kv8` | 118 sequences |

These are capacity ratios, not measured scheduler optima. The owner requested
one selected MNS instead of the old 24/48/96 sweep. **Do not choose it silently.**
Present the pilot data to the owner and obtain the selected MNS. Then use that
same value for all four official precision arms.

The earlier run `20260816_134943_accuracy_gpqa_length_pilot_w16kv16_mns24`
failed before request submission because argparse abbreviated `--dataset` as
an adapter option. Commit `3e4b6ea` fixed this with `allow_abbrev=False` and a
regression test. Do not resume or register the failed run.

## 3. Precision mapping

| Variant | Model directory | Weight runtime | KV dtype | vLLM quantization |
|---|---|---|---|---|
| `w16kv16` | BF16 path | BF16 | `bfloat16` | none |
| `w8kv16` | FP8 path | FP8 E4M3 weight-only Marlin | `bfloat16` | `--quantization fp8` |
| `w8kv8` | FP8 path | FP8 E4M3 weight-only Marlin | `fp8` | `--quantization fp8` |
| `w16kv8` | BF16 path | BF16 | `fp8` | none |

KV8 uses vLLM's intended default KV scale 1.0. This was explicitly accepted by
the owner. Do not enable runtime KV-scale calculation for normal Part 1 runs.
Runtime logs and Stage B—not filenames alone—are the precision evidence.

The validated four-precision scheduler smoke gate is:

```text
manifests/smoke_matrix_mbt16384_20260816.json
```

Every integrated workload wrapper refuses to launch unless this gate is PASS
for all four precisions at MBT=16384.

## 4. Workload contract

| Mode | Dataset | Requests | Arrival | Thinking | Max output |
|---|---|---:|---|---|---:|
| Performance | PubMed test set attached to selected Azure trace | 1000 | Azure | enabled, budget 6144 | 10240 |
| Accuracy pilot | GPQA Diamond | 198 × 1 | none | high/unbounded within cap | 32768 |
| Official accuracy | GPQA Diamond | 198 × 5 = 990 | none | high/unbounded within cap | 32768 |

Performance uses the frozen files:

```text
traces/processed/azure_multimodal_bursty_1000.csv
/run/user/1009/ducct/rivf26/datasets/processed/pubmed_azure_bursty_1000.jsonl
```

The trace is a reproducibly selected high-load, high-inter-arrival-CV window.
The workload attaches 1,000 intact PubMed `document/test` prompts without
changing the Azure arrival offsets.

GPQA uses the canonical gated local CSV at revision
`633f5ee89ab8ad4522a9f850766b73f62147ffdd`, split `train`, with SHA-256
`41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305`.
The wrapper validates revision, hash, schema, and exactly 198 rows offline.

## 5. Start every session here

```bash
cd /home/ducct/repos/inference-bench/rivf26
export RIVF26_ROOT="$PWD"
export RIVF26_VENV_BIN=/home/ducct/repos/vllm/.venv/bin
export RIVF26_BULK_ROOT=/run/user/1009/ducct/rivf26
export PATH="$RIVF26_VENV_BIN:$PATH"
```

Confirm repository and environment identity:

```bash
git rev-parse --show-toplevel
git status --short
which python
which vllm
python --version
vllm --version
python -c 'import torch, vllm; print(torch.__version__, torch.version.cuda); print(vllm.__version__)'
```

Expected vLLM version is 0.27.0. An earlier ABI mismatch was resolved by that
upgrade. Do not switch environments or install into a different Python.

## 6. Mandatory pre-run inspection

The integrated wrappers run machine-readable preflight checks, but the agent
must inspect the machine before invoking a long run:

```bash
df -h
df -h "$RIVF26_BULK_ROOT"
df -i "$RIVF26_BULK_ROOT"
du -sh "$RIVF26_BULK_ROOT"
df -h /dev/shm
du -sh /dev/shm/Qwen3.6-35B-A3B /dev/shm/Qwen3.6-35B-A3B-FP8
free -h
ps aux --sort=-%mem | head
nvidia-smi
nvidia-smi topo -m
nvidia-smi pmon -c 1
ss -ltnp | rg ':8000\b' || true
pgrep -af 'vllm|EngineCore|run_(gpqa|pubmed)|nsys profile' || true
```

Do not launch unless all four A100s are idle and have approximately their full
40 GiB HBM free. Treat output filesystem, host RAM, `/dev/shm`, and GPU HBM as
four independent constraints. The default long-run preflight requires 80 GiB
estimated output plus a 50 GiB reserve. Do not lower this merely to force a
launch without evidence.

## 7. Dry runs and tests

Run the test suite after changing harness code:

```bash
"$RIVF26_VENV_BIN/python" -m unittest discover -s tests -p 'test_*.py' -v
bash tests/test_workload_env.sh
```

Inspect one precision command without allocating GPUs:

```bash
RIVF26_DRY_RUN=1 \
scripts/performance/run_trace_azure_pubmed_Qwen3.6-35B-A3B_w16kv16.sh

RIVF26_DRY_RUN=1 \
RIVF26_MAX_NUM_SEQS=24 \
scripts/accuracy/run_trace_azure_gpqa_Qwen3.6-35B-A3B_w16kv16.sh
```

Dry-run output must show the local `/dev/shm` model path, TP=4, MBT=16384,
the intended weight/KV mapping, and the correct mode-specific token cap.

If a new machine needs renewed four-precision smoke evidence:

```bash
for precision in w16kv16 w8kv16 w8kv8 w16kv8; do
  scripts/utilities/run_smoke.sh "$precision" || break
done
```

Do not replace the existing smoke gate until all four new smokes and their
plot-data path pass.

## 8. Launch commands

### Performance

Performance is already complete on this machine. For a deliberate rerun:

```bash
scripts/performance/run_performance_matrix.sh
```

The driver runs `w16kv16`, `w8kv8`, `w8kv16`, `w16kv8`, stops at the first
failure, and stores matrix events under
`$RIVF26_BULK_ROOT/logs/<matrix-id>/status.jsonl`.

### GPQA length pilot

The pilot is already complete. To reproduce it deliberately:

```bash
scripts/accuracy/run_gpqa_length_pilot_w16kv16.sh
```

It is fixed to one repeat and defaults to MNS=24. It fails finalization unless
all 198 requests succeed.

### Official GPQA matrix after owner selects MNS

First update the pilot row in `configs/part1_matrix.csv` to four official rows,
one per precision, with five repeats, 990 total requests, and the selected MNS.
Keep `configs/workloads.json` at five accuracy repeats.

Preview the exact four commands:

```bash
RIVF26_ACCURACY_MAX_NUM_SEQS=<owner-selected-mns> \
RIVF26_DRY_RUN=1 \
scripts/accuracy/run_accuracy_matrix.sh
```

Then, only after a fresh resource inspection:

```bash
RIVF26_ACCURACY_MAX_NUM_SEQS=<owner-selected-mns> \
scripts/accuracy/run_accuracy_matrix.sh
```

The official driver runs all four precisions at the same MNS and stops on the
first failure. Each arm sends 990 requests with `BENCH_ARRIVAL_RATE=none`.

## 9. Monitoring a live run

Discover recent run IDs:

```bash
find "$RIVF26_BULK_ROOT/results/part1" -mindepth 2 -maxdepth 2 -type d \
  -printf '%T@ %p\n' | sort -n | tail
```

Monitor a GPQA pilot (`total=198`) or official accuracy run (`total=990`):

```bash
mode=accuracy
run_id=<run-id>
total=198
log="$RIVF26_BULK_ROOT/results/part1/$mode/$run_id/logs/client.log"
watch -n 5 "n=\$(tr '\r' '\n' < '$log' 2>/dev/null | rg -o -- '[0-9]+/$total \\[' | tail -n1 | cut -d' ' -f1); echo \"\${n:-0/$total}\""
```

The trailing `[` restriction selects tqdm progress records and avoids mistaking
an accuracy score such as `145/198` for request progress. For performance, set
`mode=performance` and `total=1000`.

Inspect live server and collector state:

```bash
tail -f "$RIVF26_BULK_ROOT/results/part1/$mode/$run_id/logs/server.log"
tail -f "$RIVF26_BULK_ROOT/results/part1/$mode/$run_id/logs/resource_guard.log"
nvidia-smi
curl -fsS http://127.0.0.1:8000/metrics | rg \
  'num_requests_(running|waiting)|kv_cache_usage|num_preemptions'
```

Matrix progress is in its status JSONL, not the client log:

```bash
tail -f "$RIVF26_BULK_ROOT/logs/<matrix-id>/status.jsonl"
```

## 10. Artifact and data flow

For run ID `<id>`:

```text
Repository compact artifacts:
  results/part1/<mode>/<id>/
    manifest.json
    summary.json
    plot_data.json
    client_command.txt
    logs -> /run/user/1009/ducct/rivf26/.../logs
    raw  -> /run/user/1009/ducct/rivf26/.../raw

Preflight and Stage B evidence:
  manifests/<id>/preflight.json
  manifests/<id>/post_server.json
  manifests/<id>/snapshots/

High-volume source of truth:
  /run/user/1009/ducct/rivf26/results/part1/<mode>/<id>/logs/
  /run/user/1009/ducct/rivf26/results/part1/<mode>/<id>/raw/
```

Accuracy raw files use `bench_results.*` plus `generations.json`. Performance
raw files use `responses.jsonl`, `per_request.csv`, and `iteration_metrics.*`.
Both modes include `hbm.nsys-rep`, parsed `hbm.csv`, resource-guard samples,
server logs, and client logs.

The deterministic analysis path is:

```text
raw per-request + Prometheus + HBM telemetry
  -> analysis/build_plot_data.py
  -> compact plot_data.json (default 1-second bins)
  -> analysis/plot_stacked_timeline.py
  -> SVG / HTML / PNG
```

Raw sampling is 5 Hz for Prometheus scheduler/KV metrics and 10 Hz for A100 HBM
metrics. One-second plot bins retain approximately five and ten raw samples,
respectively. Override plot bin width with `RIVF26_PLOT_BIN_SECONDS` without
changing raw collection.

Every successful integrated run invokes the unchanged parent
`/home/ducct/repos/inference-bench/record_e2e_metrics.py` and appends to
`e2e_metrics_record.csv`. Never hand-edit a missing row; repair/finalize using
the recorder path.

## 11. Analysis commands

Score every registered PubMed run that lacks ROUGE output:

```bash
nice -n 10 "$RIVF26_VENV_BIN/python" \
  scripts/performance/score_registered_pubmed_runs.py
```

The scorer uses the frozen PubMed references and reports macro ROUGE-1,
ROUGE-2, and sentence-agnostic ROUGE-L F1. It excludes thinking text and joins
strictly by request ID.

Render a single run:

```bash
run_dir=results/part1/accuracy/<run-id>
"$RIVF26_VENV_BIN/python" analysis/plot_stacked_timeline.py \
  "$run_dir/plot_data.json" \
  --output-svg "$run_dir/stacked_timeline.svg" \
  --output-html "$run_dir/stacked_timeline.html" \
  --output-png "$run_dir/stacked_timeline.png"
```

Render a four-precision comparison by passing one plot-data file per variant:

```bash
"$RIVF26_VENV_BIN/python" analysis/plot_stacked_timeline.py \
  results/part1/performance/20260816_064441_performance_pubmed_w16kv16_mns256/plot_data.json \
  results/part1/performance/20260816_094633_performance_pubmed_w8kv16_mns256/plot_data.json \
  results/part1/performance/20260816_111414_performance_pubmed_w16kv8_mns256/plot_data.json \
  results/part1/performance/20260816_081708_performance_pubmed_w8kv8_mns256/plot_data.json \
  --output-svg results/part1/performance/comparison_mns256.svg \
  --output-png results/part1/performance/comparison_mns256.png
```

Comparison colors are stable: blue `w16kv16`, orange `w8kv16`, green
`w16kv8`, purple `w8kv8`. HBM, KV, running, waiting, and cumulative-preemption
panels share elapsed inference time on the x-axis.

## 12. Failure handling

The wrappers trap exit, stop the resource guard and server process group, and
write `failure.json`. On failure:

1. Confirm all vLLM/worker/Nsight processes from that run terminated.
2. Preserve its run directory, bulk logs, preflight, and Stage B evidence.
3. Read `client.log`, the final 100 lines of `server.log`, `failure.json`, and
   `post_server.json` before changing code.
4. Fix and test the root cause.
5. Commit the fix.
6. Rerun with a new timestamped run ID; never reuse the failed ID.

Expected non-fatal messages include Nsight's warning that CPU context-switch
profiling is disabled; Part 1 intentionally records GA100 GPU metrics without
CPU or CUDA kernel tracing. Transformer warnings about undocumented Qwen video
processor frame fields have also appeared during startup and did not block
text inference.

Do not infer that a progress log showing `0/N` means the run is healthy. Check
the process list and `failure.json`: a client can fail before sending request 1.

## 13. Git discipline

Before committing:

```bash
git status --short
git diff --check -- <changed-files>
git diff -- <changed-files>
git add <explicit-file-list>
git diff --cached --check
git diff --cached
git commit -m 'rivf26: <logical milestone>'
```

Do not stage `e2e_metrics_record.csv` accidentally while a run is finalizing,
and do not stage raw logs, Nsight reports, profiler data, response JSONL, model
weights, caches, or editor swap files. Generated compact manifests, summaries,
and plot data may be versioned deliberately after review.

## 14. Immediate next actions

1. Report the completed GPQA pilot length table and concurrency bounds to the
   owner.
2. Obtain one explicit official accuracy `max-num-seqs` value. Do not assume it.
3. Update `configs/part1_matrix.csv` to four official accuracy rows at that MNS.
4. Dry-run `run_accuracy_matrix.sh` and verify four precisions × 990 requests.
5. Recheck disk, RAM, `/dev/shm`, all four GPUs, port 8000, and stale processes.
6. Launch the official four-arm accuracy matrix only after those checks pass.
7. Independently run offline ROUGE scoring for the completed MNS=256 PubMed
   arms; this does not require GPUs.
8. After all four official accuracy arms complete, generate one overlaid
   four-precision stacked timeline rather than separate figures.
