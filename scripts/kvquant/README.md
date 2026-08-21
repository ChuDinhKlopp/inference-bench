# Part 3 — KV-cache quantization across attention architectures

Answers: **when does a KV quantization method pay off for a given attention architecture?**

Three architectures with deliberately different KV dependence, three KV dtypes on one
pinned kernel, swept over `max_num_seqs`. Heavy-decode is implemented; heavy-prefill is
still unspecified.

## Quick start

```bash
cd "$HOME/projects/h100_vllm/inference-bench/rivf26"

# ./run_kvquant.sh already exports these two for this machine; set them yourself only
# when calling a wrapper script directly, bypassing run_kvquant.sh (as below), or to
# override. RIVF26_BULK_ROOT defaults to real disk under projects/h100_vllm, not tmpfs --
# unlike the A100 box, there's no tmpfs budget here and the guard wants 120+ GiB free.
export RIVF26_VENV_BIN="$HOME/projects/h100_vllm/vllm/.venv/bin"
export RIVF26_BULK_ROOT="$HOME/projects/h100_vllm/rivf26-bulk"

# one arm, one MNS
RIVF26_MAX_NUM_SEQS=128 scripts/kvquant/run_azure_livecodebench_qwen3_30b_kv4.sh

# the whole ladder -- ./run_kvquant.sh at the repo root is the entry point.
# The three models run as parallel streams, each on its own port and its own
# tensor_parallel_size-wide CUDA_VISIBLE_DEVICES slice (tp=2 here -> GPUs 0,1 / 2,3 / 4,5,
# ports 8000/8001/8002; override the base with RIVF26_PORT_BASE). Within one model's
# stream its kv-dtype arms and the MNS ladder still run one cell at a time, fail-fast per
# stream (KVQ_CONTINUE_ON_FAIL=1 to keep going). Needs 3 x tp GPUs free; it refuses to
# start otherwise.
./run_kvquant.sh --dry-run              # plan only, launches nothing
./run_kvquant.sh                        # 9 arms x 3 MNS = 27 cells across 3 parallel streams
./run_kvquant.sh qwen3_30b              # substring filter: one model, all dtypes
./run_kvquant.sh kv4                    # all models at int4, one stream per model
./run_kvquant.sh --mns 128              # override the ladder
./run_kvquant.sh --include-secondary    # + the 2 TurboQuant arms (33 cells)

# 3 separate terminals instead of 1 invocation: each process only sees its own model, so
# it must be pinned explicitly or all three collide on port 8000 / GPUs 0,1.
#   term 1: ./run_kvquant.sh qwen36_35b  --port 8000 --cuda 0,1
#   term 2: ./run_kvquant.sh gptoss_120b --port 8001 --cuda 2,3
#   term 3: ./run_kvquant.sh qwen3_30b   --port 8002 --cuda 4,5

python scripts/kvquant/gen_matrix.py --plan            # capacity arithmetic for the ladder
```

## The two design decisions

**1. `TRITON_ATTN` is pinned for every primary arm.** Under vLLM's default selection the
backend *changes with the dtype being swept* — on A100: bf16 → `FLASH_ATTN`, fp8 →
`FLASHINFER`, turboquant → `TURBOQUANT`. Any latency delta would then be a kernel effect
confounded with a dtype effect. `TRITON_ATTN` is the only backend that validates for
**bfloat16, fp8 and `int4_per_token_head` across all three models on SM80, SM90 and
gfx90a**, GPT-OSS attention sinks included — so pinning it makes the study single-variable
*and* portable across all three hardware targets without changing kernel.

**2. Model weight precision is never swept.** Each checkpoint runs in its native format;
only `--kv-cache-dtype` varies. GPT-OSS-120B ships MXFP4 and cannot run BF16 weights on
4×40 GB, so weight format differs *between* models but is constant *within* one. Cross-model
comparison is therefore about attention architecture, not about weights.

## Why KV footprint is the real independent variable

| model | attention architecture | growing-KV layers | bf16 | fp8 | int4 |
|---|---|---:|---:|---:|---:|
| Qwen3.6-35B-A3B | hybrid Gated DeltaNet + full attn | **10 / 40** | 20 KB/tok | 10 | 5 |
| GPT-OSS-120B | dense + sink + sliding window | **18 / 36** | 36 KB/tok | 18 | 9 |
| Qwen3-30B-A3B-Thinking-2507 | dense GQA | **48 / 48** | 96 KB/tok | 48 | 24 |

A 4.8× spread. Quantizing KV on Qwen3.6 touches a quarter of its layers; on Qwen3-30B it
touches all of them. Expected leverage ordering:
`Qwen3-30B > GPT-OSS > Qwen3.6`.

## Workload regimes

| id | status | workload | measured ISL | measured OSL | tok/req |
|---|---|---|---:|---:|---:|
| `decode_heavy` | implemented | LiveCodeBench release_v6, Azure bursty trace (1,055 req) | ~602 | ~22,670 | ~23,272 |
| `prefill_heavy` | **not yet specified** | needs a long-context dataset | 32,768 (target) | 256 (target) | ~33,024 |

LiveCodeBench **is** the heavy-decode regime — roughly 37 output tokens per input token,
confirmed on the MI250 arms. Real prompts with their natural length distribution; the
earlier synthetic fixed-length generator was dropped.

`prefill_heavy` has no runner yet. Candidates already supported by the parent `bench.py`:
`mrcr`, `longproc`, `repobench`.

## Files

```
configs/part3_kvquant.json                          study spec
configs/part3_kvquant_matrix.csv                    generated; one row per cell
scripts/kvquant/
  run_azure_livecodebench_common.sh                 the runner; <model_key> <kv_dtype_key>
  run_azure_livecodebench_<model>_<kv>.sh           11 thin wrappers, one per arm
  run_server.sh                                     server launcher; resolves from the spec
  validate_runtime.py                               asserts the pin took effect; fails closed
  gen_matrix.py                                     spec -> matrix; --plan shows the capacity arithmetic

../../run_kvquant.sh                                sweep entry point (repo root)
```

The 11 wrappers mirror `scripts/performance/run_trace_azure_livecodebench_*`:
`qwen36_35b`, `gptoss_120b`, `qwen3_30b` x `kv16`, `kv8`, `kv4`, plus `kvtq4` for the two
Qwen models. Each owns its full lifecycle — preflight guard, server, runtime validation,
`bench.py` under the Azure trace, HBM capture, `plot_data.json`, `e2e_metrics_record.csv`.

It does **not** call `scripts/utilities/preflight.py`: that resolves model paths from
`configs/precision_configs.json`, which only knows the four Qwen3.6 Part 1 arms. The
essential resource guards (free space, port, GPU snapshot) are inlined instead.

## The validation step matters

A dry run proves the CLI string; it does **not** prove the engine used that backend. vLLM
falls back silently when a backend rejects a configuration, and a silent fallback turns the
controlled variable back into a confound — the exact failure the pin exists to prevent. So
every cell parses the server log and **refuses to measure** on mismatch:

```
RUNTIME VALIDATION FAILED -- refusing to measure this cell
```

## Structural exclusions

`TURBOQUANT` supports neither attention sinks nor sliding window, so **GPT-OSS + TurboQuant
has no valid backend on any platform** — not a missing build, an architectural limit.
`run_server.sh` refuses that combination (exit 1) and `gen_matrix.py` never emits it.

TurboQuant is `role: secondary` and opt-in (`--include-secondary`, 6 extra cells) because
it runs a *different kernel* from the primary arms. Read it as a quantization-**scheme**
comparison at 4 bits, never as a same-kernel dtype step.

⚠️ TurboQuant on MI250 is unverified: its full-prefill path guards on
`is_flash_attn_varlen_func_available()`, and `get_flash_attn_version()` returns `None` on
ROCm. Smoke-test before trusting those cells.

## Staging

The matrix is 27 primary cells (9 arms x 3 MNS) and they are long runs. Suggested order:

1. **Stage 0 — capacity probe.** Launch each `model × kv_dtype` once, read
   `GPU KV cache size` from the log, compute `capacity ÷ (ISL+OSL)`. Minutes each, and it
   tells you where to site the `max_num_seqs` ladder.
2. **Stage 1** — `decode_heavy` (LiveCodeBench), all three MNS points: 27 cells.
3. **Stage 2** — optional TurboQuant scheme comparison: 6 cells.
4. **Stage 3** — `prefill_heavy`, once a dataset is chosen.

**Site the ladder before running.** If effective concurrency exceeds `max_num_seqs` for
every point, the scheduler cap binds instead of the cache and the arm cannot convert its
capacity advantage — the comparison is then invalid. The ladder must extend above the
highest predicted effective concurrency of any arm.
