# vLLM 0.27.0 smoke environment — 2026-08-15 UTC

- Python environment: `/home/ducct/repos/vllm/.venv/bin`
- vLLM Python package: `0.27.0`
- `vllm --version`: `0.27.0+precompiled`
- PyTorch: `2.13.0+cu130`
- PyTorch CUDA runtime: `13.0`
- Compiled/Python MoE ABI: PASS; both `topk_softmax` interfaces have seven
  parameters including optional `is_padding`.
- Smoke run: `20260815_smoke_v5_w16kv16`
- Stage A: PASS
- Stage B: PASS after recognizing the v0.27 metric name
  `vllm:kv_cache_usage_perc`.
- Runtime precision: `dtype=torch.bfloat16`, `quantization=None`,
  `kv_cache_dtype=bfloat16`.
- Attention backend: FLASHINFER, decode KV dtype `torch.bfloat16`, SM80.
- KV capacity: 1,533,440 tokens; 15.08 GiB reported KV-cache memory per GPU.
- Post-load HBM: 36,103 MiB used and 4,339 MiB free per GPU.
- Smoke workload: 4/4 successful requests, 401 generated tokens, 5.106 seconds.
- Telemetry: 26 Prometheus snapshots, 222 parsed iteration rows, 430 HBM rows
  across four GPUs, maximum sampled aggregate HBM utilization 13%.
- Plot conversion: PASS for TTFT, TPOT, running requests, KV use, and HBM.
  Waiting/preemption were zero under the intentionally small smoke load.
