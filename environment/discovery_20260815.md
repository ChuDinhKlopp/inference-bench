# Discovery snapshot: 2026-08-15 UTC

This is a lightweight human-readable summary. Exact command outputs will be
captured by each preflight under `manifests/<run-id>/snapshots/`.

| Item | Observed value |
|---|---|
| GPU | 4 x NVIDIA A100-PCIE-40GB, 40,960 MiB each |
| GPU state | About 40,442 MiB free per GPU; no compute processes |
| Driver / reported CUDA | 580.82.09 / 13.0 |
| Topology | PCIe; GPU0 crosses SYS to GPU1-3; no NVLink |
| Host RAM | 1.5 TiB total, 469 GiB available |
| Swap | 8 GiB total, 7.7 GiB used |
| `/dev/shm` | 756 GiB total, 47 GiB free (94% used) |
| Repository filesystem | 438 GiB total, 64 GiB free (85% used) |
| BF16 weights | `/dev/shm/Qwen3.6-35B-A3B`, revision `995ad96...` |
| FP8 weights | `/dev/shm/Qwen3.6-35B-A3B-FP8`, revision `95a723d...` |
| vLLM source | `/home/ducct/repos/vllm`, commit `448344c0...` |
| vLLM runtime | Imports: Python 3.13.14, PyTorch 2.11.0+cu130, CUDA 13.0, vLLM 0.26.0; compiled MoE extension API mismatch blocks startup |
| Bulk log filesystem | `/run/user/1009/ducct`, tmpfs, 152 GiB total / 138 GiB free |
| Git | `inference-bench` is not a usable Git work tree |

The repository filesystem still has only about 64 GiB free, so bulk output now
uses `/run/user/1009/ducct`. That destination is volatile tmpfs and consumes
host RAM as it grows. Swap remains nearly exhausted; smoke still requires a new
preflight plus post-load HBM validation.
