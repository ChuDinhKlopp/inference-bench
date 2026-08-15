"""Smoke-only audit hook for vLLM's runtime KV-scale calculation.

Loaded through PYTHONPATH only when RIVF26_CALCULATE_KV_SCALES=1. Each worker
writes its own compact JSONL file, avoiding cross-process locking. This is not
enabled for measured Part 1 runs.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path


audit_dir_value = os.environ.get("RIVF26_KV_SCALE_AUDIT_DIR")
if audit_dir_value:
    from vllm.model_executor.layers.attention.attention import Attention

    original_calc_kv_scales = Attention.calc_kv_scales

    def audited_calc_kv_scales(self, query, key, value):
        original_calc_kv_scales(self, query, key, value)
        values = {
            "q_scale": float(self._q_scale_float),
            "k_scale": float(self._k_scale_float),
            "v_scale": float(self._v_scale_float),
            "prob_scale": float(self._prob_scale.item()),
        }
        record = {
            "schema_version": "rivf26.kv_scale_audit.v1",
            "timestamp_epoch_s": time.time(),
            "pid": os.getpid(),
            "layer_name": self.layer_name,
            "kv_cache_dtype": self.kv_cache_dtype,
            "query_quant_enabled": self.query_quant is not None,
            "query_dtype": str(query.dtype),
            "key_dtype": str(key.dtype),
            "value_dtype": str(value.dtype),
            **values,
            "all_scales_finite_positive": all(
                math.isfinite(scale) and scale > 0.0 for scale in values.values()
            ),
        }
        audit_dir = Path(audit_dir_value)
        audit_dir.mkdir(parents=True, exist_ok=True)
        output = audit_dir / f"worker_{os.getpid()}.jsonl"
        with output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")

    Attention.calc_kv_scales = audited_calc_kv_scales
