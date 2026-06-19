# This module reads AC-Core outputs / writes observed evidence.
# It must not implement compiler logic.
"""
Shape grids for kernel microbenchmarks (§9.1).

These describe WHICH measurements a kernel benchmark should collect,
not what they will be. The shapes are intentionally small "default
grids" — enough to be useful, small enough to dry-run.

Adding a new kernel = add an entry here + add it to KERNEL_CATEGORIES.
"""
from __future__ import annotations

from typing import Any

KERNEL_CATEGORIES: tuple[str, ...] = (
    "gemm_bf16",
    "gemm_fp8",
    "attention_prefill",
    "attention_decode",
    "kv_read_bw",
    "kv_quant_dequant",
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "moe_all_to_all",
    "moe_dispatch",
    "state_scan",
)

_DEFAULT_GRIDS: dict[str, list[dict[str, Any]]] = {
    "gemm_bf16": [
        {"M": 4096, "N": 4096, "K": 4096, "dtype": "bf16"},
        {"M": 8192, "N": 8192, "K": 8192, "dtype": "bf16"},
    ],
    "gemm_fp8": [
        {"M": 4096, "N": 4096, "K": 4096, "dtype": "fp8_e4m3"},
    ],
    "attention_prefill": [
        {"batch": 1, "seq": 8192, "n_heads": 32, "n_kv_heads": 32, "head_dim": 128},
        {"batch": 4, "seq": 2048, "n_heads": 32, "n_kv_heads": 8, "head_dim": 128},
    ],
    "attention_decode": [
        {"batch": 32, "seq": 4096, "n_heads": 32, "n_kv_heads": 8, "head_dim": 128},
        {"batch": 64, "seq": 8192, "n_heads": 32, "n_kv_heads": 8, "head_dim": 128},
    ],
    "kv_read_bw": [
        {"batch": 32, "seq": 8192, "n_kv_heads": 8, "head_dim": 128, "dtype": "bf16"},
        {"batch": 32, "seq": 8192, "n_kv_heads": 8, "head_dim": 128, "dtype": "fp8"},
    ],
    "kv_quant_dequant": [
        {"batch": 32, "seq": 8192, "n_kv_heads": 8, "head_dim": 128, "scheme": "fp8"},
    ],
    "all_reduce": [{"world_size": 8, "elements": 2**20, "dtype": "bf16"}],
    "all_gather": [{"world_size": 8, "elements": 2**20, "dtype": "bf16"}],
    "reduce_scatter": [{"world_size": 8, "elements": 2**20, "dtype": "bf16"}],
    "moe_all_to_all": [
        {"world_size": 8, "experts": 8, "tokens_per_rank": 4096, "dtype": "bf16"},
        {"world_size": 16, "experts": 8, "tokens_per_rank": 2048, "dtype": "bf16"},
    ],
    "moe_dispatch": [{"experts": 8, "tokens": 4096, "topk": 2}],
    "state_scan": [{"batch": 8, "seq": 16384, "state_size": 16, "dtype": "bf16"}],
}

# Metric set a sane kernel benchmark should emit.
_DEFAULT_METRICS: dict[str, list[str]] = {
    "gemm_bf16": ["throughput_tflops", "achieved_efficiency"],
    "gemm_fp8": ["throughput_tflops", "achieved_efficiency"],
    "attention_prefill": ["latency_ms", "throughput_tflops"],
    "attention_decode": ["latency_ms", "bandwidth_gbps"],
    "kv_read_bw": ["bandwidth_gbps", "decode_kv_bandwidth_gbps"],
    "kv_quant_dequant": ["latency_ms", "bandwidth_gbps"],
    "all_reduce": ["bandwidth_gbps", "achieved_efficiency"],
    "all_gather": ["bandwidth_gbps", "achieved_efficiency"],
    "reduce_scatter": ["bandwidth_gbps", "achieved_efficiency"],
    "moe_all_to_all": ["bandwidth_gbps", "achieved_efficiency"],
    "moe_dispatch": ["latency_ms", "load_imbalance"],
    "state_scan": ["throughput_tps", "latency_ms"],
}


def default_shapes(kernel: str) -> list[dict[str, Any]]:
    return list(_DEFAULT_GRIDS.get(kernel, []))


def default_metrics(kernel: str) -> list[str]:
    return list(_DEFAULT_METRICS.get(kernel, []))
