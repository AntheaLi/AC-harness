# This module reads AC-Core outputs / writes observed evidence.
# It must not implement compiler logic.
"""
vLLM `benchmark_serving.py` → AC-Harness adapter.

Run vLLM's serving benchmark against any model:

    python -m vllm.entrypoints.openai.api_server \\
        --model meta-llama/Meta-Llama-3-8B --tensor-parallel-size 1 &

    python vllm/benchmarks/benchmark_serving.py \\
        --backend vllm \\
        --model meta-llama/Meta-Llama-3-8B \\
        --dataset-name sharegpt \\
        --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \\
        --num-prompts 500 \\
        --request-rate 8 \\
        --save-result --result-filename /tmp/vllm_llama3_8b.json

then convert the result:

    python -m ac_harness.adapters.vllm_serving \\
        --candidate-id llama3_8b \\
        --input /tmp/vllm_llama3_8b.json \\
        --out   /tmp/llama3_8b_throughput.json \\
        --hardware-id h100_sxm

then ingest:

    ach import-results --plan-id throughput_smoke \\
        --input /tmp/llama3_8b_throughput.json \\
        --store runs/lab.sqlite

The adapter extracts the canonical aggregate metrics that
benchmark_serving.py prints at the end of a run:

  - request_throughput  (req/s)              → request_throughput_rps
  - output_throughput   (tok/s, decode only) → throughput_tps
  - total_token_throughput (tok/s, in+out)   → total_token_throughput_tps
  - mean_ttft_ms                             → ttft_ms_p50
  - median_ttft_ms                           → ttft_ms_p50 (preferred if present)
  - p99_ttft_ms                              → ttft_ms_p99
  - mean_tpot_ms                             → tbt_ms_p50
  - median_tpot_ms                           → tbt_ms_p50 (preferred)
  - p99_tpot_ms                              → tbt_ms_p99

`throughput_tps` is the metric the harness's bucket classifier counts
as throughput coverage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


# (vLLM field name, harness metric name, unit, optional float cast)
_FIELD_MAP: tuple[tuple[str, str, str], ...] = (
    ("request_throughput",        "request_throughput_rps",     "req/s"),
    ("output_throughput",         "throughput_tps",             "tok/s"),
    ("total_token_throughput",    "total_token_throughput_tps", "tok/s"),
    ("median_ttft_ms",            "ttft_ms_p50",                "ms"),
    ("p99_ttft_ms",               "ttft_ms_p99",                "ms"),
    ("median_tpot_ms",            "tbt_ms_p50",                 "ms"),
    ("p99_tpot_ms",               "tbt_ms_p99",                 "ms"),
    ("median_itl_ms",             "itl_ms_p50",                 "ms"),
    ("p99_itl_ms",                "itl_ms_p99",                 "ms"),
)
_FALLBACKS: dict[str, str] = {
    # If the median variant is missing, fall back to the mean.
    "ttft_ms_p50": "mean_ttft_ms",
    "tbt_ms_p50": "mean_tpot_ms",
    "itl_ms_p50": "mean_itl_ms",
}


def convert(
    *,
    candidate_id: str,
    vllm_payload: dict[str, Any],
    provenance: str | None = None,
) -> dict[str, Any]:
    """Convert one vLLM benchmark_serving result file into one row."""
    metrics: dict[str, float] = {}
    units: dict[str, str] = {}

    for vllm_key, m_name, unit in _FIELD_MAP:
        val = vllm_payload.get(vllm_key)
        if val is None and m_name in _FALLBACKS:
            val = vllm_payload.get(_FALLBACKS[m_name])
        if val is None:
            continue
        try:
            metrics[m_name] = float(val)
            units[m_name] = unit
        except (TypeError, ValueError):
            continue

    return {
        "candidate_id": candidate_id,
        "metrics": metrics,
        "units": units,
        "seed": None,
        "provenance": provenance or "vLLM benchmark_serving.py output",
    }


def build_payload(
    *,
    rows: list[dict[str, Any]],
    hardware_id: str = "unknown",
    runtime: str = "vllm",
    workload_id: str = "sharegpt",
    benchmark_type: str = "serving_bench",
    notes: str = "",
) -> dict[str, Any]:
    return {
        "benchmark_type": benchmark_type,
        "hardware_id": hardware_id,
        "runtime": runtime,
        "workload_id": workload_id,
        "notes": notes or f"Converted from vLLM benchmark_serving.py ({len(rows)} candidate(s))",
        "results": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="vLLM benchmark_serving.py → AC-Harness adapter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--candidate-id", required=True)
    p.add_argument("--input", required=True,
                   help="Path to vLLM benchmark_serving result JSON")
    p.add_argument("--out", required=True)
    p.add_argument("--hardware-id", default="unknown")
    p.add_argument("--runtime", default="vllm")
    p.add_argument("--workload-id", default="sharegpt")
    p.add_argument("--benchmark-type", default="serving_bench")
    p.add_argument("--notes", default="")
    args = p.parse_args()

    payload = json.loads(Path(args.input).read_text())
    row = convert(
        candidate_id=args.candidate_id,
        vllm_payload=payload,
        provenance=f"vLLM benchmark_serving.py: {Path(args.input).name}",
    )
    out_payload = build_payload(
        rows=[row],
        hardware_id=args.hardware_id,
        runtime=args.runtime,
        workload_id=args.workload_id,
        benchmark_type=args.benchmark_type,
        notes=args.notes,
    )
    Path(args.out).write_text(json.dumps(out_payload, indent=2))
    print(f"wrote {args.out} ({len(row['metrics'])} metric(s) for {args.candidate_id})")


if __name__ == "__main__":
    main()
