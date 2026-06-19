# This module reads AC-Core outputs / writes observed evidence.
# It must not implement compiler logic.
"""
lm-evaluation-harness → AC-Harness adapter.

Run lm-eval-harness against any HF model:

    lm_eval --model hf \\
            --model_args pretrained=meta-llama/Meta-Llama-3-8B \\
            --tasks mmlu,arc_challenge,piqa,winogrande \\
            --batch_size 4 \\
            --output_path /tmp/lm_eval_out/llama3_8b/

then convert the result:

    python -m ac_harness.adapters.lm_eval \\
        --candidate-id llama3_8b \\
        --input /tmp/lm_eval_out/llama3_8b/results_*.json \\
        --out   /tmp/llama3_8b_quality.json

then ingest:

    ach import-results \\
        --plan-id quality_smoke \\
        --input /tmp/llama3_8b_quality.json \\
        --store runs/lab.sqlite

To batch multiple candidates into one import payload, call `convert()`
once per candidate from your own script and write the combined payload
yourself, or run the script per candidate and concatenate the `results`
lists.

This adapter also exposes `downstream_score` (= the first available
recognized metric in priority order MMLU > ARC-Challenge > average of
the others). The harness's bucket classifier counts this as
quality_task coverage; the per-task `lmeval_*` metrics remain for
provenance and finer-grained fitting.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


# Priority order for picking the headline `downstream_score`.
DOWNSTREAM_PRIORITY = (
    "mmlu",
    "arc_challenge",
    "hellaswag",
    "piqa",
    "winogrande",
    "commonsense_qa",
    "boolq",
)


def _pick_metric(scores: dict[str, Any]) -> float | None:
    """Pick acc_norm > acc > exact_match for a single lm-eval task block."""
    for key in ("acc_norm,none", "acc,none", "exact_match,none", "acc_norm", "acc"):
        if key in scores and scores[key] is not None:
            try:
                return float(scores[key])
            except (TypeError, ValueError):
                continue
    return None


def convert(
    *,
    candidate_id: str,
    lm_eval_payload: dict[str, Any],
    provenance: str | None = None,
) -> dict[str, Any]:
    """Convert one lm-eval-harness results payload to one result row.

    Args:
        candidate_id: must match an entry in the AC-Core CandidateSet.
        lm_eval_payload: parsed JSON from lm_eval --output_path.
        provenance: optional override for the per-row provenance string.

    Returns:
        A dict shaped like one entry in the harness's `results` list.
    """
    tasks = lm_eval_payload.get("results", {})
    metrics: dict[str, float] = {}
    units: dict[str, str] = {}

    for task_name, scores in tasks.items():
        if not isinstance(scores, dict):
            continue
        val = _pick_metric(scores)
        if val is None:
            continue
        m_name = f"lmeval_{task_name}"
        metrics[m_name] = val
        units[m_name] = "accuracy"

    # Headline quality_task score for bucket coverage.
    for task in DOWNSTREAM_PRIORITY:
        key = f"lmeval_{task}"
        if key in metrics:
            metrics["downstream_score"] = metrics[key]
            units["downstream_score"] = "accuracy"
            break

    return {
        "candidate_id": candidate_id,
        "metrics": metrics,
        "units": units,
        "seed": None,
        "provenance": provenance or "lm-evaluation-harness output",
    }


def build_payload(
    *,
    rows: list[dict[str, Any]],
    hardware_id: str = "unknown",
    runtime: str = "lm_eval_harness",
    workload_id: str = "general",
    benchmark_type: str = "eval",
    notes: str = "",
) -> dict[str, Any]:
    """Wrap a list of converted rows in the import-results envelope."""
    return {
        "benchmark_type": benchmark_type,
        "hardware_id": hardware_id,
        "runtime": runtime,
        "workload_id": workload_id,
        "notes": notes or f"Converted from lm-evaluation-harness ({len(rows)} candidate(s))",
        "results": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="lm-evaluation-harness → AC-Harness adapter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--candidate-id", required=True,
                   help="ID matching a candidate in the AC-Core CandidateSet")
    p.add_argument("--input", required=True,
                   help="Path (or glob) to lm-eval-harness results JSON")
    p.add_argument("--out", required=True, help="Output JSON path")
    p.add_argument("--hardware-id", default="unknown")
    p.add_argument("--runtime", default="lm_eval_harness")
    p.add_argument("--workload-id", default="general")
    p.add_argument("--benchmark-type", default="eval")
    p.add_argument("--notes", default="")
    args = p.parse_args()

    paths = sorted(glob.glob(args.input)) or [args.input]
    rows = []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        rows.append(convert(
            candidate_id=args.candidate_id,
            lm_eval_payload=payload,
            provenance=f"lm-eval-harness: {Path(path).name}",
        ))

    out_payload = build_payload(
        rows=rows,
        hardware_id=args.hardware_id,
        runtime=args.runtime,
        workload_id=args.workload_id,
        benchmark_type=args.benchmark_type,
        notes=args.notes,
    )
    Path(args.out).write_text(json.dumps(out_payload, indent=2))
    print(f"wrote {args.out} ({len(rows)} row(s))")


if __name__ == "__main__":
    main()
