"""
convert_lm_eval_output.py — turn lm-evaluation-harness JSON output into the
shape `ach import-results` consumes.

This shim exists because the fastest way to get real, non-synthetic quality
numbers into AC-Harness is to run lm-eval-harness against any small HF model
(free Colab T4 is enough) and pipe its output through here.

Usage
-----
    # 1) run lm-eval-harness against your model(s)
    lm_eval --model hf \
            --model_args pretrained=meta-llama/Meta-Llama-3-8B \
            --tasks mmlu,arc_challenge,commonsense_qa \
            --output_path /tmp/lm_eval_out/llama3_8b/

    # 2) convert to the harness shape (one candidate per HF model)
    python convert_lm_eval_output.py \
        --candidate-id llama3_8b \
        --input /tmp/lm_eval_out/llama3_8b/results_*.json \
        --out   /tmp/llama3_8b_quality.json

    # 3) ingest
    ach import-results \
        --plan-id manual_quality \
        --input /tmp/llama3_8b_quality.json \
        --store runs/demo.sqlite

Run the script repeatedly for each candidate, then concatenate the `results`
lists into a single JSON before `import-results` to ingest them as one batch.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def convert(candidate_id: str, lm_eval_path: str) -> dict:
    """Read an lm-eval-harness results JSON; return a one-row result block."""
    raw = json.loads(Path(lm_eval_path).read_text())
    # lm-eval-harness emits {"results": {"<task>": {"acc,none": 0.7, "acc_stderr,none": 0.01, ...}}}
    task_blocks = raw.get("results", {})

    metrics: dict[str, float] = {}
    units: dict[str, str] = {}
    for task, scores in task_blocks.items():
        # Prefer normalised accuracy if present, otherwise accuracy.
        for key in ("acc_norm,none", "acc,none", "exact_match,none"):
            if key in scores:
                metric_name = f"lmeval_{task}"
                metrics[metric_name] = float(scores[key])
                units[metric_name] = "accuracy"
                break

    return {
        "candidate_id": candidate_id,
        "metrics": metrics,
        "units": units,
        "seed": 0,
        "provenance": f"lm-evaluation-harness output: {Path(lm_eval_path).name}",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-id", required=True,
                   help="ID matching a candidate in the AC-Core CandidateSet")
    p.add_argument("--input", required=True,
                   help="Path (or glob) to lm-eval-harness results JSON")
    p.add_argument("--out", required=True,
                   help="Output JSON in the shape `ach import-results` consumes")
    p.add_argument("--hardware-id", default="h100_sxm")
    p.add_argument("--runtime", default="lm_eval_harness")
    p.add_argument("--workload-id", default="general_chat")
    p.add_argument("--benchmark-type", default="lm_eval")
    args = p.parse_args()

    paths = sorted(glob.glob(args.input)) or [args.input]
    rows = [convert(args.candidate_id, path) for path in paths]

    payload = {
        "benchmark_type": args.benchmark_type,
        "hardware_id": args.hardware_id,
        "runtime": args.runtime,
        "workload_id": args.workload_id,
        "notes": f"Converted from lm-evaluation-harness output for {args.candidate_id}",
        "results": rows,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out} ({len(rows)} row(s))")


if __name__ == "__main__":
    main()
