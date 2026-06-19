# This module reads AC-Core outputs / writes observed evidence.
# It must not implement compiler logic.
"""
Generic training-loop callback → AC-Harness adapter.

Drop this into your training recipe (torchtitan, Megatron-LM, Nemo, an
internal stack — anything with an eval / step hook). It buffers
`val_loss`, downstream-eval scores, and any custom metrics per step,
then writes them as one JSON file per training run that
`ach import-results` consumes.

Two ways to use it.

1) **In-process callback** — call from your training loop's eval hook:

       from ac_harness.adapters.training_callback import TrainingMetricsBuffer

       buf = TrainingMetricsBuffer(
           candidate_id="my_run_42",
           hardware_id="h100_sxm",
           runtime="torchtitan",
           workload_id="pretrain_8b",
           out_path="/runs/my_run_42.metrics.json",
       )

       # in your training loop:
       for step in range(...):
           ...
           if step % eval_every == 0:
               buf.record(step=step, val_loss=val_loss.item(),
                          downstream_score=mmlu_score)

       buf.flush()    # writes the JSON; safe to call multiple times

2) **From a CSV / JSONL log file** the recipe already writes — `convert()`
   takes a list of {step, metric: value} dicts and emits the same shape:

       python -m ac_harness.adapters.training_callback \\
           --candidate-id my_run_42 \\
           --input  /runs/my_run_42.metrics.jsonl \\
           --out    /runs/my_run_42.metrics.json \\
           --hardware-id h100_sxm \\
           --runtime torchtitan

Per training step the adapter writes one Measurement row per (metric,
step) pair. The harness's residual fitter consumes `val_loss` directly;
any `downstream_score` rows count as quality_task coverage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


# Metric names the harness's bucket classifier recognises by default.
_HARNESS_BUCKETS = {
    "val_loss":            "loss",
    "validation_loss":     "loss",
    "downstream_score":    "accuracy",
    "needle_accuracy":     "accuracy",
    "mqar_accuracy":       "accuracy",
}


def convert(
    *,
    candidate_id: str,
    log_rows: list[dict[str, Any]],
    provenance: str | None = None,
) -> list[dict[str, Any]]:
    """Turn a list of {step, <metric>: <value>, ...} dicts into
    per-step result rows.

    Each step becomes one row with `seed=<step>` so the harness can
    keep multiple datapoints per candidate without merging them.
    """
    rows: list[dict[str, Any]] = []
    for entry in log_rows:
        if not isinstance(entry, dict):
            continue
        step = entry.get("step")
        metrics: dict[str, float] = {}
        units: dict[str, str] = {}
        for k, v in entry.items():
            if k in ("step", "wall_time", "tokens_seen"):
                continue
            if v is None:
                continue
            try:
                metrics[k] = float(v)
            except (TypeError, ValueError):
                continue
            units[k] = _HARNESS_BUCKETS.get(k, "value")
        if not metrics:
            continue
        rows.append({
            "candidate_id": candidate_id,
            "metrics": metrics,
            "units": units,
            "seed": int(step) if step is not None else None,
            "provenance": provenance or "training-loop callback",
        })
    return rows


def build_payload(
    *,
    rows: list[dict[str, Any]],
    hardware_id: str = "unknown",
    runtime: str = "training_callback",
    workload_id: str = "pretrain",
    benchmark_type: str = "training",
    notes: str = "",
) -> dict[str, Any]:
    return {
        "benchmark_type": benchmark_type,
        "hardware_id": hardware_id,
        "runtime": runtime,
        "workload_id": workload_id,
        "notes": notes or f"Training-loop callback log ({len(rows)} step row(s))",
        "results": rows,
    }


class TrainingMetricsBuffer:
    """In-process buffer for training callbacks."""

    def __init__(
        self,
        *,
        candidate_id: str,
        out_path: str,
        hardware_id: str = "unknown",
        runtime: str = "training_callback",
        workload_id: str = "pretrain",
        benchmark_type: str = "training",
        notes: str = "",
    ) -> None:
        self.candidate_id = candidate_id
        self.out_path = out_path
        self.hardware_id = hardware_id
        self.runtime = runtime
        self.workload_id = workload_id
        self.benchmark_type = benchmark_type
        self.notes = notes
        self._log: list[dict[str, Any]] = []

    def record(self, *, step: int, **metrics: float) -> None:
        """Record one eval/step snapshot. Metrics with value None are dropped."""
        entry = {"step": int(step)}
        entry.update({k: v for k, v in metrics.items() if v is not None})
        self._log.append(entry)

    def flush(self) -> str:
        """Write the buffered rows to `out_path`; returns the path written."""
        rows = convert(candidate_id=self.candidate_id, log_rows=self._log)
        payload = build_payload(
            rows=rows,
            hardware_id=self.hardware_id,
            runtime=self.runtime,
            workload_id=self.workload_id,
            benchmark_type=self.benchmark_type,
            notes=self.notes,
        )
        Path(self.out_path).write_text(json.dumps(payload, indent=2))
        return self.out_path


def _load_log(path: str) -> list[dict[str, Any]]:
    """Accept either a JSON array, a JSONL file, or a CSV with a header row."""
    p = Path(path)
    text = p.read_text()
    if p.suffix.lower() == ".csv":
        import csv
        from io import StringIO
        rows: list[dict[str, Any]] = []
        for r in csv.DictReader(StringIO(text)):
            entry = {}
            for k, v in r.items():
                if v == "" or v is None:
                    continue
                try:
                    entry[k] = float(v) if k != "step" else int(v)
                except (TypeError, ValueError):
                    entry[k] = v
            rows.append(entry)
        return rows
    # try JSON-array first, fall back to JSONL
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Training-loop callback → AC-Harness adapter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--candidate-id", required=True)
    p.add_argument("--input", required=True,
                   help="Path to training log: .json (array), .jsonl, or .csv")
    p.add_argument("--out", required=True)
    p.add_argument("--hardware-id", default="unknown")
    p.add_argument("--runtime", default="training_callback")
    p.add_argument("--workload-id", default="pretrain")
    p.add_argument("--benchmark-type", default="training")
    p.add_argument("--notes", default="")
    args = p.parse_args()

    log_rows = _load_log(args.input)
    rows = convert(
        candidate_id=args.candidate_id,
        log_rows=log_rows,
        provenance=f"training log: {Path(args.input).name}",
    )
    out_payload = build_payload(
        rows=rows,
        hardware_id=args.hardware_id,
        runtime=args.runtime,
        workload_id=args.workload_id,
        benchmark_type=args.benchmark_type,
        notes=args.notes,
    )
    Path(args.out).write_text(json.dumps(out_payload, indent=2))
    print(f"wrote {args.out} ({len(rows)} step row(s) for {args.candidate_id})")


if __name__ == "__main__":
    main()
