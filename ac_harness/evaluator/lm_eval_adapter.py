# This module reads AC-Core outputs / writes observed evidence.
# It must not implement compiler logic.
"""
lm-eval-harness adapter stub (§11).

`plan_mode()` describes which tasks to run. `import_mode()` ingests the
JSON dump that lm-eval-harness produces (per-task accuracy / loss).

Stays a stub in v1: no execution. The harness only reads / writes
Measurement records.
"""
from __future__ import annotations

from typing import Any

from ..schemas import Measurement
from ..store import EvidenceStore, new_id
from ..store.provenance import make_provenance


def plan_mode(
    *,
    candidate_ids: list[str],
    tasks: list[str] | None = None,
    notes: str = "AC-Harness lm-eval adapter plan.",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "evaluator": "lm_eval",
        "candidate_ids": list(candidate_ids),
        "tasks": list(tasks or ["arc_challenge", "hellaswag", "mmlu"]),
        "notes": notes,
    }


def import_mode(
    store: EvidenceStore,
    *,
    result_payload: dict[str, Any],
    plan_id: str | None = None,
    source_path: str | None = None,
) -> list[str]:
    """Ingest a result payload of shape:
        {"results": [
            {"candidate_id": "...", "task_scores": {"arc_challenge": 0.55, ...}}
        ]}
    Emits one Measurement per (candidate, task) with metric_name=f"lmeval_{task}".
    """
    out: list[str] = []
    for r in result_payload.get("results", []):
        cid = r.get("candidate_id")
        scores = r.get("task_scores") or {}
        if cid is None or not isinstance(scores, dict):
            continue
        for task, val in scores.items():
            if val is None:
                continue
            m = Measurement(
                id=new_id("meas"),
                candidate_id=cid,
                experiment_id=plan_id,
                measurement_type="eval",
                metric_name=f"lmeval_{task}",
                metric_value=float(val),
                seed=r.get("seed"),
                extra={"task": task},
                provenance=make_provenance(
                    source_path=source_path,
                    command="evaluator.lm_eval_adapter.import_mode",
                ),
            )
            store.insert_measurement(m)
            out.append(m.id)
    return out
