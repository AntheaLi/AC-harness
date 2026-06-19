# This module reads AC-Core outputs / writes observed evidence.
# It must not implement compiler logic.
"""
Needle-in-haystack evaluator (§11).

`plan_mode()` describes the (context_length, needle_depth) grid.
`import_mode()` parses results where each row reports
`needle_accuracy` for a (context_length, depth) cell and emits one
Measurement per cell.
"""
from __future__ import annotations

from typing import Any

from ..schemas import Measurement
from ..store import EvidenceStore, new_id
from ..store.provenance import make_provenance


METRIC_NAME = "needle_accuracy"


def plan_mode(
    *,
    candidate_ids: list[str],
    context_lengths: list[int] | None = None,
    depths: list[float] | None = None,
    notes: str = "AC-Harness needle-in-haystack eval plan.",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "evaluator": "needle",
        "candidate_ids": list(candidate_ids),
        "context_lengths": list(context_lengths or [4096, 16384, 65536, 131072]),
        "depths": list(depths or [0.0, 0.25, 0.5, 0.75, 1.0]),
        "metrics": [METRIC_NAME],
        "notes": notes,
    }


def import_mode(
    store: EvidenceStore,
    *,
    result_payload: dict[str, Any],
    plan_id: str | None = None,
    source_path: str | None = None,
) -> list[str]:
    out: list[str] = []
    for r in result_payload.get("results", []):
        cid = r.get("candidate_id")
        if cid is None or "needle_accuracy" not in r:
            continue
        m = Measurement(
            id=new_id("meas"),
            candidate_id=cid,
            experiment_id=plan_id,
            measurement_type="eval",
            metric_name=METRIC_NAME,
            metric_value=float(r["needle_accuracy"]),
            extra={
                "context_length": r.get("context_length"),
                "depth": r.get("depth"),
            },
            seed=r.get("seed"),
            provenance=make_provenance(
                source_path=source_path,
                command="evaluator.needle.import_mode",
            ),
        )
        store.insert_measurement(m)
        out.append(m.id)
    return out
